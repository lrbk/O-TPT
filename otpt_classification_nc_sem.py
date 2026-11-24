import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np
import csv
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from scipy.stats import wasserstein_distance
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip_otpt_sem import get_coop
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

import ipdb
import math
import pickle

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def ECE_Loss(num_bins, predictions, confidences, correct):
    #ipdb.set_trace()
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_accuracy = [0]*num_bins
    bin_confidence = [0]*num_bins
    bin_num_sample = [0]*num_bins

    for idx in range(len(predictions)):
        #prediction = predictions[idx]
        confidence = confidences[idx]
        bin_idx = -1
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            bin_idx += 1 
            bin_lower = bin_lower.item()
            bin_upper = bin_upper.item()
            #if bin_lower <= confidence and confidence < bin_upper:
            if bin_lower < confidence and confidence <= bin_upper:
                bin_num_sample[bin_idx] += 1
                bin_accuracy[bin_idx] += correct[idx]
                bin_confidence[bin_idx] += confidences[idx]
    
    for idx in range(num_bins):
        if bin_num_sample[idx] != 0:
            bin_accuracy[idx] = bin_accuracy[idx]/bin_num_sample[idx]
            bin_confidence[idx] = bin_confidence[idx]/bin_num_sample[idx]

    ece_loss = 0.0
    for idx in range(num_bins):
        temp_abs = abs(bin_accuracy[idx]-bin_confidence[idx])
        ece_loss += (temp_abs*bin_num_sample[idx])/len(predictions)

    return ece_loss, bin_accuracy, bin_confidence, bin_num_sample


def Calculator(result_dict): 
    
    list_max_confidence = result_dict['max_confidence']
    list_prediction = result_dict['prediction']
    list_label = result_dict['label']

    torch_list_prediction = torch.tensor(list_prediction).int()
    torch_list_label = torch.tensor(list_label).int()

    torch_correct = (torch_list_prediction == torch_list_label)
    list_correct = torch_correct.tolist()

    # Identify incorrect predictions using tensor operations
    incorrect_indices = (torch_list_prediction != torch_list_label)
    torch_max_confidence = torch.tensor(list_max_confidence)

    # Extract confidences for incorrect predictions
    incorrect_confidences = torch_max_confidence[incorrect_indices].tolist()

    ece_data = ECE_Loss(20, list_prediction, list_max_confidence, list_correct)
    acc = sum(list_correct)/len(list_correct)

    print('acc: ', acc*100)
    print('ece: ', ece_data[0]*100)
          
    return acc*100, ece_data[0]*100, ece_data[1], incorrect_confidences



def temperature_grid_metrics(result_dict, temperatures):
    logits = torch.cat(result_dict['logits'], dim=0).float()
    labels = torch.tensor(result_dict['label']).long()
    rows = []
    for temperature in temperatures:
        probs = torch.softmax(logits / float(temperature), dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        correct = (predictions == labels).tolist()
        ece_data = ECE_Loss(
            20,
            predictions.tolist(),
            confidences.tolist(),
            correct,
        )
        acc = sum(correct) / len(correct) * 100.0
        rows.append((float(temperature), acc, ece_data[0] * 100.0, confidences.mean().item() * 100.0))
    return rows


def conf_acc(logits):
      Nb =  logits.shape[0]
      prob,_ = torch.max(logits.softmax(1),dim=1)
      #print("prob value:", prob)
      q_val = torch.ones(Nb).to(device=args.gpu)
      #print("ones values", q_val)
      cosi = torch.nn.CosineSimilarity(dim=0) 
      dw=cosi(prob,q_val)
      #print("similarity:",dw)
      return dw
def select_confident_samples(logits, top):
    #computing the softmax and log-values +summing up
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1) #batch entropy shape [64]
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    #this returns the maximum confidence for the each augmented image and it's index
    return logits[idx], idx


def avg_entropy(outputs):
   
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]  tensor of shape [6, 1]
   
   
    #computing the average logit from the augmented views of the image.
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    #print("avg-logits:", -(avg_logits * torch.exp(avg_logits)).sum(dim=-1))
    #this returns the average entropy for the aumented images
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def build_etf_gram(num_classes, device, dtype):
    if num_classes <= 1:
        return torch.eye(num_classes, device=device, dtype=dtype)
    eye = torch.eye(num_classes, device=device, dtype=dtype)
    ones = torch.ones(num_classes, num_classes, device=device, dtype=dtype) / num_classes
    return (num_classes / (num_classes - 1)) * (eye - ones)


def load_semantic_prior(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pt", ".pth"]:
        prior = torch.load(path, map_location="cpu")
    elif ext in [".pkl", ".pickle"]:
        with open(path, "rb") as fp:
            prior = pickle.load(fp)
    elif ext == ".npy":
        prior = np.load(path)
    elif ext in [".csv", ".tsv", ".txt"]:
        delimiter = "\t" if ext == ".tsv" else ","
        prior = np.loadtxt(path, delimiter=delimiter)
    else:
        raise ValueError("Unsupported --sem_prior_path extension: {}".format(ext))

    if isinstance(prior, dict):
        for key in ["prior", "semantic_prior", "S", "S_prior", "gram", "matrix"]:
            if key in prior:
                prior = prior[key]
                break
        else:
            raise KeyError("Semantic prior dict must contain prior/semantic_prior/S/S_prior/gram/matrix")
    return torch.as_tensor(prior, dtype=torch.float32)


def get_nc_sem_target(model, text_features, args):
    num_classes = text_features.shape[0]
    cache_key = (
        num_classes,
        text_features.device,
        text_features.dtype,
        args.sem_prior_path,
        float(args.sem_alpha),
    )
    if getattr(model, "_nc_sem_target_key", None) == cache_key:
        return model._nc_sem_target

    if args.sem_prior_path:
        sem_gram = load_semantic_prior(args.sem_prior_path)
        sem_gram = sem_gram.to(device=text_features.device, dtype=text_features.dtype)
    else:
        anchor_text = getattr(model, "anchor_textfeatures_", text_features.detach())
        anchor_text = F.normalize(anchor_text.float(), dim=-1)
        sem_gram = anchor_text @ anchor_text.t()
        sem_gram = sem_gram.to(dtype=text_features.dtype)

    if sem_gram.shape != (num_classes, num_classes):
        raise ValueError("Semantic prior shape {} does not match {} classes".format(tuple(sem_gram.shape), num_classes))

    sem_gram = 0.5 * (sem_gram + sem_gram.t())
    sem_gram = sem_gram.clamp(min=-1.0, max=1.0)
    sem_gram.fill_diagonal_(1.0)
    etf_gram = build_etf_gram(num_classes, text_features.device, text_features.dtype)
    alpha = min(max(float(args.sem_alpha), 0.0), 1.0)
    target = alpha * etf_gram + (1.0 - alpha) * sem_gram
    target.fill_diagonal_(1.0)

    model._nc_sem_target = target.detach()
    model._nc_sem_target_key = cache_key
    return model._nc_sem_target


def compute_nc_sem_loss(model, args):
    text_features = F.normalize(model.textfeatures_.float(), dim=-1)
    gram = text_features @ text_features.t()
    target = get_nc_sem_target(model, text_features, args)
    offdiag = ~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
    loss = ((gram[offdiag] - target[offdiag]) ** 2).mean()

    counter = getattr(args, "_nc_sem_log_counter", 0) + 1
    args._nc_sem_log_counter = counter
    if counter <= args.nc_sem_log_limit or (args.nc_sem_log_every > 0 and counter % args.nc_sem_log_every == 0):
        print(
            "[nc_sem] loss={:.6f}, gram_offdiag={:.4f}, target_offdiag={:.4f}, alpha={:.2f}".format(
                loss.item(),
                gram[offdiag].mean().item() if offdiag.any() else 0.0,
                target[offdiag].mean().item() if offdiag.any() else 0.0,
                float(args.sem_alpha),
            )
        )
    return loss


def get_selected_image_features(model, output, args):
    image_features = F.normalize(model.imagefeatures_.float(), dim=-1)
    selected_idx = getattr(args, "_selected_view_idx", None)
    if image_features.shape[0] != output.shape[0] and selected_idx is not None:
        image_features = image_features[selected_idx]
    if image_features.shape[0] != output.shape[0]:
        image_features = image_features[: output.shape[0]]
    return image_features


def compute_mm_ortho_loss(model, output, args):
    text_features = F.normalize(model.textfeatures_.float(), dim=-1)
    image_features = get_selected_image_features(model, output, args)
    probs = output.float().softmax(dim=-1).detach()
    class_mass = probs.sum(dim=0).clamp_min(1e-6)
    visual_proto = F.normalize(probs.t() @ image_features / class_mass.unsqueeze(1), dim=-1)
    target = get_nc_sem_target(model, text_features, args)
    visual_gram = visual_proto @ visual_proto.t()
    cross_gram = text_features @ visual_proto.t()
    offdiag = ~torch.eye(target.shape[0], dtype=torch.bool, device=target.device)
    visual_loss = ((visual_gram[offdiag] - target[offdiag]) ** 2).mean()
    cross_loss = ((cross_gram - target) ** 2).mean()
    return args.lam_mm_visual * visual_loss + args.lam_mm_cross * cross_loss


def compute_cache_ortho_loss(model, args):
    cache = getattr(model, "otpt_cache", None)
    if cache is None or cache["count"].sum() <= 0:
        return model.textfeatures_.new_tensor(0.0)
    text_features = F.normalize(model.textfeatures_.float(), dim=-1)
    valid = cache["count"] > 0
    if valid.sum() < 2:
        return text_features.new_tensor(0.0)
    proto = F.normalize(cache["sum"][valid] / cache["count"][valid].unsqueeze(1).clamp_min(1.0), dim=-1)
    selected_text = text_features[valid]
    logits = selected_text @ proto.t()
    eye = torch.eye(logits.shape[0], device=logits.device, dtype=logits.dtype)
    return ((logits - eye) ** 2).mean()


def update_otpt_cache(model, output, args):
    if not getattr(args, "use_cache_ortho", False):
        return
    image_features = F.normalize(model.imagefeatures_.float(), dim=-1).detach()
    probs = output.float().softmax(dim=-1).detach()
    conf, pred = probs.max(dim=-1)
    num_classes = probs.shape[-1]
    if not hasattr(model, "otpt_cache") or model.otpt_cache["sum"].shape[0] != num_classes:
        model.otpt_cache = {
            "sum": torch.zeros(num_classes, image_features.shape[-1], device=image_features.device),
            "count": torch.zeros(num_classes, device=image_features.device),
        }
    for feat, cls, score in zip(image_features, pred, conf):
        if score >= args.cache_conf:
            model.otpt_cache["sum"][cls] += feat
            model.otpt_cache["count"][cls] += 1.0


def compute_dist_align_loss(model, output, args):
    image_features = get_selected_image_features(model, output, args)
    anchor_text = F.normalize(model.anchor_textfeatures_.float(), dim=-1)
    anchor_logits = model.logit_scale.exp().float() * image_features @ anchor_text.t()
    anchor_probs = anchor_logits.softmax(dim=-1).detach()
    log_probs = output.float().log_softmax(dim=-1)
    return -(anchor_probs * log_probs).sum(dim=-1).mean()


def compute_proda_ortho_loss(model, args):
    image_features = F.normalize(model.imagefeatures_.float(), dim=-1)
    ctx = model.prompt_learner.ctx
    logits = []
    grams = []
    for m in range(args.proda_prompts):
        if m == 0:
            ctx_m = ctx
        else:
            noise = torch.randn_like(ctx) * args.proda_noise_std
            ctx_m = ctx + noise
        text_m = F.normalize(model.get_text_features(ctx_m).float(), dim=-1)
        logits.append(model.logit_scale.exp().float() * image_features @ text_m.t())
        grams.append(text_m @ text_m.t())
    stacked_logits = torch.stack(logits, dim=0)
    ensemble_probs = stacked_logits.softmax(dim=-1).mean(dim=0)
    entropy = -(ensemble_probs * ensemble_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
    diversity = torch.stack([g.flatten() for g in grams], dim=0).std(dim=0).mean()
    return entropy - args.lam_proda_div * diversity


def worst_view_entropy(output):
    probs = output.float().softmax(dim=-1)
    ent = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return ent.max()


def logit_normalize(output, tau):
    return F.normalize(output.float(), dim=-1) / float(tau)


def compute_logitnorm_entropy_loss(output, args):
    return avg_entropy(logit_normalize(output, args.logitnorm_tau))


def compute_entropy_target_loss(model, output, args):
    image_features = get_selected_image_features(model, output, args)
    anchor_text = F.normalize(model.anchor_textfeatures_.float(), dim=-1)
    anchor_logits = model.logit_scale.exp().float() * image_features @ anchor_text.t()
    anchor_probs = anchor_logits.softmax(dim=-1).detach()
    anchor_entropy = -(anchor_probs * anchor_probs.clamp_min(1e-8).log()).sum(dim=-1)
    adapt_probs = output.float().softmax(dim=-1)
    adapt_entropy = -(adapt_probs * adapt_probs.clamp_min(1e-8).log()).sum(dim=-1)
    target_entropy = anchor_entropy + args.entropy_target_margin
    if args.entropy_target_mode == "symmetric":
        return ((adapt_entropy - target_entropy) ** 2).mean()
    return F.relu(target_entropy - adapt_entropy).pow(2).mean()


def compute_trust_region_loss(model, args):
    losses = []
    mode = getattr(args, "trust_region_mode", "ctx")
    if mode in ["ctx", "both"]:
        ctx = model.prompt_learner.ctx.float()
        ctx_init = model.prompt_learner.ctx_init_state.to(device=ctx.device, dtype=ctx.dtype)
        losses.append((ctx - ctx_init).pow(2).mean())

    if mode in ["text", "both"]:
        text_features = F.normalize(model.textfeatures_.float(), dim=-1)
        anchor_text = getattr(model, "anchor_textfeatures_", None)
        if anchor_text is None:
            with torch.no_grad():
                anchor_text = model.get_text_features(model.prompt_learner.ctx_init_state)
        anchor_text = F.normalize(anchor_text.float().detach(), dim=-1)
        losses.append((text_features - anchor_text).pow(2).mean())

    if not losses:
        return model.textfeatures_.new_tensor(0.0)
    return torch.stack(losses).mean()


def add_direction_losses(model, output, base_loss, args):
    if getattr(args, "use_nc_sem", False):
        base_loss = base_loss + args.lam_nc_sem * compute_nc_sem_loss(model, args)
    if getattr(args, "use_mm_ortho", False):
        base_loss = base_loss + compute_mm_ortho_loss(model, output, args)
    if getattr(args, "use_cache_ortho", False):
        base_loss = base_loss + args.lam_cache * compute_cache_ortho_loss(model, args)
    if getattr(args, "use_dist_align", False):
        base_loss = base_loss + args.lam_dist_align * compute_dist_align_loss(model, output, args)
    if getattr(args, "use_proda_ortho", False):
        base_loss = base_loss + args.lam_proda * compute_proda_ortho_loss(model, args)
    if getattr(args, "use_adv_ortho", False):
        base_loss = base_loss + args.lam_adv * worst_view_entropy(output)
    if getattr(args, "use_entropy_target", False):
        base_loss = base_loss + args.lam_entropy_target * compute_entropy_target_loss(model, output, args)
    if getattr(args, "use_logitnorm_tpt", False):
        base_loss = base_loss + args.lam_logitnorm * compute_logitnorm_entropy_loss(output, args)
    if getattr(args, "use_trust_region", False):
        base_loss = base_loss + args.lam_trust_region * compute_trust_region_loss(model, args)
    return base_loss


def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def test_time_tuning(model, inputs, optimizer, scaler, args, cons):
    output = None
    output2 = None
    single_output = None
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    
    selected_idx = None
    for j in range(args.tta_steps):
        if 'tpt' in args.run_type:
            with torch.cuda.amp.autocast():
                if args.cocoop:
                    #it does require prompt to tune
                    output = model((image_feature, pgen_ctx),cons,args)
                else:
                    #it doresn't because it use pre-tuned prompts                   
                    output = model(inputs,cons,args) 
                    #print("Output shape:", output.shape)
                    #Output shape: torch.Size([64, 47]) >> 64 Augmented views

                if selected_idx is not None:
                    output = output[selected_idx]
                else:
                    output, selected_idx = select_confident_samples(output, args.selection_p)
                args._selected_view_idx = selected_idx
                   
                if selected_idx is None:
                    args._selected_view_idx = None
                    
                if selected_idx is not None:
                    softmax_out = torch.softmax(output,dim=-1)
                    soft_mean = torch.mean(softmax_out,dim=0)
                                        
                    
                    number_of_class = output.shape[1]
                    
                loss = add_direction_losses(model, output, avg_entropy(output), args)
                #getting dw value
                dw = conf_acc(output)
        else:
            loss = 0
         


        if args.two_step and 'tpt' in args.run_type:
            optimizer.zero_grad()
            # compute gradient and do SGD step
            scaler.scale(loss).backward(retain_graph=True)
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.step(optimizer)
            scaler.update()
            loss = 0

            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output2 = model((image_feature, pgen_ctx),cons,args)
                else:
                    output2,text_varience = model(inputs,cons,args) 

        if 'otpt' in args.run_type:
            if output == None and output2 == None:
                single_output = model(args.image)

            lambda_ = args.lambda_term

            number_of_class = output.shape[1]            
            #------------------------------------------------- Householder Transform--------
            text_feature = model.textfeatures_
            #print("text feature shape model:",text_feature.shape)
            #computing orthogonal constrained  SVD
            Wwt  =  torch.matmul(text_feature,text_feature.T)
            wwt_norm_col_HT = torch.linalg.norm(Wwt,dim=-1)
            Wwt_val_HT = wwt_norm_col_HT.mean()
            #wtW  =  torch.matmul(text_feature.T,text_feature)
            e = torch.eye(Wwt.shape[1], device=args.gpu)
            M_norm = torch.linalg.norm(Wwt, dim=0,keepdim=True)
            scaled_e = e * M_norm
            # Subtract the scaled identity matrix from Wwt
            u = Wwt - scaled_e
            u_norm = torch.linalg.norm(u, dim=-1,keepdim=True)
            #u_norm = u_norm ** 2
            # We need to expand u_norm to shape (47, 47, 1) for broadcasting
            #u_norm_exp = u_norm.unsqueeze(2)  # Shape: (1, 47, 1)
            
            #Transposing the u for batch element column and coresponding column transpose matrix multiplication
          
            v = u/u_norm
            normalized_matrix_exp = v.unsqueeze(2)  # Shape: (47, 47, 1)
            normalized_matrix_T_exp = v.unsqueeze(1)  # Shape: (47, 1, 47)
            
            # This will create a batch of 3 matrices, each of shape (47, 47)
            outer_products = normalized_matrix_exp @ normalized_matrix_T_exp  # Shape: (47, 47, 47)
            
            # Perform element-wise division of each outer product by the corresponding u_norm value
            divided_matrix = outer_products #/ u_norm_exp  # Shape: (47, 47, 47)

            # Multiply the result by 2
            scaled_matrix = 2 * divided_matrix  # Shape: (47, 47, 47)
            # Subtract the scaled result from the corresponding identity matrix for each batch
            identity_matrix_dim = e.unsqueeze(0).expand(Wwt.shape[1], -1, -1)  # Shape: (47, 47, 47)
            # Subtract from identity matrix
            transformed_matrix = identity_matrix_dim - scaled_matrix  # Shape: (47, 47, 47)
            # Reshape M so that its columns are aligned for batch multiplication
            Wwt_exp = Wwt.unsqueeze(2)  # Shape: (47, 47, 1)

            # Perform batched matrix multiplication between transformed matrix and M_exp
            Hx = torch.bmm(transformed_matrix, Wwt_exp)  # Shape: (47, 47, 1)
            
            # Reshape back the result to (3, 3) by removing the last singleton dimension
            Hx = Hx.squeeze(2)  # Shape: (47, 47)
            #print("shape of Hx:",Hx.shape)
            #print("Hx:",Hx)
            #normalizing Column wise
            #Hx_norm =torch.linalg.norm(Hx, dim=0,keepdim=True)
            #Hx = Hx/Hx_norm
            Ht_ortho = Hx - e  
            Ht_ortho_norm = torch.linalg.norm(Ht_ortho, dim=-1)
            Ht_ortho_norm_val = Ht_ortho_norm.mean()
            
            
            #-------------------------------------House holder end-----------------------------------------------          
      
            loss += (+(lambda_ * Ht_ortho_norm_val))
                    
        if args.run_type not in ['baseline', 'baseline_cocoop', 'baseline_coop', 'baseline_ts']:
            optimizer.zero_grad()
            # compute gradient and do SGD step
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place (using the calculated gradient from the loss function to update the model parameter.)
            scaler.step(optimizer)
            scaler.update()
            #number_of_class = output.shape[1]

    if args.cocoop:
        return pgen_ctx

    return               #output_logit_norm.item(),diagonal_norm.item()


def main(args):

    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
    if args.cocoop:
        model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx,args.disp_cons)
        assert args.load is not None
        load_model_weight(args.load, model, "cuda:{}".format(args.gpu), args) # to load to cuda: device="cuda:{}".format(args.gpu)
        model_state = deepcopy(model.state_dict())
    else:
        model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx,args.ctx_init,args.disp_cons)
        if args.load is not None:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            #pre-trained weights
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                #model.prompt_learner[0].ctx.copy_(pretrained_ctx)
                #model.prompt_learner[0].ctx_init_state = pretrained_ctx
                #Copies the pre-trained context to the model's prompt learner without gradient tracking.
                model.prompt_learner.ctx.copy_(pretrained_ctx)
                model.prompt_learner.ctx_init_state = pretrained_ctx

        model_state = None

    for name, param in model.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define optimizer
    if args.cocoop:
        optimizer = None
        optim_state = None
    else:
        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    
    #pre-pare dataset
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    print('length of datset', len(datasets))
    for set_id in datasets:
          print('name id of dataset:',set_id)
          
          
    #assert len(datasets) == 1
    results = {}
    matrics = {}
    accuracy_data = {}
    ece_data = {}
    confidence_data = {}
    temperature_data = {}
  
    for set_id in datasets:
        accuracy_data[set_id] = []  # Initialize list for each set_id
        ece_data[set_id] = []       # Initialize list for each set_id 
        confidence_data[set_id] =[]

        if args.tpt:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            
            if args.I_augmix:
                data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                            augmix=len(set_id)>=1)
            else:
                data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                            augmix=len(set_id)>1)
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size

        print("evaluating: {}".format(set_id))
        # reset the model
        # Reset classnames of custom CLIP model
        if len(set_id) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
            #print("classnames:",classnames)
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V',]:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            
            else:
                classnames = classnames_all
        if args.cocoop:
            model.prompt_generator.reset_classnames(classnames, args.arch)
            model = model.cpu()
            model_state = model.state_dict()
            model = model.cuda(args.gpu)
        else:
            model.reset_classnames(classnames, args.arch)

        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        if args.max_test_samples is not None and args.max_test_samples > 0 and len(val_dataset) > args.max_test_samples:
            subset_generator = torch.Generator().manual_seed(args.seed)
            subset_idx = torch.randperm(len(val_dataset), generator=subset_generator)[:args.max_test_samples].tolist()
            val_dataset = torch.utils.data.Subset(val_dataset, subset_idx)
            print("using deterministic subset: {} samples".format(len(val_dataset)))
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
            
        results[set_id],result_dict = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args,args.disp_cons,classnames,set_id)
        #find the mean value of confidences
           
        acc, ece,bin_acc,incorrect_confidences = Calculator(result_dict)
        accuracy_data[set_id].append(acc)
        ece_data[set_id].append(ece)
        if args.temperature_grid:
            temperatures = [round(0.5 + 0.1 * idx, 2) for idx in range(46)]
            temperature_data[set_id] = temperature_grid_metrics(result_dict, temperatures)
        
        try:
                    print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
                    print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))        

        del val_dataset, val_loader        
      
    print("======== Result Summary ========")

    dataset_ids = list(results.keys())

    if args.csv_log:
        custom_path = args.csv_log
    else:
        custom_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "log",
            "result_summary_{}sets.csv".format(len(dataset_ids)),
        )
    directory = os.path.dirname(custom_path)
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Check if the file exists
    file_exists = os.path.isfile(custom_path)
    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Open the file in append mode if it exists, otherwise in write mode
    
    print("params: nstep	lr	bs")
    print("params: {}	    {}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")

    with open(custom_path, 'a' if file_exists else 'w', newline='') as csvfile:    
        csvwriter = csv.writer(csvfile)

        if not file_exists:
            #csvwriter.writerow(["======== Result Summary ========"])
            csvwriter.writerow(["params: nstep", "lr", "bs"])
            csvwriter.writerow([current_datetime,"params: {} {} {}".format(args.tta_steps, args.lr, args.batch_size)])
            csvwriter.writerow(["", "[set_id]", "Top-1 acc.", "Top-5 acc."])

               
        # Write the dataset ids in the first row
        dataset_ids = list(results.keys())
        csvwriter.writerow(current_datetime)

        # code without text disperssion test
        csvwriter.writerow([""] + dataset_ids)
        
        # Write the Top-1 accuracies
        top1_accs = ["Top-1 acc."] + ["{:.2f}".format(results[id][0]) for id in dataset_ids]
        csvwriter.writerow(top1_accs)

        # Write the Top-5 accuracies
        top5_accs = ["Top-5 acc."] + ["{:.2f}".format(results[id][1]) for id in dataset_ids]
        csvwriter.writerow(top5_accs)  

         # Write final accuracies
        final_acc = ["Accuracy"] + ["{:.2f}".format(accuracy_data[id][0]) for id in dataset_ids]
        csvwriter.writerow(final_acc)

        # Write the ECE
        ECE = ["ECE."] + ["{:.2f}".format(ece_data[id][0]) for id in dataset_ids]
        csvwriter.writerow(ECE)   

    if args.temperature_grid:
        temp_path = "{}.temperature_grid.csv".format(custom_path)
        with open(temp_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["dataset", "temperature", "accuracy", "ece", "mean_confidence"])
            for set_id in dataset_ids:
                for temperature, acc, ece, mean_conf in temperature_data[set_id]:
                    csvwriter.writerow([
                        set_id,
                        "{:.2f}".format(temperature),
                        "{:.4f}".format(acc),
                        "{:.4f}".format(ece),
                        "{:.4f}".format(mean_conf),
                    ])
        print("temperature grid written to: {}".format(temp_path))

        
def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, cons,classnames,set_id):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.reset()
    end = time.time()

    #define a softmax layer
    softmax = torch.nn.Softmax(dim=1)

    if 'otpt' in args.run_type or any(
        getattr(args, flag, False)
        for flag in [
            "use_nc_sem",
            "use_mm_ortho",
            "use_cache_ortho",
            "use_dist_align",
            "use_proda_ortho",
            "use_adv_ortho",
            "use_entropy_target",
            "use_logitnorm_tpt",
        ]
    ):
        model.l2_norm_cal = True
    else:
        model.l2_norm_cal = False

    #initializing result_dictionary    
    result_dict = {'max_confidence': [], 'prediction': [], 'label': [], 'logits': []}  
    for i, (images, target) in enumerate(val_loader):
        
        assert args.gpu is not None
        
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)
        if args.tpt:
            images = torch.cat(images, dim=0)

        if 'otpt' in args.run_type:
            args.image = image

        # reset the tunable prompt to its initial state
        if not args.cocoop: # no need to reset cocoop because it's fixed
            if args.tta_steps > 0:
                with torch.no_grad():
                    model.reset()
            optimizer.load_state_dict(optim_state)
            #for each image updating the prompt parameter
            #images shape for tunning: torch.Size([64, 3, 224, 224])
            test_time_tuning(model, images, optimizer, scaler, args,cons)
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_feature, pgen_ctx = model.gen_ctx(images, args.tpt)
            optimizer = None
            #fine-tuned prompt embeddings
            pgen_ctx = test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args, cons)

        # The actual inference goes here
        if args.tpt:
            if args.cocoop:
                image_feature = image_feature[0].unsqueeze(0)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output = model((image_feature, pgen_ctx),cons,args)
                else:
                    output = model(image,cons,args)
        
        if 'ts' not in args.run_type:
            softmax_output = softmax(output) 
        elif 'ts' in args.run_type:
            if 'ViT' in args.arch:
                softmax_output =  softmax(output/temperature_value['ViT']) #softmax(output)
            elif 'RN' in args.arch:
                softmax_output =  softmax(output/temperature_value['RN'])  #softmax(output)
            else:
                ipdb.set_trace()      

        #maximum confidence of the softmax_output and its index
        max_confidence, max_index = torch.max(softmax_output, 1)
        update_otpt_cache(model, output, args)
 
        result_dict['max_confidence'].append(max_confidence.item())
        result_dict['prediction'].append(max_index.item())
        result_dict['label'].append(target.item())    
        result_dict['logits'].append(output.detach().float().cpu())
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #print("accuracy acc1:",acc1)        
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    num_classes = output.shape[0]
    
    progress.display_summary()
    
    return [top1.avg, top5.avg], result_dict


temperature_value = {'ViT': 1.16, 'RN': 1.15} #for temperature scaling experiments 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--csv_log',type=str,help='path to save the CSV summary')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('--max_test_samples', type=int, default=None, help='evaluate a deterministic subset for quick one-seed experiments')
    parser.add_argument('--temperature_grid', action='store_true', default=False, help='write ECE/accuracy for T=0.5..5.0 without changing predictions')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=1, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)

    # added args for c-tpt --------------------------------
    parser.add_argument('--lambda_term' , type=float, default=0.0, help='lambda for c-tpt')
    parser.add_argument('--disp_cons' , type=int, nargs='+',default=[18.0],help='List of display constants')
    parser.add_argument('--run_type' , type=str, default='baseline_tpt', choices=['baseline', 'tpt', 'tpt_otpt', 'tpt_ts', 'tpt_nc_sem'])
    parser.add_argument('--two_step', action='store_true', default=False, help='two step training')
    parser.add_argument('--I_augmix', action='store_true', default=False, help='augmix for I')
    parser.add_argument('--use_nc_sem', action='store_true', default=False, help='enable NC-TPT + semantic-aware Gram regularizer')
    parser.add_argument('--lam_nc_sem', type=float, default=1.0, help='weight for NC/semantic Gram regularizer')
    parser.add_argument('--sem_alpha', type=float, default=0.5, help='target Gram mix: alpha * ETF + (1-alpha) * semantic prior')
    parser.add_argument('--sem_prior_path', type=str, default=None, help='optional offline CxC semantic prior matrix; default uses frozen CLIP text Gram')
    parser.add_argument('--nc_sem_log_limit', type=int, default=5, help='always log NC/semantic stats for first N updates')
    parser.add_argument('--nc_sem_log_every', type=int, default=200, help='then log NC/semantic stats every k updates')
    parser.add_argument('--use_mm_ortho', action='store_true', default=False, help='direction 1: visual prototype and cross-modal Gram regularizer')
    parser.add_argument('--lam_mm_visual', type=float, default=0.2, help='weight for visual prototype Gram loss')
    parser.add_argument('--lam_mm_cross', type=float, default=0.2, help='weight for text-visual cross Gram loss')
    parser.add_argument('--use_cache_ortho', action='store_true', default=False, help='direction 2: online high-confidence visual prototype cache')
    parser.add_argument('--lam_cache', type=float, default=0.5, help='weight for cache prototype duality loss')
    parser.add_argument('--cache_conf', type=float, default=0.8, help='confidence threshold for inserting samples into cache')
    parser.add_argument('--use_dist_align', action='store_true', default=False, help='direction 4: frozen-prompt distribution alignment proxy')
    parser.add_argument('--lam_dist_align', type=float, default=0.1, help='weight for frozen-prompt distribution alignment')
    parser.add_argument('--use_proda_ortho', action='store_true', default=False, help='direction 5: prompt-noise ensemble entropy and diversity')
    parser.add_argument('--lam_proda', type=float, default=0.2, help='weight for prompt distribution loss')
    parser.add_argument('--proda_prompts', type=int, default=4, help='number of prompt samples')
    parser.add_argument('--proda_noise_std', type=float, default=0.01, help='std of prompt embedding perturbation')
    parser.add_argument('--lam_proda_div', type=float, default=0.1, help='diversity reward inside prompt distribution loss')
    parser.add_argument('--use_adv_ortho', action='store_true', default=False, help='direction 6: worst-view entropy robustness proxy')
    parser.add_argument('--lam_adv', type=float, default=0.2, help='weight for worst-view entropy')
    parser.add_argument('--use_entropy_target', action='store_true', default=False, help='calibration method: keep adapted entropy close to frozen zero-shot entropy')
    parser.add_argument('--lam_entropy_target', type=float, default=2.0, help='weight for zero-shot entropy target')
    parser.add_argument('--entropy_target_margin', type=float, default=0.0, help='extra entropy margin above zero-shot target')
    parser.add_argument('--entropy_target_mode', type=str, default='floor', choices=['floor', 'symmetric'], help='floor only prevents overconfidence; symmetric matches target entropy')
    parser.add_argument('--use_logitnorm_tpt', action='store_true', default=False, help='calibration method: add LogitNorm entropy as an extra loss on top of O-TPT')
    parser.add_argument('--lam_logitnorm', type=float, default=0.0, help='weight for extra LogitNorm entropy loss')
    parser.add_argument('--logitnorm_tau', type=float, default=0.04, help='temperature after logit normalization for LogitNorm-TPT')
    parser.add_argument('--use_trust_region', action='store_true', default=False, help='calibration-preserving method: constrain prompt/text drift from zero-shot anchor')
    parser.add_argument('--lam_trust_region', type=float, default=0.0, help='weight for trust-region drift penalty')
    parser.add_argument('--trust_region_mode', type=str, default='ctx', choices=['ctx', 'text', 'both'], help='trust-region penalty space')
    # ------------------------------------------------

    args = parser.parse_args()
    
    if 'otpt' not in args.run_type:
        args.lambda_term = 0.0

    #result_dict = {'max_confidence': [], 'prediction': [], 'label': []}
    main(args)
    #acc, ece = Calculator(result_dict)
