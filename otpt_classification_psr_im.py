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

from clip.custom_clip_psr_im import get_coop
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
    nll = float(np.mean(result_dict['nll'])) if result_dict['nll'] else float('nan')
    brier = float(np.mean(result_dict['brier'])) if result_dict['brier'] else float('nan')

    print('acc: ', acc*100)
    print('ece: ', ece_data[0]*100)
    print('nll: ', nll)
    print('brier: ', brier)
          
    return acc*100, ece_data[0]*100, nll, brier, ece_data[1], incorrect_confidences



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


def sharpen_probs(probs, temperature):
    temperature = max(float(temperature), 1e-6)
    sharpened = torch.softmax(torch.log(probs.clamp_min(1e-8)) / temperature, dim=-1)
    return sharpened


def maybe_log_pseudo_target_stats(args, pseudo_target, objective_name):
    if args is None:
        return

    counter = getattr(args, "_pseudo_log_counter", 0) + 1
    args._pseudo_log_counter = counter

    log_limit = getattr(args, "pseudo_log_limit", 5)
    log_every = getattr(args, "pseudo_log_every", 200)
    should_log = counter <= log_limit or (log_every > 0 and counter % log_every == 0)
    if not should_log:
        return

    pseudo_entropy = -(pseudo_target * pseudo_target.clamp_min(1e-8).log()).sum().item()
    pseudo_max_prob = pseudo_target.max().item()
    print(
        "[{}] tilde_p entropy = {:.3f}, max prob = {:.3f}, T = {:.2f}".format(
            objective_name,
            pseudo_entropy,
            pseudo_max_prob,
            getattr(args, "psr_T", 1.0),
        )
    )


def psr_loss(outputs, temperature, args=None):
    probs = outputs.softmax(dim=-1)
    pseudo_target = sharpen_probs(probs.mean(dim=0), temperature).detach()
    objective_name = getattr(args, "tta_objective", "psr") if args is not None else "psr"
    maybe_log_pseudo_target_stats(args, pseudo_target, objective_name)
    return ((probs - pseudo_target.unsqueeze(0)) ** 2).sum(dim=-1).mean()


def im_loss(outputs, lam_im):
    probs = outputs.softmax(dim=-1)
    cond_ent = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
    p_bar = probs.mean(dim=0)
    marg_ent = -(p_bar * p_bar.clamp_min(1e-8).log()).sum()
    return cond_ent - lam_im * marg_ent


def build_etf_gram(num_classes, device, dtype):
    if num_classes <= 1:
        return torch.eye(num_classes, device=device, dtype=dtype)
    eye = torch.eye(num_classes, device=device, dtype=dtype)
    ones = torch.ones(num_classes, num_classes, device=device, dtype=dtype) / num_classes
    return (num_classes / (num_classes - 1)) * (eye - ones)


def maybe_log_nc_stats(args, etf_loss, align_loss, raw_gram_matrix, proxy_gram_matrix, raw_mean_sq):
    if args is None or not getattr(args, "use_etf", False):
        return

    counter = getattr(args, "_nc_log_counter", 0) + 1
    args._nc_log_counter = counter

    log_limit = getattr(args, "nc_log_limit", 5)
    log_every = getattr(args, "nc_log_every", 200)
    should_log = counter <= log_limit or (log_every > 0 and counter % log_every == 0)
    if not should_log:
        return

    num_classes = proxy_gram_matrix.shape[0]
    offdiag_mask = ~torch.eye(num_classes, dtype=torch.bool, device=proxy_gram_matrix.device)
    proxy_offdiag_mean = proxy_gram_matrix[offdiag_mask].mean().item() if num_classes > 1 else 0.0
    proxy_diag_mean = proxy_gram_matrix.diag().mean().item()
    raw_offdiag_mean = raw_gram_matrix[offdiag_mask].mean().item() if num_classes > 1 else 0.0
    target_offdiag = -1.0 / (num_classes - 1) if num_classes > 1 else 0.0
    print(
        "[nc] L_etf = {:.4f}, L_align = {:.4f}, raw_offdiag = {:.4f}, proxy_diag = {:.4f}, proxy_offdiag = {:.4f}, target_offdiag = {:.4f}, raw_mean_sq = {:.6f}".format(
            etf_loss.item(),
            align_loss.item() if torch.is_tensor(align_loss) else float(align_loss),
            raw_offdiag_mean,
            proxy_diag_mean,
            proxy_offdiag_mean,
            target_offdiag,
            raw_mean_sq.item() if torch.is_tensor(raw_mean_sq) else float(raw_mean_sq),
        )
    )


def compute_nc_regularizer(model, outputs, selected_idx, args):
    if not getattr(args, "use_etf", False):
        return outputs.new_tensor(0.0)

    # Keep raw CLIP classifier for logits; ETF acts only as a weak auxiliary
    # prior on a centered proxy geometry plus a small bias penalty on raw text.
    raw_text_feature = F.normalize(getattr(model, "raw_textfeatures_", model.textfeatures_).float(), dim=-1)
    proxy_text_feature = F.normalize(getattr(model, "nc_proxy_textfeatures_", raw_text_feature).float(), dim=-1)
    num_classes = proxy_text_feature.shape[0]

    cached_gram = getattr(model, "_etf_gram", None)
    if (
        cached_gram is None
        or cached_gram.shape[0] != num_classes
        or cached_gram.device != proxy_text_feature.device
        or cached_gram.dtype != proxy_text_feature.dtype
    ):
        model._etf_gram = build_etf_gram(num_classes, proxy_text_feature.device, proxy_text_feature.dtype)

    raw_gram_matrix = raw_text_feature @ raw_text_feature.t()
    proxy_gram_matrix = proxy_text_feature @ proxy_text_feature.t()
    offdiag_mask = ~torch.eye(num_classes, dtype=torch.bool, device=proxy_gram_matrix.device)
    offdiag_loss = ((proxy_gram_matrix[offdiag_mask] - model._etf_gram[offdiag_mask]) ** 2).mean()
    # Penalize only the shared bias that remains in the raw classifier, instead
    # of forcing the deployed classifier itself into an ETF simplex.
    raw_mean_sq = raw_text_feature.mean(dim=0).pow(2).mean()
    etf_loss = offdiag_loss + 0.1 * raw_mean_sq
    total_loss = args.lam_etf * etf_loss

    align_loss = outputs.new_tensor(0.0)
    if getattr(args, "lam_align", 0.0) > 0.0:
        image_feature = F.normalize(model.imagefeatures_.float(), dim=-1)
        if selected_idx is not None:
            image_feature = image_feature[selected_idx]
        # Keep alignment as a soft, detached auxiliary prior on the proxy
        # geometry; raw classifier logits stay untouched.
        soft_assignment = outputs.float().softmax(dim=-1).detach()
        soft_anchor = F.normalize(soft_assignment @ proxy_text_feature, dim=-1)
        align_loss = 1.0 - (image_feature * soft_anchor).sum(dim=-1).mean()
        total_loss = total_loss + args.lam_align * align_loss

    maybe_log_nc_stats(args, etf_loss, align_loss, raw_gram_matrix, proxy_gram_matrix, raw_mean_sq)
    return total_loss


def compute_anchor_prior_regularizer(model, outputs, selected_idx, args):
    if not getattr(args, "use_anchor_prior", False):
        return outputs.new_tensor(0.0)

    image_feature = F.normalize(model.imagefeatures_.float(), dim=-1)
    if selected_idx is not None:
        image_feature = image_feature[selected_idx]

    anchor_text_feature = F.normalize(
        getattr(model, "anchor_textfeatures_", model.textfeatures_).float(),
        dim=-1,
    )
    logit_scale = model.logit_scale.float().exp()
    anchor_logits = logit_scale * image_feature @ anchor_text_feature.t()

    adapt_probs = outputs.float().softmax(dim=-1).mean(dim=0)
    anchor_probs = anchor_logits.softmax(dim=-1).mean(dim=0).detach()

    top_m = int(getattr(args, "anchor_topm", 0))
    if top_m > 0 and top_m < adapt_probs.numel():
        top_idx = torch.unique(torch.cat([adapt_probs.detach().topk(top_m).indices, anchor_probs.topk(top_m).indices]))
        adapt_probs = adapt_probs[top_idx]
        anchor_probs = anchor_probs[top_idx]
        adapt_probs = adapt_probs / adapt_probs.sum().clamp_min(1e-6)
        anchor_probs = anchor_probs / anchor_probs.sum().clamp_min(1e-6)

    loss_anchor = (
        adapt_probs
        * (adapt_probs.clamp_min(1e-6).log() - anchor_probs.clamp_min(1e-6).log())
    ).sum()
    anchor_conf = anchor_probs.max()
    anchor_weight = args.lam_anchor * anchor_conf if getattr(args, "anchor_conf_gate", False) else args.lam_anchor

    ctx = model.prompt_learner.ctx.float()
    ctx_init = model.prompt_learner.ctx_init_state.float().to(ctx.device)
    loss_pull = (ctx - ctx_init).pow(2).mean()
    return anchor_weight * loss_anchor + args.lam_pull * loss_pull


def compute_tpt_loss(outputs, args):
    objective = getattr(args, "tta_objective", "entropy")
    if objective == "entropy":
        return avg_entropy(outputs)
    if objective == "psr":
        return psr_loss(outputs, args.psr_T, args)
    if objective == "im":
        return im_loss(outputs, args.lam_im)
    if objective == "psr_im":
        return psr_loss(outputs, args.psr_T, args) + im_loss(outputs, args.lam_im)
    raise ValueError(f"Unknown --tta_objective: {objective}")

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
                   
                    softmax_out = torch.softmax(output,dim=-1)
                    soft_mean = torch.mean(softmax_out,dim=0)
                                        
                    
                    number_of_class = output.shape[1]
                    
                loss = compute_tpt_loss(output, args)
                loss = loss + compute_nc_regularizer(model, output, selected_idx, args)
                loss = loss + compute_anchor_prior_regularizer(model, output, selected_idx, args)
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
    nll_data = {}
    brier_data = {}
    confidence_data = {}
  
    for set_id in datasets:
        accuracy_data[set_id] = []  # Initialize list for each set_id
        ece_data[set_id] = []       # Initialize list for each set_id 
        nll_data[set_id] = []
        brier_data[set_id] = []
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
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
            
        results[set_id],result_dict = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args,args.disp_cons,classnames,set_id)
        #find the mean value of confidences
           
        acc, ece, nll, brier, bin_acc, incorrect_confidences = Calculator(result_dict)
        accuracy_data[set_id].append(acc)
        ece_data[set_id].append(ece)
        nll_data[set_id].append(nll)
        brier_data[set_id].append(brier)
        
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

        # Write the NLL
        NLL = ["NLL"] + ["{:.4f}".format(nll_data[id][0]) for id in dataset_ids]
        csvwriter.writerow(NLL)

        # Write the Brier score
        BRIER = ["Brier"] + ["{:.4f}".format(brier_data[id][0]) for id in dataset_ids]
        csvwriter.writerow(BRIER)

        
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
    args._pseudo_log_counter = 0
    args._nc_log_counter = 0
    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.reset()
    end = time.time()

    if (
        'otpt' in args.run_type
        or getattr(args, "use_etf", False)
        or getattr(args, "use_anchor_prior", False)
    ):
        model.l2_norm_cal = True
    else:
        model.l2_norm_cal = False

    #initializing result_dictionary    
    result_dict = {'max_confidence': [], 'prediction': [], 'label': [], 'nll': [], 'brier': []}
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
        
        output_for_metrics = output.float()
        if 'ts' not in args.run_type:
            log_probs = F.log_softmax(output_for_metrics, dim=1)
        elif 'ts' in args.run_type:
            if 'ViT' in args.arch:
                log_probs = F.log_softmax(output_for_metrics / temperature_value['ViT'], dim=1)
            elif 'RN' in args.arch:
                log_probs = F.log_softmax(output_for_metrics / temperature_value['RN'], dim=1)
            else:
                ipdb.set_trace()
        softmax_output = log_probs.exp()

        #maximum confidence of the softmax_output and its index
        max_confidence, max_index = torch.max(softmax_output, 1)

        true_log_probs = log_probs.gather(1, target.view(-1, 1)).squeeze(1)
        batch_nll = -true_log_probs
        batch_brier = ((softmax_output - F.one_hot(target, softmax_output.shape[1]).float()) ** 2).sum(dim=-1)

        result_dict['max_confidence'].extend(max_confidence.detach().cpu().tolist())
        result_dict['prediction'].extend(max_index.detach().cpu().tolist())
        result_dict['label'].extend(target.detach().cpu().tolist())
        result_dict['nll'].extend(batch_nll.detach().cpu().tolist())
        result_dict['brier'].extend(batch_brier.detach().cpu().tolist())
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
    parser.add_argument('--run_type' , type=str, default='baseline_tpt', choices=['baseline', 'tpt', 'tpt_otpt', 'tpt_ts'])
    parser.add_argument('--two_step', action='store_true', default=False, help='two step training')
    parser.add_argument('--I_augmix', action='store_true', default=False, help='augmix for I')
    parser.add_argument('--tta_objective', type=str, default='entropy', choices=['entropy', 'psr', 'im', 'psr_im'])
    parser.add_argument('--psr_T', type=float, default=2.0, help='temperature for PSR pseudo-target sharpening')
    parser.add_argument('--lam_im', type=float, default=0.8, help='weight for information maximization marginal entropy term')
    parser.add_argument('--pseudo_log_limit', type=int, default=5, help='always log pseudo-target stats for the first N TTA updates')
    parser.add_argument('--pseudo_log_every', type=int, default=200, help='after the first N updates, log pseudo-target stats every k updates')
    parser.add_argument('--use_etf', action='store_true', default=False, help='enable Neural Collapse ETF regularization')
    parser.add_argument('--lam_etf', type=float, default=1.0, help='weight for ETF Gram alignment')
    parser.add_argument('--lam_align', type=float, default=0.5, help='weight for soft NC classifier-image alignment term')
    parser.add_argument('--use_anchor_prior', action='store_true', default=False, help='keep adapted prediction close to frozen zero-shot anchor')
    parser.add_argument('--lam_anchor', type=float, default=0.05, help='weight for zero-shot anchor KL trust region')
    parser.add_argument('--lam_pull', type=float, default=0.01, help='weight for prompt residual pull-back to initial context')
    parser.add_argument('--anchor_topm', type=int, default=20, help='top-m classes used for anchor trust-region KL; <=0 uses full distribution')
    parser.add_argument('--anchor_conf_gate', action='store_true', default=False, help='scale anchor KL by anchor confidence')
    parser.add_argument('--nc_log_limit', type=int, default=5, help='always log NC geometry stats for the first N TTA updates')
    parser.add_argument('--nc_log_every', type=int, default=200, help='after the first N updates, log NC geometry stats every k updates')
    # ------------------------------------------------

    args = parser.parse_args()
    
    if 'otpt' not in args.run_type:
        args.lambda_term = 0.0

    #result_dict = {'max_confidence': [], 'prediction': [], 'label': []}
    main(args)
    #acc, ece = Calculator(result_dict)
