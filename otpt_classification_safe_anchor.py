import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np
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

from clip.custom_clip_iptp_bas import get_coop
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
    num_selected = max(1, int(batch_entropy.size(0) * top))
    num_selected = min(batch_entropy.size(0), num_selected)
    idx = torch.argsort(batch_entropy, descending=False)[:num_selected]
    #this returns the maximum confidence for the each augmented image and it's index
    return logits[idx], idx


def select_views_for_adaptation(logits, args):
    selection_mode = getattr(args, "view_selection", "hard_toprho")
    if selection_mode == "all":
        idx = torch.arange(logits.shape[0], device=logits.device)
        return logits, idx
    if selection_mode == "hard_toprho":
        return select_confident_samples(logits, args.selection_p)
    raise ValueError(f"Unknown --view_selection: {selection_mode}")


def avg_entropy(outputs):
   
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]  tensor of shape [6, 1]
   
   
    #computing the average logit from the augmented views of the image.
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    #print("avg-logits:", -(avg_logits * torch.exp(avg_logits)).sum(dim=-1))
    #this returns the average entropy for the aumented images
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def normalized_entropy_from_logits(logits):
    probs = logits.softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=-1)
    return entropy / math.log(probs.shape[-1])


def compute_view_reliability(mix_logits, anchor_logits):
    mix_probs = mix_logits.softmax(dim=-1)
    anchor_probs = anchor_logits.softmax(dim=-1)
    agreement = (mix_probs * anchor_probs).sum(dim=-1)
    top2 = mix_probs.topk(min(2, mix_probs.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        margin = top2[..., 0]
    else:
        margin = top2[..., 0] - top2[..., 1]
    uncertainty = normalized_entropy_from_logits(mix_logits)
    scores = 2.0 * agreement + 2.0 * margin - uncertainty
    weights = torch.sigmoid(scores)
    return weights / weights.sum().clamp_min(1e-6)


def rasc_consistency_loss(mix_logits, anchor_logits):
    view_weights = compute_view_reliability(mix_logits, anchor_logits)
    mix_probs = mix_logits.softmax(dim=-1)
    anchor_probs = anchor_logits.softmax(dim=-1)
    teacher = (view_weights.unsqueeze(-1) * 0.5 * (mix_probs.detach() + anchor_probs.detach())).sum(dim=0)
    teacher = teacher / teacher.sum().clamp_min(1e-6)
    per_view_loss = -(teacher.unsqueeze(0) * mix_logits.log_softmax(dim=-1)).sum(dim=-1)
    return (view_weights * per_view_loss).sum()


def rasc_semantic_loss(text_features, anchor_text_features):
    text_features = F.normalize(text_features, dim=-1)
    anchor_text_features = F.normalize(anchor_text_features, dim=-1)
    current_sim = text_features @ text_features.t()
    anchor_sim = anchor_text_features @ anchor_text_features.t()
    mask = ~torch.eye(current_sim.shape[0], dtype=torch.bool, device=current_sim.device)
    far_weights = (1.0 - anchor_sim.detach()).clamp(min=0.0)
    near_weights = anchor_sim.detach().clamp(min=0.0)
    far_loss = ((far_weights * current_sim.pow(2))[mask]).mean()
    near_loss = ((near_weights * (current_sim - anchor_sim.detach()).pow(2))[mask]).mean()
    return far_loss + 0.5 * near_loss


def _distribution_top_indices(probs, top_m):
    top_m = min(int(top_m), probs.numel())
    if top_m <= 0:
        return torch.empty(0, dtype=torch.long, device=probs.device)
    return probs.topk(top_m).indices


def sao_semantic_orth_loss(text_features, anchor_text_features, output, anchor_output, args):
    text_features = F.normalize(text_features, dim=-1)
    anchor_text_features = F.normalize(anchor_text_features, dim=-1)
    current_gram = text_features @ text_features.t()
    anchor_gram = (anchor_text_features @ anchor_text_features.t()).detach()
    num_classes = current_gram.shape[0]
    device = current_gram.device

    off_diag = anchor_gram.masked_fill(torch.eye(num_classes, dtype=torch.bool, device=device), -float("inf"))
    top_k = min(args.sao_topk, max(num_classes - 1, 1))
    semantic_mask = torch.zeros_like(anchor_gram, dtype=torch.bool)
    if top_k > 0 and num_classes > 1:
        semantic_idx = off_diag.topk(top_k, dim=-1).indices
        semantic_mask.scatter_(1, semantic_idx, True)
        semantic_mask = semantic_mask | semantic_mask.t()

    target_gram = torch.zeros_like(anchor_gram)
    target_gram[semantic_mask] = args.sao_alpha * anchor_gram[semantic_mask]
    target_gram.fill_diagonal_(1.0)

    adapt_probs = output.softmax(dim=-1).mean(dim=0).detach()
    anchor_probs = anchor_output.softmax(dim=-1).mean(dim=0).detach()
    adapt_top = _distribution_top_indices(adapt_probs, args.sao_topm)
    anchor_top = _distribution_top_indices(anchor_probs, args.sao_topm)
    adapt_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
    anchor_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
    if adapt_top.numel() > 0:
        adapt_mask[adapt_top] = True
    if anchor_top.numel() > 0:
        anchor_mask[anchor_top] = True
    local_mask = adapt_mask & anchor_mask
    if not local_mask.any():
        local_mask = adapt_mask | anchor_mask
    local_pair = local_mask[:, None] | local_mask[None, :]

    anchor_conf = anchor_probs.max()
    beta = args.sao_beta * anchor_conf
    weights = torch.ones_like(current_gram)
    weights = weights + beta * local_pair.to(current_gram.dtype)

    diff = weights * (current_gram - target_gram)
    return diff.pow(2).mean()


def sao_anchor_kl_loss(output, anchor_output, args):
    adapt_probs = output.softmax(dim=-1).mean(dim=0)
    anchor_probs = anchor_output.softmax(dim=-1).mean(dim=0).detach()
    if args.sao_anchor_topm > 0:
        top_m = min(args.sao_anchor_topm, adapt_probs.numel())
        top_idx = torch.unique(torch.cat([adapt_probs.detach().topk(top_m).indices, anchor_probs.topk(top_m).indices]))
        adapt_probs = adapt_probs[top_idx]
        anchor_probs = anchor_probs[top_idx]
        adapt_probs = adapt_probs / adapt_probs.sum().clamp_min(1e-6)
        anchor_probs = anchor_probs / anchor_probs.sum().clamp_min(1e-6)
    return (adapt_probs * (adapt_probs.clamp_min(1e-6).log() - anchor_probs.clamp_min(1e-6).log())).sum()


def sao_calibration_loss(output, anchor_output, args):
    adapt_conf = output.softmax(dim=-1).max(dim=-1).values.mean()
    anchor_conf = anchor_output.softmax(dim=-1).max(dim=-1).values.mean().detach()
    return F.relu(adapt_conf - anchor_conf - args.sao_conf_margin).pow(2)


def sao_view_consistency_loss(output):
    probs = output.softmax(dim=-1)
    teacher = probs.mean(dim=0).detach()
    return (probs * (probs.clamp_min(1e-6).log() - teacher.clamp_min(1e-6).log().unsqueeze(0))).sum(dim=-1).mean()


def sao_tpt_objective(model, output, anchor_output, args):
    cached = model.cached_outputs
    loss_tpt = avg_entropy(output)
    loss_sem = sao_semantic_orth_loss(
        cached["text_features"],
        cached["anchor_text_features"],
        output,
        anchor_output,
        args,
    )
    anchor_conf = anchor_output.softmax(dim=-1).max(dim=-1).values.mean().detach()
    loss_anchor = sao_anchor_kl_loss(output, anchor_output, args)
    loss_calib = sao_calibration_loss(output, anchor_output, args)
    loss_view = sao_view_consistency_loss(output)
    loss_pull = model.prompt_learner.get_delta_norm(gate_override=cached.get("sample_gate"))
    return (
        loss_tpt
        + args.sao_lambda_sem * loss_sem
        + args.sao_lambda_anchor * anchor_conf * loss_anchor
        + args.sao_lambda_calib * loss_calib
        + args.sao_lambda_view * loss_view
        + args.sao_lambda_pull * loss_pull
    )


def eat_topology_preserving_dispersion(text_features, anchor_text_features, args):
    text_features = F.normalize(text_features, dim=-1)
    anchor_text_features = F.normalize(anchor_text_features, dim=-1)
    
    S_current = text_features @ text_features.t()
    S_anchor = (anchor_text_features @ anchor_text_features.t()).detach()
    num_classes = S_current.shape[0]
    device = S_current.device
    off_diag = ~torch.eye(num_classes, dtype=torch.bool, device=device)
    mode = getattr(args, "eat_text_loss", "hard_mask")

    if mode == "none":
        return S_current.new_tensor(0.0)

    if mode == "hard_mask":
        threshold = getattr(args, "eat_dispersion_thresh", 0.5)
        weights = (S_anchor < threshold).float() * off_diag.float()
        return (S_current.pow(2) * weights).sum() / weights.sum().clamp_min(1.0)

    if mode == "continuous":
        weights = (1.0 - S_anchor).clamp(min=0.0) * off_diag.float()
        return (S_current.pow(2) * weights).sum() / weights.sum().clamp_min(1.0)

    if mode == "distill":
        diff = (S_current - S_anchor) * off_diag.float()
        return diff.pow(2).sum() / off_diag.float().sum().clamp_min(1.0)

    if mode == "otpt_full":
        return S_current[off_diag].pow(2).mean()

    if mode == "random_mask":
        threshold = getattr(args, "eat_dispersion_thresh", 0.5)
        hard_mask = (S_anchor < threshold) & off_diag
        num_selected = int(hard_mask.sum().item())
        if num_selected <= 0:
            return S_current.new_tensor(0.0)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(getattr(args, "eat_random_mask_seed", args.seed)) + num_classes * 997)
        random_scores = torch.rand((num_classes, num_classes), device=device, generator=generator)
        random_scores = random_scores.masked_fill(~off_diag, -1.0)
        flat_idx = torch.topk(random_scores.flatten(), k=min(num_selected, off_diag.sum().item())).indices
        weights = torch.zeros_like(S_current)
        weights.flatten()[flat_idx] = 1.0
        return (S_current.pow(2) * weights).sum() / weights.sum().clamp_min(1.0)

    raise ValueError(f"Unknown --eat_text_loss: {mode}")


def anchor_global_distribution(anchor_logits):
    return anchor_logits.softmax(dim=-1).mean(dim=0).detach()


def class_focus_from_anchor(anchor_probs, args):
    top_m = int(getattr(args, "eat_text_topm", 0))
    if top_m <= 0 or top_m >= anchor_probs.numel():
        focus = anchor_probs
    else:
        focus = torch.zeros_like(anchor_probs)
        top_idx = anchor_probs.topk(top_m).indices
        focus[top_idx] = anchor_probs[top_idx]
    return focus / focus.sum().clamp_min(1e-6)


def eat_sample_aware_structure_distill(text_features, anchor_text_features, anchor_probs, args):
    text_features = F.normalize(text_features, dim=-1)
    anchor_text_features = F.normalize(anchor_text_features, dim=-1)

    current_sim = text_features @ text_features.t()
    anchor_sim = (anchor_text_features @ anchor_text_features.t()).detach()
    focus = class_focus_from_anchor(anchor_probs, args)
    pair_weights = torch.outer(focus, focus)
    off_diag = ~torch.eye(current_sim.shape[0], dtype=torch.bool, device=current_sim.device)

    loss_map = (current_sim - anchor_sim).pow(2) * pair_weights
    return loss_map[off_diag].sum() / pair_weights[off_diag].sum().clamp_min(1e-6)


def eat_text_residual_anchoring(text_features, anchor_text_features, anchor_probs, args):
    text_features = F.normalize(text_features, dim=-1)
    anchor_text_features = F.normalize(anchor_text_features, dim=-1)

    focus = class_focus_from_anchor(anchor_probs, args)
    per_class_drift = (text_features - anchor_text_features.detach()).pow(2).sum(dim=-1)
    return (focus * per_class_drift).sum()


def eat_anchor_consistency_loss(adapt_probs, anchor_probs, args):
    top_m = int(getattr(args, "eat_anchor_topm", 0))
    if top_m > 0 and top_m < adapt_probs.numel():
        top_idx = torch.unique(torch.cat([adapt_probs.topk(top_m).indices, anchor_probs.topk(top_m).indices]))
        adapt_probs = adapt_probs[top_idx]
        anchor_probs = anchor_probs[top_idx]
        adapt_probs = adapt_probs / adapt_probs.sum().clamp_min(1e-6)
        anchor_probs = anchor_probs / anchor_probs.sum().clamp_min(1e-6)
    return (adapt_probs * (adapt_probs.clamp_min(1e-6).log() - anchor_probs.clamp_min(1e-6).log())).sum()


def eat_adaptive_all_view_loss(mix_logits, anchor_logits, args):
    probs_all = mix_logits.softmax(dim=-1)
    num_views = probs_all.shape[0]
    anchor_global = anchor_global_distribution(anchor_logits)

    tau = max(float(getattr(args, "eat_tau_router", 1.0)), 1e-6)
    kl_divs = torch.sum(
        probs_all * torch.log((probs_all + 1e-6) / (anchor_global.unsqueeze(0) + 1e-6)),
        dim=-1,
    )
    kl_weights = torch.exp(-kl_divs / tau)
    kl_weights = kl_weights / kl_weights.sum().clamp_min(1e-6)

    uniform_weights = torch.full_like(kl_weights, 1.0 / max(num_views, 1))
    anchor_entropy = normalized_entropy_from_logits(anchor_global.unsqueeze(0)).squeeze(0)
    confidence = 1.0 - anchor_entropy
    pivot = float(getattr(args, "eat_visual_conf_threshold", 0.55))
    temp = max(float(getattr(args, "eat_visual_mix_temp", 0.1)), 1e-6)
    alpha = torch.sigmoid((confidence - pivot) / temp)
    mixed_weights = alpha * kl_weights + (1.0 - alpha) * uniform_weights
    mixed_weights = mixed_weights / mixed_weights.sum().clamp_min(1e-6)

    weighted_probs = torch.sum(probs_all * mixed_weights.unsqueeze(-1), dim=0)
    entropy_loss = -torch.sum(weighted_probs * torch.log(weighted_probs + 1e-6))

    return entropy_loss, weighted_probs, anchor_global, alpha


def eat_safe_anchor_residual_loss(mix_logits, anchor_logits, args):
    probs_all = mix_logits.softmax(dim=-1)
    num_views = probs_all.shape[0]
    anchor_global = anchor_global_distribution(anchor_logits)

    q_uniform = probs_all.mean(dim=0)

    tau = max(float(getattr(args, "eat_tau_router", 1.0)), 1e-6)
    kl_divs = torch.sum(
        probs_all * torch.log((probs_all + 1e-6) / (anchor_global.unsqueeze(0) + 1e-6)),
        dim=-1,
    )
    kl_weights = torch.exp(-kl_divs / tau)
    kl_weights = kl_weights / kl_weights.sum().clamp_min(1e-6)
    q_kl = torch.sum(probs_all * kl_weights.unsqueeze(-1), dim=0)

    # Only use the anchor-guided residual when it is both anchor-consistent and not too peaky.
    kl_uniform_to_anchor = eat_anchor_consistency_loss(q_uniform.detach(), anchor_global, args)
    kl_kl_to_anchor = eat_anchor_consistency_loss(q_kl.detach(), anchor_global, args)
    anchor_gain = kl_uniform_to_anchor - kl_kl_to_anchor
    ess_ratio = (1.0 / kl_weights.detach().pow(2).sum().clamp_min(1e-6)) / max(float(num_views), 1.0)

    gain_margin = float(getattr(args, "eat_safe_gain_margin", 0.0))
    gain_temp = max(float(getattr(args, "eat_safe_gain_temp", 0.02)), 1e-6)
    ess_min = float(getattr(args, "eat_safe_min_ess", 0.35))
    ess_temp = max(float(getattr(args, "eat_safe_ess_temp", 0.08)), 1e-6)
    gain_gate = torch.sigmoid((anchor_gain - gain_margin) / gain_temp)
    ess_gate = torch.sigmoid((ess_ratio - ess_min) / ess_temp)
    beta = (gain_gate * ess_gate).detach()

    mixed_probs = q_uniform + beta * (q_kl - q_uniform)
    mixed_probs = mixed_probs / mixed_probs.sum().clamp_min(1e-6)
    entropy_loss = -torch.sum(mixed_probs * torch.log(mixed_probs + 1e-6))

    return entropy_loss, mixed_probs, anchor_global, beta


def eat_visual_entropy_loss(mix_logits, anchor_logits, args):
    mode = getattr(args, "eat_visual_loss", "kl_soft")

    if mode in ["tpt_entropy", "uniform"]:
        return avg_entropy(mix_logits)

    if mode == "adaptive_kl":
        entropy_loss, _, _, _ = eat_adaptive_all_view_loss(mix_logits, anchor_logits, args)
        return entropy_loss

    if mode == "safe_kl":
        entropy_loss, _, _, _ = eat_safe_anchor_residual_loss(mix_logits, anchor_logits, args)
        return entropy_loss

    P_all = mix_logits.softmax(dim=-1)
    tau = max(float(getattr(args, "eat_tau_router", 1.0)), 1e-6)

    if mode == "kl_soft":
        P_anchor = anchor_logits.softmax(dim=-1).mean(dim=0, keepdim=True).detach()
        kl_divs = torch.sum(
            P_all * torch.log((P_all + 1e-5) / (P_anchor.expand_as(P_all) + 1e-5)),
            dim=-1,
        )
        weights = torch.exp(-kl_divs / tau)
    elif mode == "anchor_reward":
        P_anchor = anchor_logits.softmax(dim=-1).mean(dim=0).detach()
        rewards = (P_all * P_anchor.unsqueeze(0)).sum(dim=-1)
        weights = torch.exp(rewards / tau)
    else:
        raise ValueError(f"Unknown --eat_visual_loss: {mode}")

    weights = weights / weights.sum().clamp_min(1e-6)
    weighted_avg_prob = torch.sum(P_all * weights.unsqueeze(-1), dim=0)
    return -torch.sum(weighted_avg_prob * torch.log(weighted_avg_prob + 1e-5))

def eat_dual_guided_objective(model, mix_logits, anchor_logits, args):
    cached = model.cached_outputs
    text_features = cached["text_features"] # [num_classes, dim]
    anchor_text_features = cached["anchor_text_features"]
    anchor_probs = anchor_global_distribution(anchor_logits)

    visual_loss_mode = getattr(args, "eat_visual_loss", "kl_soft")
    if visual_loss_mode == "adaptive_kl":
        loss_ent, adapt_probs, anchor_probs, alpha = eat_adaptive_all_view_loss(mix_logits, anchor_logits, args)
    elif visual_loss_mode == "safe_kl":
        loss_ent, adapt_probs, anchor_probs, alpha = eat_safe_anchor_residual_loss(mix_logits, anchor_logits, args)
    else:
        loss_ent = eat_visual_entropy_loss(mix_logits, anchor_logits, args)
        adapt_probs = mix_logits.softmax(dim=-1).mean(dim=0)
        alpha = normalized_entropy_from_logits(anchor_probs.unsqueeze(0)).new_tensor(1.0)

    text_loss_mode = getattr(args, "eat_text_loss", "hard_mask")
    if text_loss_mode == "sample_distill":
        loss_text = eat_sample_aware_structure_distill(text_features, anchor_text_features, anchor_probs, args)
    elif text_loss_mode == "residual_anchor":
        loss_text = eat_text_residual_anchoring(text_features, anchor_text_features, anchor_probs, args)
    elif text_loss_mode == "none":
        loss_text = loss_ent.new_tensor(0.0)
    else:
        loss_text = eat_topology_preserving_dispersion(text_features, anchor_text_features, args)

    lambda_text = getattr(args, "eat_lambda_disp", 0.1)
    loss = loss_ent + lambda_text * loss_text

    lambda_anchor = float(getattr(args, "eat_lambda_anchor", 0.0))
    if lambda_anchor > 0:
        loss = loss + lambda_anchor * alpha * eat_anchor_consistency_loss(adapt_probs, anchor_probs, args)

    return loss

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
    adaptive_prompt_tpt = (
        getattr(args, "rasc_tpt", False)
        or getattr(args, "rag_tpt", False)
        or getattr(args, "eat_pt", False)
        or getattr(args, "sao_tpt", False)
    )
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
                    output, selected_idx = select_views_for_adaptation(output, args)
                   
                    softmax_out = torch.softmax(output,dim=-1)
                    soft_mean = torch.mean(softmax_out,dim=0)
                                        
                    
                    number_of_class = output.shape[1]
                    
                if adaptive_prompt_tpt:
                    anchor_output = model.cached_outputs["anchor_logits"]
                    if selected_idx is not None:
                        anchor_output = anchor_output[selected_idx]
                    if getattr(args, "sao_tpt", False):
                        loss = sao_tpt_objective(
                            model,
                            output,
                            anchor_output,
                            args,
                        )
                    elif getattr(args, "eat_pt", False):
                        loss = eat_dual_guided_objective(
                            model,
                            output,
                            anchor_output,
                            args,
                        )
                    elif getattr(args, "rag_tpt", False):
                        neighbor_probs = model.cached_outputs.get("neighbor_probs")
                        neighbor_reliability = model.cached_outputs.get("neighbor_reliability")
                        loss, _, _, _ = rag_adaptive_objective(
                            output,
                            anchor_output,
                            neighbor_probs,
                            neighbor_reliability,
                            args,
                        )
                    else:
                        loss = rasc_consistency_loss(output, anchor_output)
                else:
                    loss = avg_entropy(output)
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

        if adaptive_prompt_tpt:
            cached = model.cached_outputs
            
            if getattr(args, "sao_tpt", False):
                # SAO-TPT already combines semantic, anchor, calibration, view and pull losses.
                pass
            elif getattr(args, "eat_pt", False):
                # We already computed the text dispersion loss inside eat_dual_guided_objective
                # Just add residual loss
                residual_loss = model.prompt_learner.get_delta_norm(gate_override=cached.get("sample_gate"))
                loss = loss + args.eat_lambda_res * residual_loss
            else:
                semantic_loss = rasc_semantic_loss(cached["text_features"], cached["anchor_text_features"])
                residual_loss = model.prompt_learner.get_delta_norm(gate_override=cached.get("sample_gate"))
                trust_loss = rasc_trust_loss(cached["raw_logits"], cached["anchor_logits"])
                if getattr(args, "rag_tpt", False):
                    neighbor_loss = rag_neighbor_loss(
                        cached["raw_logits"],
                        cached.get("neighbor_probs"),
                        cached.get("neighbor_reliability"),
                        args,
                    )
                    loss = (
                        loss
                        + args.rag_lambda_sem * semantic_loss
                        + args.rag_lambda_pull * residual_loss
                        + args.rag_lambda_trust * trust_loss
                        + args.rag_lambda_neighbor * neighbor_loss
                    )
                else:
                    loss = (
                        loss
                        + args.rasc_lambda_sem * semantic_loss
                        + args.rasc_lambda_pull * residual_loss
                        + args.rasc_lambda_trust * trust_loss
                    )

        if 'otpt' in args.run_type and not (getattr(args, "rasc_tpt", False) or getattr(args, "sao_tpt", False)):
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
        model = get_coop(
            args.arch,
            args.test_sets,
            args.gpu,
            args.n_ctx,
            args.ctx_init,
            args.disp_cons,
            gate_init=args.gate_init,
            gate_temp=args.gate_temp,
        )
        if args.load is not None:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            #pre-trained weights
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner.set_anchor(pretrained_ctx)

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
            "result_summary_{}sets.txt".format(len(dataset_ids)),
        )
    directory = os.path.dirname(custom_path)
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Check if the file exists
    file_exists = os.path.isfile(custom_path)
    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Open the file in append mode if it exists, otherwise in write mode
    
    print("params: nstep\tlr\tbs")
    print("params: {}\t    {}\t{}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="\t")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="\t")
    print("\n")

    with open(custom_path, 'a' if file_exists else 'w') as f:
        if not file_exists:
            f.write(f"[{current_datetime}] params: nstep={args.tta_steps} lr={args.lr} bs={args.batch_size}\n")
            f.write("Seed\t[set_id]\tTop-1 acc.\tTop-5 acc.\tAccuracy\tECE.\n")
        
        for id in dataset_ids:
            top1 = "{:.2f}".format(results[id][0])
            top5 = "{:.2f}".format(results[id][1])
            acc = "{:.2f}".format(accuracy_data[id][0])
            ece = "{:.2f}".format(ece_data[id][0])
            f.write(f"{args.seed}\t{id}\t{top1}\t{top5}\t{acc}\t{ece}\n")

        
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

    if 'otpt' in args.run_type:
        model.l2_norm_cal = True
    else:
        model.l2_norm_cal = False

    #initializing result_dictionary    
    result_dict = {'max_confidence': [], 'prediction': [], 'label': []}  
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
 
        result_dict['max_confidence'].append(max_confidence.item())
        result_dict['prediction'].append(max_index.item())
        result_dict['label'].append(target.item())
        if getattr(args, "rag_tpt", False) and not args.cocoop:
            model.update_neighbor_memory_from_cache(args)
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
    parser.add_argument(
        '--view_selection',
        default='hard_toprho',
        choices=['hard_toprho', 'all'],
        help='how augmented views are selected before adaptation',
    )
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
    parser.add_argument('--rasc_tpt', action='store_true', default=False, help='enable residual-anchored semantic-calibrated TPT')
    parser.add_argument('--gate_init', default=0.1, type=float, help='initial residual gate value')
    parser.add_argument('--gate_temp', default=2.0, type=float, help='temperature for the residual gate sigmoid')
    parser.add_argument('--rasc_lambda_sem', default=0.25, type=float, help='weight for semantic calibration loss')
    parser.add_argument('--rasc_lambda_pull', default=0.02, type=float, help='weight for residual magnitude penalty')
    parser.add_argument('--rasc_lambda_trust', default=0.05, type=float, help='weight for trust-region loss')
    parser.add_argument('--rasc_base_tau', default=1.0, type=float, help='base temperature for adaptive scaling')
    parser.add_argument('--rasc_tau_uncertainty', default=0.5, type=float, help='uncertainty coefficient for adaptive temperature')
    parser.add_argument('--rasc_tau_agreement', default=0.5, type=float, help='anchor disagreement coefficient for adaptive temperature')
    parser.add_argument('--rasc_use_temp', action='store_true', default=False, help='enable adaptive temperature scaling for RASC-TPT inference')
    parser.add_argument('--rag_tpt', action='store_true', default=False, help='enable relation-aware and neighborhood-guided TPT')
    parser.add_argument('--rag_lambda_sem', default=0.25, type=float, help='weight for relation-aware semantic loss')
    parser.add_argument('--rag_lambda_pull', default=0.02, type=float, help='weight for residual magnitude penalty')
    parser.add_argument('--rag_lambda_trust', default=0.05, type=float, help='weight for anchor trust-region loss')
    parser.add_argument('--rag_lambda_neighbor', default=0.08, type=float, help='weight for neighborhood prototype trust loss')
    parser.add_argument('--rag_memory_size', default=64, type=int, help='number of confident samples cached in neighborhood memory')
    parser.add_argument('--rag_memory_thresh', default=0.55, type=float, help='minimum confidence to add a sample into neighborhood memory')
    parser.add_argument('--rag_topk_neighbors', default=8, type=int, help='number of nearest neighbors used to build neighborhood prototypes')
    parser.add_argument('--rag_neighbor_temp', default=0.07, type=float, help='temperature for neighborhood weighting')
    parser.add_argument('--rag_neighbor_activate', default=0.55, type=float, help='minimum neighbor reliability before prototype trust becomes active')
    parser.add_argument('--rag_gate_conf', default=2.0, type=float, help='confidence coefficient for sample-wise gate bias')
    parser.add_argument('--rag_gate_entropy', default=1.5, type=float, help='entropy coefficient for sample-wise gate bias')
    parser.add_argument('--rag_gate_neighbor', default=1.0, type=float, help='neighbor agreement coefficient for sample-wise gate bias')
    parser.add_argument('--rag_gate_conf_center', default=0.45, type=float, help='confidence center used for sample-wise gate bias')
    parser.add_argument('--rag_gate_agreement_center', default=0.45, type=float, help='neighbor agreement center used for sample-wise gate bias')
    parser.add_argument('--rag_gate_entropy_margin', default=0.35, type=float, help='entropy margin before uncertainty starts suppressing gate')
    parser.add_argument('--rag_memory_adapt_ratio', default=0.35, type=float, help='how much adapted prediction contributes when writing to neighborhood memory')
    parser.add_argument('--rag_lambda_cons', default=0.35, type=float, help='base weight for reliable teacher consistency in RAG-TPT')
    parser.add_argument('--rag_consistency_base', default=0.15, type=float, help='minimum teacher consistency strength in RAG-TPT')
    parser.add_argument('--rag_teacher_base', default=0.05, type=float, help='base neighbor teacher mixing ratio')
    parser.add_argument('--rag_teacher_neighbor', default=0.35, type=float, help='additional neighbor teacher mixing ratio from reliability')
    parser.add_argument('--rag_entropy_boost', default=0.5, type=float, help='extra entropy pressure when anchor confidence is low')
    parser.add_argument('--rag_confidence_target', default=0.6, type=float, help='target anchor confidence used to balance entropy and teacher consistency')
    parser.add_argument('--rag_use_temp', action='store_true', default=False, help='enable adaptive temperature scaling for RAG-TPT inference')
    parser.add_argument('--rag_base_tau', default=1.0, type=float, help='base temperature for RAG-TPT')
    parser.add_argument('--rag_tau_uncertainty', default=0.5, type=float, help='uncertainty coefficient for RAG-TPT temperature')
    parser.add_argument('--rag_tau_agreement', default=0.5, type=float, help='agreement coefficient for RAG-TPT temperature')
    # ------------------------------------------------

    # EAT-PT specific arguments
    parser.add_argument('--eat_pt', action='store_true', help='Enable Evidence-Adaptive Test-Time Prompt Tuning (EAT-PT)')
    parser.add_argument('--eat_tau_router', default=1.0, type=float, help='Temperature for evidence router')
    parser.add_argument('--eat_temp_sharp', default=0.5, type=float, help='Temperature for self-evidence sharpening')
    parser.add_argument('--eat_dispersion_thresh', default=0.5, type=float, help='Threshold for topology-preserving dispersion')
    parser.add_argument('--eat_lambda_disp', default=1.0, type=float, help='Weight for dispersion loss')
    parser.add_argument('--eat_lambda_res', default=0.1, type=float, help='Weight for residual prompt loss')
    parser.add_argument(
        '--eat_visual_loss',
        default='kl_soft',
        choices=['kl_soft', 'adaptive_kl', 'safe_kl', 'tpt_entropy', 'uniform', 'anchor_reward'],
        help='visual branch ablation used by EAT/DUAP-PT',
    )
    parser.add_argument(
        '--eat_text_loss',
        default='hard_mask',
        choices=['hard_mask', 'continuous', 'distill', 'sample_distill', 'residual_anchor', 'random_mask', 'otpt_full', 'none'],
        help='text branch ablation used by EAT/DUAP-PT',
    )
    parser.add_argument('--eat_random_mask_seed', default=12345, type=int, help='seed for same-sparsity random text mask')
    parser.add_argument('--eat_alpha_visual', default=0.5, type=float, help='Weight for visual vs semantic KNN in memory bank')
    parser.add_argument('--eat_ortho_topk', default=5, type=int, help='Top-K safe zone for Activation-Guided Orthogonal Repulsion')
    parser.add_argument('--eat_neighbor_threshold', default=0.7, type=float, help='Threshold to reject noisy neighbors')
    parser.add_argument('--eat_ent_threshold', default=1.5, type=float, help='Entropy threshold for Base Bias activation in Router')
    parser.add_argument('--eat_base_bias_self', default=0.6, type=float, help='Base weight for self-evidence when confident')
    parser.add_argument('--eat_visual_conf_threshold', default=0.55, type=float, help='anchor confidence pivot for adaptive KL vs uniform interpolation')
    parser.add_argument('--eat_visual_mix_temp', default=0.10, type=float, help='temperature for adaptive KL/uniform interpolation')
    parser.add_argument('--eat_safe_gain_margin', default=0.0, type=float, help='minimum anchor-consistency gain before KL residual is activated')
    parser.add_argument('--eat_safe_gain_temp', default=0.02, type=float, help='temperature for safe-KL gain gating')
    parser.add_argument('--eat_safe_min_ess', default=0.35, type=float, help='minimum effective sample size ratio before KL residual is trusted')
    parser.add_argument('--eat_safe_ess_temp', default=0.08, type=float, help='temperature for safe-KL ESS gating')
    parser.add_argument('--eat_lambda_anchor', default=0.05, type=float, help='weight for aggregated prediction to stay close to zero-shot anchor')
    parser.add_argument('--eat_anchor_topm', default=20, type=int, help='number of top classes used by anchor consistency; <=0 uses full distribution')
    parser.add_argument('--eat_text_topm', default=20, type=int, help='number of anchor top classes used to weight sample-aware text structure distillation')

    # SAO-TPT: Semantic-Aware Orthogonal Test-time Prompt Tuning
    parser.add_argument('--sao_tpt', action='store_true', default=False, help='enable semantic-aware local orthogonal TPT')
    parser.add_argument('--sao_topk', default=10, type=int, help='number of semantic neighbors preserved in target Gram matrix')
    parser.add_argument('--sao_topm', default=20, type=int, help='number of current sample related classes for local Gram weighting')
    parser.add_argument('--sao_anchor_topm', default=20, type=int, help='number of top classes used in anchor KL; <=0 uses full distribution')
    parser.add_argument('--sao_alpha', default=0.5, type=float, help='scale for preserved semantic cosine in target Gram matrix')
    parser.add_argument('--sao_beta', default=2.0, type=float, help='extra local Gram weight for top-M related classes')
    parser.add_argument('--sao_lambda_sem', default=0.1, type=float, help='weight for semantic-aware orthogonal loss')
    parser.add_argument('--sao_lambda_anchor', default=0.05, type=float, help='weight for confidence-gated zero-shot anchor KL')
    parser.add_argument('--sao_lambda_calib', default=0.05, type=float, help='weight for unsupervised over-confidence guard')
    parser.add_argument('--sao_lambda_view', default=0.05, type=float, help='weight for multi-view consistency')
    parser.add_argument('--sao_lambda_pull', default=0.01, type=float, help='weight for residual prompt magnitude penalty')
    parser.add_argument('--sao_conf_margin', default=0.05, type=float, help='allowed confidence increase over zero-shot anchor before calibration penalty')
    parser.add_argument('--sao_use_temp', action='store_true', default=False, help='enable zero-shot entropy driven adaptive temperature for SAO-TPT inference')
    parser.add_argument('--sao_base_tau', default=1.0, type=float, help='base temperature for SAO-TPT')
    parser.add_argument('--sao_tau_uncertainty', default=0.5, type=float, help='zero-shot uncertainty coefficient for SAO-TPT temperature')

    args = parser.parse_args()
    
    if 'otpt' not in args.run_type:
        args.lambda_term = 0.0

    #result_dict = {'max_confidence': [], 'prediction': [], 'label': []}
    main(args)
    #acc, ece = Calculator(result_dict)
