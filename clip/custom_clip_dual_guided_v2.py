import os
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.covid_prompts import covid_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import ipdb

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT = os.path.expanduser("~/.cache/clip")

class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        #print("shape after prompt positional embedding:",x.shape)
        x = x.permute(1, 0, 2)  # NLD -> LND
        #print("shape after prompt permute:",x.shape)
        x = self.transformer(x)
        #print("shape after transformer:",x.shape)
        x = x.permute(1, 0, 2)  # LND -> NLD
        #print("shape after transformer permute:",x.shape)
        x = self.ln_final(x).type(self.dtype)
        #print("shape after tfinal_layer:",x.shape)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        #print("features after eot embeddings:",x.shape)
        return x


class PromptLearner(nn.Module):
    def __init__(
        self,
        clip_model,
        classnames,
        batch_size=None,
        n_ctx=16,
        ctx_init=None,
        ctx_position='end',
        learned_cls=False,
        gate_init=0.1,
        gate_temp=2.0,
    ):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx =  len(ctx_init.split(" ")) #it is for coop & TPT ctx is 4. 2 is for MAPLE 
            prompt = tokenize(ctx_init).to(self.device)
            #print("Initial prompt tokenized shape:",prompt.shape)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
                #print("Initial prompt embeded shape:",embedding.shape)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            #print("learnable ctx_vectors shape:",ctx_vectors.shape)
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix
        
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.register_buffer("ctx_anchor_state", ctx_vectors.detach().clone())
        self.register_buffer("ctx_init_state", torch.zeros_like(ctx_vectors))
        # residual prompt update to be optimized at test time
        self.ctx = nn.Parameter(torch.zeros_like(ctx_vectors))
        gate_init = float(min(max(gate_init, 1e-4), 1.0 - 1e-4))
        self.gate_temperature = gate_temp
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.register_buffer(
            "gate_logit_init_state",
            torch.tensor(gate_logit, dtype=dtype, device=self.device),
        )
        self.gate_logit = nn.Parameter(self.gate_logit_init_state.clone())

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized
            #print("learnable class shape:",self.cls.shape)
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        #print("concatenated combined prompt class tokenized shape:",tokenized_prompts.shape)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            #print("concatenated combined prompt class embeded shape:",embedding.shape)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens     #class name length
        self.class_token_position = ctx_position
        self.n_cls = n_cls       #total number of class
        self.n_ctx = n_ctx       #intial prompt context length
        self.classnames = classnames

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        self.gate_logit.copy_(self.gate_logit_init_state)
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def set_anchor(self, ctx_vectors):
        if ctx_vectors.shape != self.ctx_anchor_state.shape:
            raise ValueError("Anchor prompt shape mismatch")
        self.ctx_anchor_state.copy_(ctx_vectors.detach())
        self.ctx_init_state.zero_()
        self.reset()

    def get_gate(self, gate_bias=None):
        gate_logit = self.gate_logit
        if gate_bias is not None:
            gate_logit = gate_logit + gate_bias.to(device=gate_logit.device, dtype=gate_logit.dtype)
        return torch.sigmoid(gate_logit / self.gate_temperature)

    def get_anchor_ctx(self):
        return self.ctx_anchor_state

    def get_delta_ctx(self):
        return self.ctx

    def get_mixed_ctx(self, gate_override=None):
        gate = gate_override if gate_override is not None else self.get_gate()
        return self.ctx_anchor_state + gate * self.ctx

    def get_delta_norm(self, gate_override=None):
        gate = gate_override if gate_override is not None else self.get_gate()
        return torch.linalg.norm(gate * self.ctx)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        #print("concatenated combined prompt class tokenized shape-reset:",tokenized_prompts.shape)
        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)
            #print("concatenated combined prompt class embeded shape-reset:",embedding.shape)
        self.token_prefix = embedding[:, :1, :]
        #print("token prefix shape:",self.token_prefix.shape)
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS
        #print("token suffix shape:",self.token_suffix.shape)
        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None, branch="mix", gate_override=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        elif branch == "anchor":
            ctx = self.get_anchor_ctx()
        elif branch == "delta":
            ctx = self.get_delta_ctx()
        else:
            ctx = self.get_mixed_ctx(gate_override=gate_override)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            #print("ctx learnable vector shape if ctx.dim==2:",ctx.shape)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
            #print("ctx learnable vector shape if ctx.size()[0]:",ctx.shape)
        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
                #print("Tuned prompt final-end:", prompts.shape)
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
            #print("Tuned prompt final -mid:", prompts.shape)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
            #print("Tuned prompt final-front:", prompts.shape)
        else:
            raise ValueError

        return prompts



class ClipTestTimeTuning(nn.Module):
    def __init__(
        self,
        device,
        classnames,
        batch_size,
        criterion='cosine',
        arch="ViT-L/14",
        n_ctx=16,
        ctx_init=None,
        ctx_position='end',
        learned_cls=False,
        gate_init=0.1,
        gate_temp=2.0,
    ):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(
            clip,
            classnames,
            batch_size,
            n_ctx,
            ctx_init,
            ctx_position,
            learned_cls,
            gate_init=gate_init,
            gate_temp=gate_temp,
        )
        self.criterion = criterion
        self.cached_outputs = {}
        self.reset_neighbor_memory()
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)
        self.reset_neighbor_memory()

    def reset_neighbor_memory(self):
        self.neighbor_feature_bank = []
        self.neighbor_prob_bank = []

    def get_text_features(self, branch="mix", gate_override=None):
        text_features = []
        prompts = self.prompt_learner(branch=branch, gate_override=gate_override)
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def get_neighbor_distribution(self, image_features, anchor_logits, args):
        if len(self.neighbor_feature_bank) == 0:
            return None, None, None
        bank_features = torch.stack(self.neighbor_feature_bank).to(image_features.device)
        bank_probs = torch.stack(self.neighbor_prob_bank).to(image_features.device)
        query = F.normalize(image_features.mean(dim=0, keepdim=True), dim=-1)
        
        # Dual-Modal KNN
        # 1. Visual Similarity
        visual_similarities = (query @ bank_features.t()).squeeze(0)
        
        # 2. Semantic Similarity (between current anchor prediction and memory prediction)
        anchor_probs = anchor_logits.softmax(dim=-1).mean(dim=0, keepdim=True)
        # Use Bhattacharyya distance or simple dot product for semantic alignment
        semantic_similarities = (anchor_probs @ bank_probs.t()).squeeze(0)
        
        # Combine similarities
        similarities = visual_similarities * args.eat_alpha_visual + semantic_similarities * (1 - args.eat_alpha_visual)

        topk = min(args.rag_topk_neighbors, similarities.numel())
        top_values, top_indices = similarities.topk(topk)
        
        # Threshold filtering (Dual-modal consistency check)
        mask = top_values > args.eat_neighbor_threshold
        if not mask.any():
            return None, None, None
            
        top_values = top_values[mask]
        top_indices = top_indices[mask]

        weights = (top_values / args.rag_neighbor_temp).softmax(dim=0)
        neighbor_probs = (weights.unsqueeze(-1) * bank_probs[top_indices]).sum(dim=0)
        
        # In EAT-PT, reliability is handled by the Evidence Router's entropy calculation,
        # but we keep a basic reliability score for legacy gate (if still used)
        sim_score = (((top_values + 1.0) / 2.0).clamp(0.0, 1.0) * weights).sum()
        consensus = 1.0 - (
            -(neighbor_probs.clamp_min(1e-6) * neighbor_probs.clamp_min(1e-6).log()).sum() / math.log(neighbor_probs.shape[-1])
        )
        reliability = 0.5 * sim_score + 0.5 * consensus
        return neighbor_probs, weights, reliability

    def compute_sample_gate(self, anchor_logits, neighbor_probs, neighbor_reliability, args):
        anchor_probs = anchor_logits.softmax(dim=-1)
        confidence = anchor_probs.max(dim=-1).values.mean()
        uncertainty = -(anchor_probs * anchor_probs.clamp_min(1e-6).log()).sum(dim=-1).mean()
        uncertainty = uncertainty / math.log(anchor_probs.shape[-1])
        if neighbor_probs is None:
            neighbor_agreement = anchor_probs.new_tensor(0.0)
        else:
            neighbor_agreement = (anchor_probs.mean(dim=0) * neighbor_probs).sum()
        if neighbor_reliability is None:
            neighbor_reliability = anchor_probs.new_tensor(0.0)
        uncertainty_penalty = F.relu(uncertainty - args.rag_gate_entropy_margin)
        neighbor_bonus = neighbor_reliability * (neighbor_agreement - args.rag_gate_agreement_center)
        gate_bias = (
            args.rag_gate_conf * (confidence - args.rag_gate_conf_center)
            - args.rag_gate_entropy * uncertainty_penalty
            + args.rag_gate_neighbor * neighbor_bonus
        )
        return self.prompt_learner.get_gate(gate_bias=gate_bias)

    def update_neighbor_memory(self, image_features, logits, args, anchor_logits=None):
        probs = logits.softmax(dim=-1).mean(dim=0)
        if anchor_logits is not None:
            anchor_probs = anchor_logits.softmax(dim=-1).mean(dim=0)
            probs = args.rag_memory_adapt_ratio * probs + (1.0 - args.rag_memory_adapt_ratio) * anchor_probs
            confidence = max(probs.max().item(), anchor_probs.max().item())
        else:
            confidence = probs.max().item()
        if confidence < args.rag_memory_thresh:
            return
        feature = F.normalize(image_features.mean(dim=0), dim=-1).detach().cpu()
        prob = probs.detach().cpu()
        self.neighbor_feature_bank.append(feature)
        self.neighbor_prob_bank.append(prob)
        if len(self.neighbor_feature_bank) > args.rag_memory_size:
            self.neighbor_feature_bank.pop(0)
            self.neighbor_prob_bank.pop(0)

    def update_neighbor_memory_from_cache(self, args):
        if "image_features" not in self.cached_outputs or "raw_logits" not in self.cached_outputs:
            return
        self.update_neighbor_memory(
            self.cached_outputs["image_features"],
            self.cached_outputs["raw_logits"],
            args,
            anchor_logits=self.cached_outputs.get("anchor_logits"),
        )

    def apply_adaptive_temperature(self, logits, anchor_logits, args):
        probs = logits.softmax(dim=-1)
        anchor_probs = anchor_logits.softmax(dim=-1)
        uncertainty = -(probs * probs.clamp_min(1e-6).log()).sum(dim=-1) / math.log(probs.shape[-1])
        anchor_uncertainty = -(anchor_probs * anchor_probs.clamp_min(1e-6).log()).sum(dim=-1) / math.log(anchor_probs.shape[-1])
        agreement = (probs * anchor_probs).sum(dim=-1)
        if getattr(args, "sao_tpt", False):
            base_tau = args.sao_base_tau
            tau = base_tau * (1.0 + args.sao_tau_uncertainty * anchor_uncertainty)
            tau = tau.clamp_min(1.0).unsqueeze(-1)
            self.cached_outputs["temperature"] = tau.detach()
            return logits / tau
        elif getattr(args, "rag_tpt", False):
            base_tau = args.rag_base_tau
            tau_uncertainty = args.rag_tau_uncertainty
            tau_agreement = args.rag_tau_agreement
        else:
            base_tau = args.rasc_base_tau
            tau_uncertainty = args.rasc_tau_uncertainty
            tau_agreement = args.rasc_tau_agreement
        tau = base_tau + tau_uncertainty * uncertainty + tau_agreement * (1.0 - agreement)
        tau = tau.clamp_min(1.0).unsqueeze(-1)
        self.cached_outputs["temperature"] = tau.detach()
        return logits / tau

    def inference(self, image,cons,args):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        if getattr(args, "rasc_tpt", False) or getattr(args, "rag_tpt", False) or getattr(args, "eat_pt", False) or getattr(args, "sao_tpt", False):
            anchor_text_features = self.get_text_features(branch="anchor")
            anchor_logits = logit_scale * image_features @ anchor_text_features.t()
            neighbor_probs, neighbor_weights, neighbor_reliability = None, None, None
            sample_gate = None
            if getattr(args, "rag_tpt", False) or getattr(args, "eat_pt", False):
                neighbor_probs, neighbor_weights, neighbor_reliability = self.get_neighbor_distribution(image_features, anchor_logits, args)
                sample_gate = self.compute_sample_gate(anchor_logits, neighbor_probs, neighbor_reliability, args)
            text_features = self.get_text_features(branch="mix", gate_override=sample_gate)
            logits = logit_scale * image_features @ text_features.t()
            gate_value = sample_gate if sample_gate is not None else self.prompt_learner.get_gate()
            self.cached_outputs = {
                "raw_logits": logits,
                "anchor_logits": anchor_logits,
                "text_features": text_features,
                "anchor_text_features": anchor_text_features,
                "gate": gate_value.detach(),
                "base_gate": self.prompt_learner.get_gate().detach(),
                "sample_gate": gate_value,
                "delta_norm": self.prompt_learner.get_delta_norm(gate_override=gate_value).detach(),
                "image_features": image_features.detach(),
                "neighbor_probs": None if neighbor_probs is None else neighbor_probs.detach(),
                "neighbor_weights": None if neighbor_weights is None else neighbor_weights.detach(),
                "neighbor_reliability": None if neighbor_reliability is None else neighbor_reliability.detach(),
            }
            if getattr(args, "rasc_use_temp", False) or getattr(args, "rag_use_temp", False) or getattr(args, "sao_use_temp", False):
                logits = self.apply_adaptive_temperature(logits, anchor_logits, args)
            self.textfeatures_ = text_features
        else:
            text_features = self.get_text_features()
            if self.l2_norm_cal:
                self.textfeatures_ = text_features
            logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward(self, input,cons,args):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input,cons,args)


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, cons, learned_cls=False, gate_init=0.1, gate_temp=2.0):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    elif test_set == 'C':
           classnames = covid_classes        
    else:
        classnames = imagenet_classes

    model = ClipTestTimeTuning(
        device,
        classnames,
        None,
        arch=clip_arch,
        n_ctx=n_ctx,
        ctx_init=ctx_init,
        learned_cls=learned_cls,
        gate_init=gate_init,
        gate_temp=gate_temp,
    )

    return model
