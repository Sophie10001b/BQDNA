import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist

from functools import cache
from typing import Sequence, Tuple, Dict, List, Any, Optional, override
from einops import rearrange, repeat
from .configuration_vq import VQConfig

@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t: torch.Tensor):
    if not is_distributed():
        return t

    dist.all_reduce(t)
    t = t / dist.get_world_size()
    return t

def maybe_distributed_std(t: torch.Tensor):
    if not is_distributed():
        return torch.std(t)
    
    t_mean = maybe_distributed_mean(t.mean())
    t_numel = torch.tensor([t.numel()], dtype=torch.int64, device=t.device)
    dist.all_reduce(t_numel)

    t_sum = (t - t_mean) ** 2
    dist.all_reduce(t_sum)

    return (t_sum / (t_numel - 1)).sqrt()

def clamp_log(x: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    return x.clamp(min=eps).log()

def maybe_distributed_min(t: torch.Tensor):
    if not is_distributed():
        return t.min()

    t_min = t.min()
    dist.all_reduce(t_min, op=dist.ReduceOp.MIN)
    return t_min


@torch.no_grad()
def linear_init(
    linear: nn.Linear,
    distribution: Optional[str]='normal',
    zero_bias: Optional[bool]=False,
    gain: Optional[float]=1.0
) ->None:
    if distribution == 'normal':
        nn.init.xavier_normal_(linear.weight, gain=gain)
    elif distribution == 'uniform':
        nn.init.xavier_uniform_(linear.weight, gain=gain)
    if linear.bias is not None:
        if zero_bias:
            nn.init.zeros_(linear.bias)
        else:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(linear.bias, -bound, bound)

@torch.no_grad()
def embedding_init(embedding: nn.Embedding, distribution: str='normal') ->None:
    if distribution == 'normal':
        fan_out = embedding.weight.size(1)
        std = 1.0 * math.sqrt(1.0 / float(fan_out))
        nn.init.normal_(embedding.weight, 0., std)
    elif distribution == 'uniform':
        nn.init.uniform_(embedding.weight, -1 / embedding.num_embeddings, 1 / embedding.num_embeddings)
    else: raise ValueError(f"Distribution {distribution} not support for embedding initialization")

    if embedding.padding_idx is not None:
        embedding.weight[embedding.padding_idx].fill_(0)
    

def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype = dtype, device = device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)

@torch.no_grad()
def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
    sample_fn = batched_sample_vectors
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        dists = -torch.cdist(samples, means)

        buckets = torch.argmax(dists, dim = -1)
        bins = batched_bincount(buckets, minlength = num_clusters)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype = dtype)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d = dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins


class EMACache(nn.Module):
    def __init__(self, init_value: torch.Tensor, decay: float):
        super().__init__()
        self.decay = decay
        self.count = 0
        self.register_buffer("value", init_value, persistent=False)
    
    def forward(self, x: torch.Tensor):
        self.count += 1
        self.value = self.decay * self.value + (1 - self.decay) * x
        return self.value / (1 - self.decay ** self.count)
    

class VQBase(nn.Module):
    def __init__(self, config: VQConfig):
        super().__init__()

        self.codebook_size = config.codebook_size
        self.codebook_dim = config.latent_size
        self.latent_size = config.latent_size

        #losses
        self.diversity_gamma = config.diversity_gamma
        self.distance_temperature = config.distance_temperature
        self.diversity_loss_factor = config.diversity_loss_factor
        self.commit_loss_factor = config.commit_loss_factor
        self.register_buffer('zero', torch.zeros((1,), dtype=torch.float), persistent=False)

        self.config = config
    
    def get_codebook(self) -> torch.Tensor:
        raise NotImplementedError()

    def index_to_code(self, indices: torch.LongTensor) -> torch.Tensor:
        return torch.nn.functional.embedding(indices, self.get_codebook())
    
    def forward(self, x: torch.Tensor):
        raise NotImplementedError()



class OptVQ(VQBase):
    def __init__(self, config: VQConfig):
        super().__init__(config)

        self.num_heads = config.num_heads

        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim // self.num_heads)
        self._init_weights()

        self.ema_cluster_size = EMACache(torch.zeros(config.codebook_size), decay=config.ema_gamma)
        self.ema_weight = EMACache(self.codebook.weight.clone(), decay=config.ema_gamma)

        self.eps = config.eps
        self.optvq_eps = config.optvq_eps
        self.optvq_niters = config.optvq_niters
    
    def _init_weights(self):
        embedding_init(self.codebook, distribution='uniform')
        self.codebook.requires_grad_(False)
    
    @override
    def get_codebook(self):
        return self.codebook.weight
    
    def sinkhorn(self, cost: torch.Tensor):
        """
        Sinkhorn algorithm.
        Args:
            cost (Tensor): shape with (B, K)
        """
        distributed = is_distributed()
        num_gpus = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        Q = torch.exp(- cost * self.optvq_eps).t() # (K, B)
        if is_distributed:
            B = Q.size(1) * num_gpus
        else:
            B = Q.size(1)
        K = Q.size(0)

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if distributed:
            torch.distributed.all_reduce(sum_Q)
        Q /= (sum_Q + 1e-8)

        for _ in range(self.optvq_niters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if distributed:
                torch.distributed.all_reduce(sum_of_rows)
            Q /= (sum_of_rows + 1e-8)
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= (torch.sum(Q, dim=0, keepdim=True) + 1e-8)
            Q /= B
        
        Q *= B # the columns must sum to 1 so that Q is an assignment
        return Q.t() # (B, K)
    
    def get_sinkhorn_idx(self, x: torch.Tensor):
        # x: [batch_size, seq_len, latent_size]

        global_min = maybe_distributed_min(x)
        global_std = maybe_distributed_std(x)
        cost = (x - global_min) / (global_std + self.eps)
        cost = cost - maybe_distributed_min(cost)

        indices = self.sinkhorn(cost)
        indices = indices.argmax(dim=-1)

        return indices
    
    @override
    def forward(self, x: torch.Tensor):
        x_flat = rearrange(x, "B L (nH dH) -> (B L nH) dH", nH=self.num_heads, dH=self.latent_size // self.num_heads)

        original_dtype = x.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            with torch.no_grad():
                dist = torch.cdist(x_flat.float(), self.get_codebook().float(), p=2)
                indices = self.get_sinkhorn_idx(dist)

        quantized = self.index_to_code(indices)

        if self.training:
            # update ema
            with torch.no_grad():
                indices_onthot = torch.nn.functional.one_hot(indices, num_classes=self.codebook_size)
                cluster_size = indices_onthot.sum(dim=0)
                updated_ema_cluster_size: torch.Tensor = self.ema_cluster_size(cluster_size)

                total_cluster_size = updated_ema_cluster_size.sum()
                updated_ema_cluster_size = (updated_ema_cluster_size + self.eps) / (total_cluster_size + self.config.codebook_size * self.eps) * total_cluster_size
                new_ema_weight = indices_onthot.float().transpose(0, 1) @ x_flat
                updated_ema_weight = self.ema_weight(new_ema_weight)

                self.codebook.weight.data = updated_ema_weight / updated_ema_cluster_size.unsqueeze(-1)
            
            x_quant = x_flat + (quantized - x_flat).detach()
            commit_loss = torch.nn.functional.mse_loss(x_flat, quantized.detach())

            # diversity loss
            x_multihead = rearrange(x, "B L (nH dH) -> B (L nH) dH", nH=self.num_heads, dH=self.latent_size // self.num_heads)
            codebook = self.get_codebook()
            dist = -2 * (x_multihead @ codebook.transpose(-1, -2))
            prob = (-dist * self.distance_temperature).softmax(dim=-1)

            sample_entropy = (-prob * clamp_log(prob)).sum(-1).mean()
            batch_prob = maybe_distributed_mean(prob.mean(dim=0))
            codebook_entropy = (-batch_prob * clamp_log(batch_prob)).sum(-1).mean()
            entropy_aux_loss = sample_entropy - self.diversity_gamma * codebook_entropy
        
        else:
            x_quant = quantized
            commit_loss = self.zero
            entropy_aux_loss = self.zero
        
        aux_loss = entropy_aux_loss * self.diversity_loss_factor + commit_loss * self.commit_loss_factor
        return dict(
            quant=rearrange(x_quant, "(B L nH) dH -> B L (nH dH)", B=x.size(0), L=x.size(1), nH=self.num_heads, dH=self.latent_size // self.num_heads),
            indices=rearrange(indices, "(B L nH) -> B L nH", B=x.size(0), L=x.size(1), nH=self.num_heads),
            aux_loss=aux_loss
        )


class LFQ(VQBase):
    def __init__(self, config: VQConfig):
        super().__init__(config)

        self.codebook_dim = int(math.log2(self.codebook_size))

        self.downsample = nn.Linear(self.latent_size, self.codebook_dim)
        self.upsample = nn.Linear(self.codebook_dim, self.latent_size)

        # get codebook
        self.register_buffer('mask', torch.arange(self.codebook_dim - 1, -1, -1) ** 2, persistent=False)
        bits = ((torch.arange(self.codebook_size)[:, None].int() & self.mask) != 0).float()
        self.register_buffer('codebook', bits * 2 - 1, persistent=False)

        self._init_weights()
    
    def _init_weights(self):
        linear_init(self.downsample)
        linear_init(self.upsample)

    @override
    def get_codebook(self):
        return self.codebook
    
    @override
    def forward(self, x: torch.Tensor):
        x = self.downsample(x)

        original_dtype = x.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            x = x.float()
            codebook_val = torch.ones_like(x)
            quantized = torch.where(x > 0, codebook_val, -codebook_val)
            indices = ((quantized > 0).to(torch.int64) * self.mask.to(torch.int64)).sum(-1)

        x = x.to(original_dtype)

        if self.training:
            x_quant = x + (quantized - x).detach()
            codebook = self.codebook.float()

            # compute loss
            dist = -2 * (x @ codebook.transpose(-1, -2))
            prob = (-dist * self.distance_temperature).softmax(dim=-1)

            sample_entropy = (-prob * clamp_log(prob)).sum(-1).mean()
            batch_prob = maybe_distributed_mean(prob.mean(dim=0))
            codebook_entropy = (-batch_prob * clamp_log(batch_prob)).sum(-1).mean()

            entropy_aux_loss = sample_entropy - self.diversity_gamma * codebook_entropy
            # commit_loss = torch.nn.functional.mse_loss(x, quantized.detach())
        
        else:
            x_quant = quantized
            entropy_aux_loss = self.zero
            commit_loss = self.zero

        x_quant = self.upsample(x_quant)
        aux_loss = entropy_aux_loss * self.diversity_loss_factor

        return dict(
            quant=x_quant,
            indices=indices,
            aux_loss=aux_loss
        )

if __name__ == '__main__':
    config = VQConfig(latent_size=128, downsample_rate=5, codebook_size=16384)
    vq = LFQ(config)

    seq = torch.randn(16, 512, 128)
    res = vq(seq)
