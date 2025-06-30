import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import transformers

from typing import Optional, Dict, Tuple, List, Union, Unpack, Sequence, Any
from torch.optim.lr_scheduler import LRScheduler
from flash_attn import (
    flash_attn_kvpacked_func,
    flash_attn_varlen_func
)
from flash_attn.layers.rotary import RotaryEmbedding, apply_rotary_emb
from flash_attn.ops.triton.layer_norm import RMSNorm
from flash_attn.modules.mlp import GatedMlp
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from einops import rearrange
from itertools import chain
from flash_attn.bert_padding import unpad_input
from mamba_ssm import Mamba2

from .vq.configuration_vq import VQConfig
from .vq.modeling_vq import VQModel
from .configuration_transformer import TransformerConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast, MaskedLMOutput, SequenceClassifierOutput

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

#########################################################
#                   --- model ---
#########################################################
class FullAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_base: int,
        dropout: float,
        layer_idx: int,
        window: int,
        **kwargs
    ):
        super(FullAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_q_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = hidden_size // num_heads
        self.dropout = dropout
        self.layer_idx = layer_idx

        self.qkv = nn.Linear(hidden_size, hidden_size + 2 * num_kv_heads * self.head_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary = RotaryEmbedding(dim=self.head_size, base=rotary_base)
        self.window = window

        self._init_weights()
    
    def _init_weights(self):
        for k, v in self.named_modules():
            if isinstance(v, nn.Linear): linear_init(v, zero_bias=True)
    
    def forward(self, x: torch.Tensor, causal: bool=False, return_attn_probs: bool=False):
        if self.window > 0:
            window = (self.window - 1, 0) if causal else (self.window // 2, self.window // 2)
        else: window = (-1, -1)

        qkv: torch.Tensor = self.qkv(x)
        qkv = rearrange(qkv, "B L (H D) -> B L H D", H=(self.num_q_heads + 2 * self.num_kv_heads), D=self.head_size)
        q, kv = torch.split(qkv, [self.num_q_heads, 2 * self.num_kv_heads], dim=-2)
        kv = rearrange(kv, "B L (C H) D -> B L C H D", C=2, H=self.num_kv_heads)

        q, kv = self.rotary(q, kv)

        attn_score = None
        if return_attn_probs:
            out, _, attn_score = flash_attn_kvpacked_func(q, kv, dropout_p=self.dropout if self.training else 0, causal=causal, return_attn_probs=return_attn_probs, window_size=window)
        else:
            out = flash_attn_kvpacked_func(q, kv, dropout_p=self.dropout if self.training else 0, causal=causal, return_attn_probs=return_attn_probs, window_size=window)
        out = self.out(rearrange(out, "B L H D -> B L (H D)"))
        
        return out, attn_score, None

class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int, window: int=-1, use_mamba: bool=False):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.eps)
        if not use_mamba:
            self.attn = FullAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                rotary_base=config.rope_base,
                dropout=config.dropout,
                layer_idx=self.layer_idx,
                window=-1 if not config.use_mamba else window
            )
        else:
            self.attn = Mamba2(
                d_model=config.hidden_size
            )

        self.ffn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.eps)
        self.ffn = GatedMlp(
            in_features=config.hidden_size,
            hidden_features=config.intermediate_size,
            activation=F.silu,
            bias1=False,
            bias2=False,
            multiple_of=1
        )

        self._init_weights()
    
    def _init_weights(self):
        for k, v in self.ffn.named_modules():
            if isinstance(v, nn.Linear): linear_init(v, zero_bias=True)
    
    def forward(self, x: torch.Tensor, causal: bool=True, return_attn_probs: bool=False):
        if isinstance(self.attn, FullAttention):
            out, attn_score, past_key_values = self.attn(self.attn_norm(x), causal, return_attn_probs)
        else:
            out = self.attn(self.attn_norm(x))
            attn_score, past_key_values = None, None
        x = x + out
        x = x + self.ffn(self.ffn_norm(x))

        return (x, attn_score, past_key_values)

class TransformerPretraindModel(PreTrainedModel):
    config_class = TransformerConfig
    supports_gradient_checkpointing = True
    _supports_cache_class = True
    _no_split_modules = ["TransformerBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            linear_init(module, zero_bias=True)
        elif isinstance(module, nn.Embedding):
            embedding_init(module, distribution="uniform")

class TransformerModel(TransformerPretraindModel):
    def __init__(self, config: TransformerConfig, vq_config: VQConfig, vq_codebook: torch.Tensor, **kwargs):
        super().__init__(config, **kwargs)

        if not config.use_mamba:
            self.layers = nn.ModuleList([TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        else:
            self.layers = nn.ModuleList([
                TransformerBlock(config, layer_idx, 1024, False if layer_idx % 2 == 0 else True) for layer_idx in range(config.num_hidden_layers)
            ])

        self.norm = RMSNorm(config.hidden_size, eps=config.eps)
        # self.embedding = nn.Embedding(vq_config.codebook_size, config.hidden_size // vq_config.num_heads)
        # self.split_project = nn.Linear(vq_config.latent_size // vq_config.num_heads, vq_config.latent_size)
        self.latent_project = nn.Linear(vq_config.latent_size, config.hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        causal: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else getattr(self.config, "output_hidden_states", False)
        output_attentions = output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", False)

        if inputs_embeds is None: raise ValueError("Token embeddings from VQ is required")
        # hidden_states = self.latent_project(inputs_embeds)
        hidden_states = inputs_embeds

        # if kwargs.get("use_gradient_checkpoint", False) is True and self.supports_gradient_checkpointing and self.training: self.gradient_checkpointing_enable()
        # else: self.gradient_checkpointing = False
        if kwargs.get("use_gradient_checkpoint", False) is True and self.supports_gradient_checkpointing: self.gradient_checkpointing = True
        else: self.gradient_checkpointing = False

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states: all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = checkpoint.checkpoint(
                    layer.__call__,
                    hidden_states,
                    causal,
                    output_attentions,
                    use_reentrant=False
                )
            else:
                hidden_states, attentions, past_key_values = layer(hidden_states, causal, output_attentions)
            
            if output_attentions: all_attentions += (attentions,)
        
        hidden_states = self.norm(hidden_states)
        if output_hidden_states: all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, past_key_values, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )

class TransformerForMaskedLM(TransformerPretraindModel):
    _tied_weights_keys = []

    def __init__(self, config: TransformerConfig, vq_config: VQConfig, vq_codebook: torch.Tensor):
        super().__init__(config)

        self.model = TransformerModel(config, vq_config, vq_codebook)
        self.lm_head = nn.Linear(config.hidden_size // vq_config.num_heads, vq_config.codebook_size + vq_config.vocab_size, bias=False)
        self.criterion = None

        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def _init_weights(self, module):
        for k, v in self.named_modules():
            if isinstance(v, nn.Linear):
                linear_init(v)
            elif isinstance(v, nn.Embedding):
                embedding_init(v)
    
    def forward(
        self,
        input_ids: torch.LongTensor=None,
        attention_mask: Optional[torch.Tensor]=None,
        inputs_embeds: Optional[torch.Tensor]=None,
        labels: Optional[torch.LongTensor|torch.FloatTensor]=None,
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool]=None,
        **kwargs: Unpack[Dict]
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            causal=False,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state.flatten(0, 1)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            self.criterion = CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction="mean")
            loss = self.criterion(logits, labels.flatten())
        
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

class TransformerForCausalLM(TransformerPretraindModel):
    _tied_weights_keys = []

    def __init__(self, config: TransformerConfig, vq_config: VQConfig, vq_codebook: torch.Tensor):
        super().__init__(config)

        self.model = TransformerModel(config, vq_config, vq_codebook)
        self.lm_head = nn.Linear(config.hidden_size // vq_config.num_heads, vq_config.codebook_size + vq_config.vocab_size, bias=False)
        self.criterion = None

        self.config = config
        self.vq_config = vq_config

        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor=None,
        attention_mask: Optional[torch.Tensor]=None,
        inputs_embeds: Optional[torch.Tensor]=None,
        past_key_values: Optional[Union[List[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor|torch.FloatTensor]=None,
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool]=None,
        **kwargs: Unpack[Dict]
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            causal=True,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state[:, :-1].contiguous()
        hidden_states = rearrange(hidden_states, "B L (nH dH) -> (B L nH) dH", nH=self.vq_config.num_heads, dH=self.config.hidden_size // self.vq_config.num_heads)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            self.criterion = CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction="mean")
            loss = self.criterion(logits, labels[:, 1:].contiguous().flatten())
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

class TransformerForSequenceClassification(TransformerPretraindModel):
    _tied_weights_keys = []

    def __init__(self, config: TransformerConfig, vq_config: VQConfig, vq_codebook: torch.Tensor):
        super().__init__(config)

        self.model = TransformerModel(config, vq_config, vq_codebook)
        self.score = nn.Linear(config.hidden_size, config.num_class)
        self.criterion = None

        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor=None,
        prefix_input_ids: Optional[torch.LongTensor] = None,
        suffix_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor]=None,
        inputs_embeds: Optional[torch.Tensor]=None,
        labels: Optional[torch.LongTensor|torch.FloatTensor]=None,
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool]=None,
        **kwargs: Unpack[Dict]
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            prefix_input_ids=prefix_input_ids,
            suffix_input_ids=suffix_input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            causal=False,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state
        # pad_mask = (input_ids == self.config.pad_token_id).unsqueeze(-1)
        # hidden_states = hidden_states.masked_fill(pad_mask, 0)
        # hidden_states = hidden_states.sum(dim=1) / (input_ids != self.config.pad_token_id).sum(1, keepdim=True)
        hidden_states = hidden_states.mean(dim=1)

        logits = self.score(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type == "single_label_classification":
                self.criterion = CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction="mean")
                loss = self.criterion(logits, labels.flatten())
            elif self.config.problem_type == "regression":
                self.criterion = torch.nn.MSELoss(reduction="mean")
                loss = self.criterion(logits, labels.reshape(-1, self.config.num_class))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=outputs.attentions
        )