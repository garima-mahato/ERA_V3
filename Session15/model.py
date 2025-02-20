import os
import math
from typing import List, Optional, Tuple, Union
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import GPT2Tokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import ModelCheckpoint

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

@dataclass
class SmolLM2Config:
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1.0e-05
    vocab_size: int = 49152
    rope_theta: float = 10000.0
    use_cache: bool = True
    tie_word_embeddings: bool = True
    torch_dtype: str = "float32"
    block_size: int = 576 #512
    compression_ratio: int = 8
    num_experts: int = 4
    num_shared_experts: int = 1
    top_k: int = 2
    device: str = device

class SmolLM2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        SmolLM2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)


class SmolLM2RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos.unsqueeze(0)  # [bs, 1, seq_len, dim]
    sin = sin.unsqueeze(0)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def _precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (cos + i*sin)"""
    # Only compute frequencies for half the dimension
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)  # [seq_len, dim//2]

    # Compute cos and sin
    freqs_cos = torch.cos(freqs)  # [seq_len, dim//2]
    freqs_sin = torch.sin(freqs)  # [seq_len, dim//2]

    # Stack real and imaginary parts
    freqs_cis = torch.stack([freqs_cos, freqs_sin], dim=-1)  # [seq_len, dim//2, 2]

    return freqs_cis


class SmolLM2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int
    ):
        super().__init__()
        self.moe = DeepSeekMoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k
        )

    def forward(self, x):
        return self.moe(x)

class DeepSeekExpertLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class DeepSeekMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_shared_experts: int = 1,
        top_k: int = 2
    ):
        super().__init__()
        #print(num_experts,num_shared_experts)
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = self.num_experts - self.num_shared_experts
        self.num_shared_experts = self.num_shared_experts
        self.top_k = top_k
        self.hidden_size = hidden_size

        # Shared Experts
        self.shared_experts = nn.ModuleList([
            DeepSeekExpertLayer(hidden_size, intermediate_size)
            for _ in range(self.num_shared_experts)
        ])

        # Routed Experts
        self.routed_experts = nn.ModuleList([
            DeepSeekExpertLayer(hidden_size, intermediate_size)
            for _ in range(self.num_routed_experts)
        ])

        # Router Components
        self.router = nn.Linear(hidden_size, self.num_routed_experts, bias=False)
        self.routing_bias = nn.Parameter(torch.zeros(self.num_routed_experts))

    def forward(self, x):
        #print(x.shape)
        batch_size, seq_len, hidden_size = x.shape

        # Process through shared experts
        shared_output = sum(expert(x) for expert in self.shared_experts)
        if self.num_shared_experts > 1:
            shared_output = shared_output / self.num_shared_experts # Average
        
        # Calculate routing scores
        routing_logits = self.router(x) + self.routing_bias

        # Get top-k experts per token
        routing_probs = torch.sigmoid(routing_logits)
        scores, indices = torch.topk(routing_probs, self.top_k, dim=-1)

        # Normalize the top-k scores
        scores = scores / scores.sum(dim=-1, keepdim=True)

        # Process through routed experts
        combined_output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_indices = indices[..., k]
            expert_scores = scores[..., k:k+1]

            # Process each expert
            for i in range(self.num_routed_experts):
                mask = (expert_indices == i)
                if mask.any():
                  expert_input = x[mask]
                  expert_output = self.routed_experts[i](expert_input)
                  combined_output[mask] += expert_output * expert_scores[mask]
        
        # Combine shared and routed outputs
        final_output = shared_output + combined_output
        return final_output

    def update_bias_terms(self, expert_load):
        # Adjust bias terms based on expert load
        target_load = 1.0 / self.num_routed_experts
        load_diff = expert_load - target_load

        # Dynamic update rate based on the magnitude of the load imbalance
        update_rate = 0.1 * torch.abs(load_diff)

        # Update the routing bias using the dynamic update rate
        self.routing_bias.data -= update_rate * load_diff


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class SmolLM2MultiHeadLatentAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.latent_dim = self.hidden_size // config.compression_ratio
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # Compression
        self.q_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.kv_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)

        # Expansion
        self.q_proj_u = nn.Linear(self.latent_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj_u = nn.Linear(self.latent_dim, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj_u = nn.Linear(self.latent_dim, self.num_key_value_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = SmolLM2RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        is_sdpa: bool = True,
        is_causal = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        kv_d = self.kv_proj_d(hidden_states)
        q_d = self.q_proj_d(hidden_states)

        k_proj_2 = self.k_proj_u(kv_d)
        q_proj_2 = self.q_proj_u(q_d)
        v = self.v_proj_u(kv_d)

        query_states = q_proj_2.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = k_proj_2.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if is_sdpa:
            key = key_states
            value = value_states
            query = query_states
            if self.num_key_value_groups:
                key = repeat_kv(key, self.num_key_value_groups)
                value = repeat_kv(value, self.num_key_value_groups)

            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key.shape[-2]]

            # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            if is_causal is None:
                is_causal = causal_mask is None and query.shape[2] > 1

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=causal_mask,
                dropout_p=0.0,
                scale=self.head_dim**-0.5,
                is_causal=is_causal,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class SmolLM2DecoderLayer(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = SmolLM2MultiHeadLatentAttention(config=config)
        self.mlp = SmolLM2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            num_shared_experts=config.num_shared_experts,
            top_k=config.top_k
        )
        self.input_layernorm = SmolLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SmolLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class SmolLM2Model(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.dtype = getattr(torch, config.torch_dtype) if hasattr(torch, config.torch_dtype) else torch.float32

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SmolLM2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = SmolLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.freqs_cis = _precompute_freqs_cis(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

        self.apply(self._init_weights)

        self.to(self.dtype)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        freqs_cis = self.freqs_cis.to(device=hidden_states.device, dtype=hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, freqs_cis)[0]

        hidden_states = self.norm(hidden_states)
        return hidden_states

class SmolLM2ForCausalLM(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.model = SmolLM2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text given a starting sequence of tokens.
        Args:
            idx (torch.Tensor): Starting token indices, shape (B, T)
            max_new_tokens (int): Number of tokens to generate
            temperature (float): Sampling temperature (1.0 = no change, < 1.0 = less random, > 1.0 = more random)
            top_k (int): If specified, only sample from the top k most probable tokens
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class plSmolLM2(pl.LightningModule):
    def __init__(self, config: SmolLM2Config, lr, warmup_steps, max_steps, step=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = SmolLM2ForCausalLM(self.config)
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.generation_prompt = "Hello there! Today, we are going to talk about "
        self._generating = False
        self.start_step = step if step is not None else 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        #with torch.cuda.amp.autocast():
        input_ids = batch["input_ids"]
        target_ids = batch["labels"]
        logits, _ = self(input_ids)
        loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        ## Update bias
    
        # Calculate expert load
        expert_load = torch.zeros(self.model.model.layers[0].mlp.moe.num_routed_experts, device=self.config.device)
        for k in range(self.model.model.layers[0].mlp.moe.top_k):
            routing_logits = self.model.model.layers[0].mlp.moe.router(input_ids.to(torch.float32)) + self.model.model.layers[0].mlp.moe.routing_bias
            routing_probs = torch.sigmoid(routing_logits)
            _, indices = torch.topk(routing_probs, self.model.model.layers[0].mlp.moe.top_k, dim=-1)
            for i in range(self.model.model.layers[0].mlp.moe.num_routed_experts):
                expert_load[i] += (indices[..., k] == i).sum()
        expert_load = expert_load / (input_ids.size(0) * input_ids.size(1) * self.model.model.layers[0].mlp.moe.top_k)

        # Update bias terms
        self.model.model.layers[0].mlp.moe.update_bias_terms(expert_load)

        # Log the loss with 4 decimal precision
        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, logger=True
        )
        print(f"Step: {self.start_step+self.global_step}, Train Loss: {loss}")

        # Generate text every n steps, but only if we're not already generating
        if (self.global_step) % log_every_n_steps == 0 and not self._generating:
            self._generating = True
            self.generate_and_log_sample()
            self._generating = False
        #self.step = self.step + 1

        return loss

    def generate_and_log_sample(self):
        """Generate and log a sample of text from the model"""
        try:
            # Encode the prompt
            prompt_ids = self.tokenizer.encode(
                self.generation_prompt, return_tensors="pt"
            ).to(self.device)

            # Generate new tokens
            generated_ids = self.model.generate(
                prompt_ids, max_new_tokens=50, temperature=0.8, top_k=40
            )

            # Decode the generated tokens
            generated_text = self.tokenizer.decode(generated_ids[0].tolist())

            # Create a formatted message
            message = (
                f"\n{'='*40}\n"
                f"Step {self.global_step} generation:\n"
                f"Prompt: {self.generation_prompt}\n"
                f"Generated: {generated_text}\n"
                f"{'='*40}\n"
            )

            print(message)

            # Log to WandB
            if hasattr(self.logger, "experiment"):
                self.logger.experiment.log(
                    {"generated_text": generated_text, "global_step": self.global_step}
                )
        except Exception as e:
            print(f"Generation failed with error: {str(e)}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                return self.hparams.lr * (current_step + 1) / self.hparams.warmup_steps
            elif current_step > self.hparams.max_steps:
                return self.hparams.lr * 0.1
            decay_ratio = (current_step - self.hparams.warmup_steps) / (
                self.hparams.max_steps - self.hparams.warmup_steps
            )
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self.hparams.lr * 0.1 + coeff * (
                self.hparams.lr - self.hparams.lr * 0.1
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]