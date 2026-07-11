"""Minimal LoRA layers required by the CLIP visual encoder."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLoRA(nn.Linear):
    def __init__(self, existing_linear, r=4, lora_alpha=1.0, dropout_rate=0.0):
        super().__init__(existing_linear.in_features, existing_linear.out_features, bias=existing_linear.bias is not None)
        self.weight.data.copy_(existing_linear.weight.data)
        if self.bias is not None:
            self.bias.data.copy_(existing_linear.bias.data)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.r = int(r)
        self.scaling = float(lora_alpha) / math.sqrt(self.r) if self.r > 0 else 0.0
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        if self.r > 0:
            self.w_lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
            self.w_lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
            nn.init.kaiming_uniform_(self.w_lora_A, a=math.sqrt(5))

    def forward(self, x):
        output = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            output = output + F.linear(F.linear(self.dropout(x), self.w_lora_A), self.w_lora_B) * self.scaling
        return output


class PlainMultiheadAttentionLoRA(nn.Module):
    def __init__(self, existing_mha, enable_lora=("q", "k", "v"), r=4, lora_alpha=1.0, dropout_rate=0.0, **_):
        super().__init__()
        if not existing_mha._qkv_same_embed_dim:
            raise ValueError("Only self-attention with equal Q/K/V dimensions is supported")
        self.embed_dim = existing_mha.embed_dim
        self.num_heads = existing_mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.batch_first = existing_mha.batch_first
        self.dropout_p = float(existing_mha.dropout)
        in_weight = existing_mha.in_proj_weight.detach()
        in_bias = existing_mha.in_proj_bias.detach() if existing_mha.in_proj_bias is not None else None

        def make_projection(index, enabled):
            linear = nn.Linear(self.embed_dim, self.embed_dim, bias=in_bias is not None)
            linear.weight.data.copy_(in_weight[index * self.embed_dim:(index + 1) * self.embed_dim])
            if in_bias is not None:
                linear.bias.data.copy_(in_bias[index * self.embed_dim:(index + 1) * self.embed_dim])
            if enabled:
                return LinearLoRA(linear, r, lora_alpha, dropout_rate)
            for parameter in linear.parameters():
                parameter.requires_grad = False
            return linear

        self.q_proj = make_projection(0, "q" in enable_lora)
        self.k_proj = make_projection(1, "k" in enable_lora)
        self.v_proj = make_projection(2, "v" in enable_lora)
        out_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.out_proj.bias is not None)
        out_linear.weight.data.copy_(existing_mha.out_proj.weight.data)
        if out_linear.bias is not None:
            out_linear.bias.data.copy_(existing_mha.out_proj.bias.data)
        self.proj = LinearLoRA(out_linear, r, lora_alpha, dropout_rate) if "o" in enable_lora else out_linear
        if "o" not in enable_lora:
            for parameter in self.proj.parameters():
                parameter.requires_grad = False

    def _reshape(self, tensor, batch, length):
        return tensor.view(length, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, average_attn_weights=True, is_causal=False, **_):
        if key_padding_mask is not None:
            raise ValueError("key_padding_mask is not used by the CLIP visual encoder")
        if self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        target_len, batch, _ = query.shape
        source_len = key.shape[0]
        q = self._reshape(self.q_proj(query), batch, target_len)
        k = self._reshape(self.k_proj(key), batch, source_len)
        v = self._reshape(self.v_proj(value), batch, source_len)
        mask = attn_mask
        if mask is not None and mask.ndim == 2:
            mask = mask.view(1, 1, target_len, source_len)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout_p if self.training else 0.0, is_causal=is_causal)
        output = output.permute(2, 0, 1, 3).reshape(target_len, batch, self.embed_dim)
        output = self.proj(output)
        if self.batch_first:
            output = output.transpose(0, 1)
        weights = None
        if need_weights:
            # CLIP always requests need_weights=False; returning None avoids an unnecessary second attention pass.
            weights = None
        return output, weights
