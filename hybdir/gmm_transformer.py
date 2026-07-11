"""Two-layer patch transformer with the Gaussian-mixture attention mask."""

import math
import torch
import torch.nn as nn


def build_spatial_mask(num_patches):
    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError("num_patches must form a square grid")
    coords = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"), dim=-1).reshape(-1, 2).float()
    distance2 = torch.cdist(coords, coords).pow(2)
    return torch.exp(-0.5 * distance2).view(1, 1, num_patches, num_patches)


class GaussianMixtureMask(nn.Module):
    def __init__(self, heads, kernels, base_mask):
        super().__init__()
        self.sigma = nn.Parameter(torch.randn(kernels, heads, 1, 1) * 10 + 10)
        self.alpha = nn.Parameter(torch.randn(kernels, heads, 1, 1) * 2)
        self.register_buffer("base_mask", base_mask, persistent=True)

    def forward(self, scores):
        exponent = 1.0 / (self.sigma.square() + 1e-5)
        modulation = (self.alpha * self.base_mask.pow(exponent)).sum(dim=0)
        return scores * modulation.unsqueeze(0) if modulation.ndim == 3 else scores * modulation


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x):
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class GMMMultiheadAttention(nn.Module):
    def __init__(self, dim=1024, heads=4, head_dim=64, kernels=5, num_patches=256, dropout=0.25):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, inner * 3, bias=False)
        nn.init.xavier_normal_(self.qkv.weight)
        self.out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
        self.mask = GaussianMixtureMask(heads, kernels, build_spatial_mask(num_patches))

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        reshape = lambda t: t.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        scores = self.mask((q @ k.transpose(-2, -1)) * self.scale)
        out = scores.softmax(dim=-1) @ v
        return self.out(out.transpose(1, 2).reshape(b, n, self.heads * self.head_dim))


class TransformerBlock(nn.Module):
    def __init__(self, dim=1024, heads=4, head_dim=64, mlp_ratio=2, kernels=5, num_patches=256, dropout=0.25, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GMMMultiheadAttention(dim, heads, head_dim, kernels, num_patches, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class GMMPatchTransformer(nn.Module):
    def __init__(self, dim=1024, num_patches=256, depth=2, heads=4, head_dim=64, mlp_ratio=2, kernels=5, dropout=0.25, drop_path=0.1):
        super().__init__()
        self.num_patches = num_patches
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, head_dim, mlp_ratio, kernels, num_patches, dropout, drop_path) for _ in range(depth)])

    def forward(self, x):
        if x.shape[1] != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} CLIP patch tokens, got {x.shape[1]}. Use CLIP ViT-L/14 with 224x224 inputs.")
        for block in self.blocks:
            x = block(x)
        return x
