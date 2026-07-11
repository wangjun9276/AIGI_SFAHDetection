"""Vision-only OpenAI CLIP ViT implementation.

Only the visual encoder is needed by Hybdir. Parameter names intentionally match
OpenAI CLIP so ViT-L/14 weights can be loaded directly from the official .pt file.
"""

from collections import OrderedDict
import torch
from torch import nn


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        dtype = x.dtype
        return super().forward(x.float()).to(dtype)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(width * 4, width)),
        ]))
        self.ln_2 = LayerNorm(width)

    def attention(self, x):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, width, layers, heads):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype).view(1, 1, -1).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)
        if x.shape[1] != self.positional_embedding.shape[0]:
            raise ValueError(f"CLIP expected {self.input_resolution}x{self.input_resolution} inputs; token count is {x.shape[1]} instead of {self.positional_embedding.shape[0]}")
        x = self.ln_pre(x + self.positional_embedding.to(x.dtype))
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        patch_tokens = x[:, 1:, :]
        global_feature = self.ln_post(x[:, 0, :]) @ self.proj
        return global_feature, patch_tokens


class VisionCLIP(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers, output_dim):
        super().__init__()
        self.visual = VisionTransformer(input_resolution, patch_size, width, layers, width // 64, output_dim)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.to(self.dtype))

    def forward(self, image):
        return self.encode_image(image)
