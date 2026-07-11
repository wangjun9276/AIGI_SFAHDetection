from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import load_stage1_weights
from .frequency_attention import MultiSpectralAttentionLayer
from .gmm_transformer import GMMPatchTransformer
from .xception import Xception


class SobelMagnitude(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        ky = kx.t()
        self.register_buffer("kx", kx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("ky", ky.view(1, 1, 3, 3), persistent=False)

    def forward(self, x):
        c = x.shape[1]
        gx = F.conv2d(x, self.kx.expand(c, 1, 3, 3), padding=1, groups=c)
        gy = F.conv2d(x, self.ky.expand(c, 1, 3, 3), padding=1, groups=c)
        return torch.sqrt(gx.square() + gy.square() + 1e-12)


def neighboring_pixel_residual(x):
    h, w = x.shape[-2:]
    x = x[..., :h - h % 2, :w - w % 2]
    down = F.interpolate(x, scale_factor=0.5, mode="nearest", recompute_scale_factor=True)
    up = F.interpolate(down, size=x.shape[-2:], mode="nearest")
    return (x - up) * (2.0 / 3.0)


def _load_xception_imagenet(backbone, path):
    payload = torch.load(path, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        # Original pretrained file stores pointwise weights as [out, in].
        if "pointwise" in key and value.ndim == 2:
            value = value.unsqueeze(-1).unsqueeze(-1)
        if key.startswith("fc.") or key.startswith("last_linear."):
            continue
        # Compatibility with the original xception.py naming.
        key = key.replace(".conv1.", ".depthwise.").replace(".skipbn.", ".skip_bn.")
        cleaned[key] = value
    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    allowed_missing = {"last_linear.weight", "last_linear.bias"}
    real_missing = [k for k in missing if k not in allowed_missing]
    if real_missing or unexpected:
        raise RuntimeError(f"Xception ImageNet checkpoint mismatch. Missing={real_missing}; unexpected={unexpected}")


class XceptionArtifactDetector(nn.Module):
    """Stage 1: Sobel-enhanced neighboring-pixel residual followed by Xception."""
    def __init__(self, xception_pretrained=None):
        super().__init__()
        self.sobel = SobelMagnitude()
        self.backbone = Xception(num_classes=1)
        if xception_pretrained:
            if not Path(xception_pretrained).is_file():
                raise FileNotFoundError(f"Xception pretrained weights not found: {xception_pretrained}")
            _load_xception_imagenet(self.backbone, xception_pretrained)

    def forward_features(self, x):
        artifact = neighboring_pixel_residual(x + self.sobel(x))
        return self.backbone.features(artifact)

    def forward(self, x, return_features=False):
        features = self.forward_features(x.float())
        logits = self.backbone.classifier(features)
        return (logits, features) if return_features else logits


class HybridDetector(nn.Module):
    """Stage 2: frozen Stage-1 Xception + CLIP ViT-L/14 LoRA hybrid detector."""
    def __init__(self, stage1_checkpoint=None, clip_path=None, lora_rank=4, lora_alpha=0.5, lora_dropout=0.0, clip_model=None, apply_clip_lora=True):
        super().__init__()
        self.xception_branch = XceptionArtifactDetector()
        if stage1_checkpoint:
            load_stage1_weights(self.xception_branch, stage1_checkpoint, strict=True)
        for parameter in self.xception_branch.parameters():
            parameter.requires_grad = False
        self.xception_branch.eval()

        if clip_model is None:
            if not clip_path:
                raise ValueError("clip_path is required when clip_model is not supplied")
            from .clip_loader import load_clip_model
            self.clip_model = load_clip_model(clip_path)
        else:
            self.clip_model = clip_model
        self.clip_model = self.clip_model.float()

        if apply_clip_lora:
            from loralib.utils import apply_lora, mark_only_lora_as_trainable
            lora_args = SimpleNamespace(encoder="vision", backbone="ViT-L/14", position="half-up", params="qkv", r=lora_rank, alpha=lora_alpha, dropout_rate=lora_dropout)
            self.lora_layers = apply_lora(lora_args, self.clip_model)
            mark_only_lora_as_trainable(self.clip_model)
        else:
            self.lora_layers = []
            for parameter in self.clip_model.parameters():
                parameter.requires_grad = False

        self.frequency_attention = MultiSpectralAttentionLayer(2048, 7, 7, reduction=4, freq_sel_method="bot16")
        self.patch_transformer = GMMPatchTransformer(dim=1024, num_patches=256, depth=2, heads=4, head_dim=64, mlp_ratio=2, kernels=5, dropout=0.25)
        self.xception_projection = nn.Linear(2048, 512)
        self.clip_projection = nn.Linear(768, 512)
        self.aux_head = nn.Linear(512, 1)
        self.classifier = nn.Linear(512, 1)
        self.freq_scale = nn.Parameter(torch.zeros(1))
        self.gmm_scale = nn.Parameter(torch.zeros(1))
        self._initialize_new_layers()

    def _initialize_new_layers(self):
        # Match the original implementation: explicitly initialize CLIP projection
        # and classification heads; leave the Xception map and attention MLP at
        # their native PyTorch initialization.
        for layer in (self.clip_projection, self.aux_head, self.classifier):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def train(self, mode=True):
        super().train(mode)
        # Frozen BatchNorm statistics must never be updated in Stage 2.
        self.xception_branch.eval()
        return self

    def forward(self, x, return_aux=None):
        x = x.float()
        with torch.no_grad():
            xception_map = self.xception_branch.forward_features(x)
        base_vector = F.adaptive_avg_pool2d(F.relu(xception_map, inplace=False), 1).flatten(1)
        attended_map = self.frequency_attention(xception_map)
        attended_vector = F.adaptive_avg_pool2d(F.relu(attended_map, inplace=False), 1).flatten(1)
        base_projected = self.xception_projection(base_vector)
        attended_projected = self.xception_projection(attended_vector)
        xception_vector = self.xception_projection(base_vector + self.freq_scale * attended_vector)

        clip_output = self.clip_model.encode_image(x)
        if not isinstance(clip_output, (tuple, list)) or len(clip_output) != 2:
            raise RuntimeError("The bundled modified CLIP is required: encode_image must return (global_feature, patch_tokens).")
        clip_global, patch_tokens = clip_output
        if clip_global.shape[-1] != 768 or patch_tokens.shape[-1] != 1024:
            raise RuntimeError(f"Expected CLIP ViT-L/14 dimensions 768/1024, got {clip_global.shape[-1]}/{patch_tokens.shape[-1]}")
        patch_global = self.patch_transformer(patch_tokens).mean(dim=1) @ self.clip_model.visual.proj
        clip_global_projected = self.clip_projection(clip_global)
        patch_projected = self.clip_projection(patch_global)
        clip_vector = self.clip_projection(clip_global - self.gmm_scale * patch_global)
        logits = self.classifier(xception_vector + clip_vector)

        return_aux = self.training if return_aux is None else return_aux
        if not return_aux:
            return logits
        return {"logits": logits, "aux_logits": [self.aux_head(base_projected), self.aux_head(attended_projected), self.aux_head(clip_global_projected), self.aux_head(patch_projected)]}
