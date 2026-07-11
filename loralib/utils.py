import torch.nn as nn
from .layers import PlainMultiheadAttentionLoRA


INDEX_POSITIONS_VISION = {
    "ViT-L/14": {
        "half-up": list(range(12, 24)),
        "half-bottom": list(range(12)),
        "all": list(range(24)),
    }
}


def mark_only_lora_as_trainable(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name


def apply_lora(args, clip_model):
    if args.encoder != "vision":
        raise ValueError("The cleaned Hybdir implementation supports vision LoRA only")
    indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
    layers = []
    for index, block in enumerate(clip_model.visual.transformer.resblocks):
        if index not in indices:
            continue
        for name, submodule in block.named_children():
            if isinstance(submodule, nn.MultiheadAttention):
                replacement = PlainMultiheadAttentionLoRA(submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                setattr(block, name, replacement)
                layers.append(replacement)
    return layers
