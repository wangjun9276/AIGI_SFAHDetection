from pathlib import Path
import torch


def _unwrap_state(payload):
    if isinstance(payload, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def load_torch(path, map_location="cpu"):
    return torch.load(str(path), map_location=map_location)


def normalize_state_dict(state):
    state = _unwrap_state(state)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        cleaned[key] = value
    return cleaned


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_stage1_weights(model, checkpoint_path, strict=True):
    state = normalize_state_dict(load_torch(checkpoint_path))
    # Legacy model_engine keys already use backbone.*. Raw Xception checkpoints do not.
    if not any(k.startswith("backbone.") for k in state):
        state = {f"backbone.{k}": v for k, v in state.items()}
    # Compatibility with the original xception.py names.
    mapped = {}
    for key, value in state.items():
        key = key.replace(".skipbn.", ".skip_bn.").replace(".conv1.", ".depthwise.")
        # Do not rename the top-level backbone.conv1 convolution.
        key = key.replace("backbone.depthwise.", "backbone.conv1.")
        if ".adjust_channel." not in key:
            mapped[key] = value
    state = mapped
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Stage-1 checkpoint mismatch. Missing={missing}; unexpected={unexpected}")
    return missing, unexpected


def load_stage2_trainable_weights(model, checkpoint_path):
    payload = load_torch(checkpoint_path)
    state = payload.get("trainable_state", payload.get("model_state", payload)) if isinstance(payload, dict) else payload
    state = normalize_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    not_loaded = sorted(name for name in trainable_names if name not in state)
    if unexpected or not_loaded:
        raise RuntimeError(f"Stage-2 checkpoint mismatch. Missing trainable keys={not_loaded}; unexpected={unexpected}")
    return payload, missing


def trainable_state_dict(model):
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    full_state = model.state_dict()
    return {name: full_state[name].detach().cpu() for name in sorted(trainable)}


def is_legacy_stage2_checkpoint(payload):
    try:
        state = normalize_state_dict(payload)
    except (TypeError, AttributeError):
        return False
    return any(key.startswith("xception_sobel_pass_npr.") or key.startswith("att.") or key.startswith("transformer.layers.") for key in state)


def _map_legacy_xception_key(key):
    key = key.replace("xception_sobel_pass_npr.", "xception_branch.")
    key = key.replace(".skipbn.", ".skip_bn.").replace(".conv1.", ".depthwise.")
    return key.replace("xception_branch.backbone.depthwise.", "xception_branch.backbone.conv1.")


def _map_legacy_stage2_key(key):
    key = _map_legacy_xception_key(key)
    direct = {
        "project.": "clip_projection.",
        "map.": "xception_projection.",
        "head_aux.": "aux_head.",
        "head.": "classifier.",
        "att.fc.": "frequency_attention.fc.",
        "att.dct_layer.": "frequency_attention.dct.",
    }
    for old, new in direct.items():
        if key.startswith(old):
            return new + key[len(old):]
    if key.startswith("transformer.layers."):
        parts = key.split(".")
        block = parts[2]
        suffix = ".".join(parts[3:])
        replacements = {
            "0.norm.": "norm1.",
            "0.fn.to_qkv.": "attn.qkv.",
            "0.fn.to_out.0.": "attn.out.0.",
            "0.fn.mask.sigma": "attn.mask.sigma",
            "0.fn.mask.alpha": "attn.mask.alpha",
            "1.norm.": "norm2.",
            "1.fn.net.0.": "mlp.0.",
            "1.fn.net.3.": "mlp.3.",
        }
        for old, new in replacements.items():
            if suffix.startswith(old):
                return f"patch_transformer.blocks.{block}." + new + suffix[len(old):]
        return None
    if key == "mask" or ".fn.mask.mask" in key or key.startswith("clip_model.transformer.") or key.startswith("clip_model.token_embedding.") or key.startswith("clip_model.positional_embedding") or key.startswith("clip_model.ln_final.") or key.startswith("clip_model.text_projection") or key.startswith("clip_model.logit_scale"):
        return None
    if ".adjust_channel." in key:
        return None
    return key


def load_legacy_stage2_full_weights(model, checkpoint_or_payload):
    payload = load_torch(checkpoint_or_payload) if isinstance(checkpoint_or_payload, (str, Path)) else checkpoint_or_payload
    state = normalize_state_dict(payload)
    mapped = {}
    for key, value in state.items():
        new_key = _map_legacy_stage2_key(key)
        if new_key is not None:
            mapped[new_key] = value
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    not_loaded = sorted(name for name in trainable_names if name not in mapped)
    unexpected = [key for key in unexpected if not key.startswith("clip_model.")]
    if not_loaded or unexpected:
        raise RuntimeError(f"Legacy Stage-2 checkpoint mismatch. Missing trainable keys={not_loaded}; unexpected={unexpected}")
    return payload, missing
