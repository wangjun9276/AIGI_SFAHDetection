from pathlib import Path
import math
import torch
from clip.model import VisionCLIP


def _read_state_dict(path):
    try:
        return torch.jit.load(str(path), map_location="cpu").eval().state_dict()
    except RuntimeError:
        payload = torch.load(str(path), map_location="cpu")
        return payload.get("state_dict", payload) if isinstance(payload, dict) else payload


def build_clip_model_from_state(state):
    state = {key.removeprefix("module."): value for key, value in state.items()}
    # Accept either official visual.* keys or keys extracted from a Hybrid checkpoint.
    if any(key.startswith("clip_model.") for key in state):
        state = {key.removeprefix("clip_model."): value for key, value in state.items() if key.startswith("clip_model.")}
    required = ["visual.conv1.weight", "visual.positional_embedding", "visual.proj"]
    missing_required = [key for key in required if key not in state]
    if missing_required:
        raise RuntimeError(f"Not a compatible CLIP ViT checkpoint; missing {missing_required}")
    width = state["visual.conv1.weight"].shape[0]
    patch_size = state["visual.conv1.weight"].shape[-1]
    grid = round(math.sqrt(state["visual.positional_embedding"].shape[0] - 1))
    layers = len({key.split(".")[3] for key in state if key.startswith("visual.transformer.resblocks.")})
    output_dim = state["visual.proj"].shape[1]
    model = VisionCLIP(grid * patch_size, patch_size, width, layers, output_dim)
    visual_state = {key.removeprefix("visual."): value for key, value in state.items() if key.startswith("visual.") and ".w_lora_" not in key}
    # Legacy Hybrid checkpoints already replaced selected attention modules by LoRA
    # modules, so their visual state has q_proj/k_proj/v_proj instead of in_proj_*.
    # In that case, rebuild the base MHA state before applying LoRA again.
    for index in range(layers):
        prefix = f"transformer.resblocks.{index}.attn."
        q_key, k_key, v_key = prefix + "q_proj.weight", prefix + "k_proj.weight", prefix + "v_proj.weight"
        if q_key in visual_state:
            visual_state[prefix + "in_proj_weight"] = torch.cat([visual_state.pop(q_key), visual_state.pop(k_key), visual_state.pop(v_key)], dim=0)
            qb, kb, vb = prefix + "q_proj.bias", prefix + "k_proj.bias", prefix + "v_proj.bias"
            if qb in visual_state:
                visual_state[prefix + "in_proj_bias"] = torch.cat([visual_state.pop(qb), visual_state.pop(kb), visual_state.pop(vb)], dim=0)
            visual_state[prefix + "out_proj.weight"] = visual_state.pop(prefix + "proj.weight")
            proj_bias = prefix + "proj.bias"
            if proj_bias in visual_state:
                visual_state[prefix + "out_proj.bias"] = visual_state.pop(proj_bias)
    missing, unexpected = model.visual.load_state_dict(visual_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"CLIP visual checkpoint mismatch. Missing={missing}; unexpected={unexpected}")
    return model.float().eval()


def load_clip_model(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CLIP checkpoint not found: {path}")
    return build_clip_model_from_state(_read_state_dict(path))
