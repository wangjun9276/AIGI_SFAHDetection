import json
import logging
import os
import random
from pathlib import Path
import numpy as np
import torch


def setup_logging(output_dir=None):
    handlers = [logging.StreamHandler()]
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(output_dir) / "run.log", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers, force=True)


def set_seed(seed=1029, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def json_ready(value):
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")
