import csv
from pathlib import Path
import torch
import torch.nn as nn
from .metrics import compute_metrics


def extract_logits(output):
    if isinstance(output, dict):
        return output["logits"]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def compute_loss(output, labels, auxiliary_weight=0.0):
    criterion = nn.BCEWithLogitsLoss()
    logits = extract_logits(output).flatten()
    loss = criterion(logits, labels)
    if isinstance(output, dict) and output.get("aux_logits") and auxiliary_weight > 0:
        aux = sum(criterion(item.flatten(), labels) for item in output["aux_logits"])
        loss = loss + auxiliary_weight * aux
    return loss


def train_one_epoch(model, loader, optimizer, device, scaler=None, accumulation_steps=1, auxiliary_weight=0.0, grad_clip=0.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_samples, valid_steps = 0.0, 0, 0
    use_amp = scaler is not None and scaler.is_enabled()
    for step, batch in enumerate(loader):
        if batch is None:
            continue
        images, labels, _ = batch
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = model(images)
            raw_loss = compute_loss(output, labels, auxiliary_weight)
            loss = raw_loss / accumulation_steps
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        valid_steps += 1
        should_step = valid_steps % accumulation_steps == 0
        if should_step:
            if use_amp:
                scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += raw_loss.item() * labels.numel()
        total_samples += labels.numel()
    if valid_steps and valid_steps % accumulation_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, return_predictions=False):
    model.eval()
    labels_all, probabilities_all, paths_all = [], [], []
    for batch in loader:
        if batch is None:
            continue
        images, labels, paths = batch
        logits = extract_logits(model(images.to(device, non_blocking=True), return_aux=False) if hasattr(model, "xception_branch") else model(images.to(device, non_blocking=True)))
        probabilities = logits.flatten().sigmoid().cpu().numpy()
        labels_all.extend(labels.numpy().astype(int).tolist())
        probabilities_all.extend(probabilities.tolist())
        paths_all.extend(paths)
    if not labels_all:
        raise RuntimeError("No valid images were decoded during evaluation")
    metrics = compute_metrics(labels_all, probabilities_all, threshold)
    if not return_predictions:
        return metrics
    rows = [{"path": path, "label": label, "probability": probability, "prediction": int(probability >= threshold)} for path, label, probability in zip(paths_all, labels_all, probabilities_all)]
    return metrics, rows


def write_predictions(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "probability", "prediction"])
        writer.writeheader()
        writer.writerows(rows)


def build_optimizer(model, name, lr, weight_decay=0.0):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters found")
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")
