import logging
from pathlib import Path
import torch
from .checkpoint import load_stage1_weights, load_stage2_trainable_weights, save_checkpoint, trainable_state_dict
from .engine import evaluate, train_one_epoch
from .utils import save_json


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _score(metrics, metric_name):
    value = metrics.get(metric_name)
    if value is None or value != value:
        value = metrics["balanced_accuracy"]
    return float(value)


def fit(model, train_loader, val_loader, optimizer, device, output_dir, epochs=30, patience=5, accumulation_steps=1, auxiliary_weight=0.0, grad_clip=0.0, amp=True, monitor="ap", stage=1, metadata=None, test_loader=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = make_scaler(amp and device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_score, bad_epochs, history = float("-inf"), 0, []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, accumulation_steps, auxiliary_weight, grad_clip)
        val_metrics = evaluate(model, val_loader, device)
        score = _score(val_metrics, monitor)
        scheduler.step(score)
        record = {"epoch": epoch, "train_loss": train_loss, "learning_rate": optimizer.param_groups[0]["lr"], **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        logging.info("Epoch %03d | loss %.6f | val ACC %.4f | BACC %.4f | AP %.4f | AUC %.4f", epoch, train_loss, val_metrics["accuracy"], val_metrics["balanced_accuracy"], val_metrics["ap"], val_metrics["auc"])

        state_key = "model_state" if stage == 1 else "trainable_state"
        state_value = {k: v.detach().cpu() for k, v in model.state_dict().items()} if stage == 1 else trainable_state_dict(model)
        payload = {"stage": stage, state_key: state_value, "epoch": epoch, "monitor": monitor, "score": score, "metrics": val_metrics, "metadata": metadata or {}}
        save_checkpoint(output_dir / "last.pt", payload)
        if score > best_score:
            best_score, bad_epochs = score, 0
            save_checkpoint(output_dir / "best.pt", payload)
            logging.info("Saved new best checkpoint: %s", output_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logging.info("Early stopping after %d non-improving epochs.", bad_epochs)
                break

    save_json(output_dir / "history.json", history)
    if stage == 1:
        load_stage1_weights(model, output_dir / "best.pt", strict=True)
    else:
        load_stage2_trainable_weights(model, output_dir / "best.pt")
    final = {"best_score": best_score, "best_checkpoint": str(output_dir / "best.pt"), "validation": evaluate(model, val_loader, device)}
    if test_loader is not None:
        final["test"] = evaluate(model, test_loader, device)
        logging.info("Final test | ACC %.4f | BACC %.4f | AP %.4f | AUC %.4f", final["test"]["accuracy"], final["test"]["balanced_accuracy"], final["test"]["ap"], final["test"]["auc"])
    save_json(output_dir / "summary.json", final)
    return final
