import argparse
import csv
import logging
from pathlib import Path
from hybdir.checkpoint import is_legacy_stage2_checkpoint, load_legacy_stage2_full_weights, load_stage1_weights, load_stage2_trainable_weights, load_torch, normalize_state_dict
from hybdir.data import create_loader
from hybdir.engine import evaluate, write_predictions
from hybdir.models import HybridDetector, XceptionArtifactDetector
from hybdir.metrics import compute_metrics
from hybdir.clip_loader import build_clip_model_from_state
from hybdir.utils import resolve_device, save_json, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Complete evaluation for Stage-1 or Stage-2 detector")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--subsets", default="", help="Comma-separated subdirectories under test-root; empty evaluates test-root directly")
    parser.add_argument("--stage1-checkpoint", default=None, help="Stage-1 checkpoint; falls back to Stage-2 checkpoint metadata")
    parser.add_argument("--clip-path", default=None, help="CLIP ViT-L/14 weights; falls back to Stage-2 checkpoint metadata")
    parser.add_argument("--output-dir", default="results/test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1029)
    return parser.parse_args()


def build_model(args, device):
    if args.stage == 1:
        model = XceptionArtifactDetector()
        load_stage1_weights(model, args.checkpoint, strict=True)
    else:
        payload = load_torch(args.checkpoint)
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        lora_rank = args.lora_rank if args.lora_rank is not None else metadata.get("lora_rank", 4)
        lora_alpha = args.lora_alpha if args.lora_alpha is not None else metadata.get("lora_alpha", 0.5)
        lora_dropout = args.lora_dropout if args.lora_dropout is not None else metadata.get("lora_dropout", 0.0)
        if is_legacy_stage2_checkpoint(payload):
            clip_model = build_clip_model_from_state(normalize_state_dict(payload))
            model = HybridDetector(clip_model=clip_model, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            load_legacy_stage2_full_weights(model, payload)
        else:
            stage1_checkpoint = args.stage1_checkpoint or metadata.get("stage1_checkpoint")
            clip_path = args.clip_path or metadata.get("clip_path")
            if not stage1_checkpoint or not clip_path:
                raise ValueError("Stage 2 testing requires Stage-1 and CLIP weights, either through CLI arguments or checkpoint metadata")
            model = HybridDetector(stage1_checkpoint, clip_path, lora_rank, lora_alpha, lora_dropout)
            load_stage2_trainable_weights(model, args.checkpoint)
    return model.to(device).eval()


def main():
    args = parse_args()
    setup_logging(args.output_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)
    model = build_model(args, device)
    root = Path(args.test_root)
    subsets = [name.strip() for name in args.subsets.split(",") if name.strip()]
    targets = [(name, root / name) for name in subsets] if subsets else [(root.name or "test", root)]
    summaries, all_predictions = [], []
    for name, subset_root in targets:
        loader = create_loader(subset_root, args.batch_size, train=False, num_workers=args.num_workers)
        metrics, predictions = evaluate(model, loader, device, args.threshold, return_predictions=True)
        metrics["dataset"] = name
        metrics["num_discovered"] = len(loader.dataset)
        metrics["num_skipped"] = len(loader.dataset) - metrics["num_samples"]
        summaries.append(metrics)
        all_predictions.extend(predictions)
        write_predictions(Path(args.output_dir) / f"predictions_{name}.csv", predictions)
        logging.info("%-20s | N %d | ACC %.4f | BACC %.4f | AP %.4f | AUC %.4f | real %.4f | fake %.4f", name, metrics["num_samples"], metrics["accuracy"], metrics["balanced_accuracy"], metrics["ap"], metrics["auc"], metrics["real_accuracy"], metrics["fake_accuracy"])
    overall = compute_metrics([row["label"] for row in all_predictions], [row["probability"] for row in all_predictions], args.threshold)
    overall.update({"dataset": "OVERALL", "num_discovered": sum(row["num_discovered"] for row in summaries), "num_skipped": sum(row["num_skipped"] for row in summaries)})
    metric_names = ["accuracy", "balanced_accuracy", "ap", "auc", "eer", "f1", "precision", "recall", "real_accuracy", "fake_accuracy"]
    macro = {}
    for key in metric_names:
        values = [row[key] for row in summaries if row[key] == row[key]]
        macro[key] = sum(values) / len(values) if values else float("nan")
    macro.update({"num_samples": sum(row["num_samples"] for row in summaries), "num_real": sum(row["num_real"] for row in summaries), "num_fake": sum(row["num_fake"] for row in summaries), "threshold": args.threshold, "dataset": "MACRO_MEAN", "num_discovered": sum(row["num_discovered"] for row in summaries), "num_skipped": sum(row["num_skipped"] for row in summaries)})
    report_rows = summaries + [macro, overall]
    save_json(Path(args.output_dir) / "metrics.json", {"subsets": summaries, "macro_mean": macro, "overall": overall})
    write_predictions(Path(args.output_dir) / "predictions_all.csv", all_predictions)
    keys = list(summaries[0].keys())
    with open(Path(args.output_dir) / "metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(report_rows)


if __name__ == "__main__":
    main()
