import argparse
from pathlib import Path
from hybdir.data import create_loader
from hybdir.engine import build_optimizer
from hybdir.models import XceptionArtifactDetector
from hybdir.trainer import fit
from hybdir.utils import resolve_device, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: train Sobel+NPR Xception detector")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--test-root", default=None)
    parser.add_argument("--xception-pretrained", required=True, help="ImageNet Xception checkpoint, e.g. xception-b5690688.pth")
    parser.add_argument("--output-dir", default="outputs/stage1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="adamw")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--monitor", choices=["ap", "auc", "balanced_accuracy", "accuracy"], default="ap")
    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.output_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)
    model = XceptionArtifactDetector(args.xception_pretrained).to(device)
    train_loader = create_loader(args.train_root, args.batch_size, train=True, num_workers=args.num_workers)
    val_loader = create_loader(args.val_root, args.batch_size, train=False, num_workers=args.num_workers)
    test_loader = create_loader(args.test_root, args.batch_size, train=False, num_workers=args.num_workers) if args.test_root else None
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    metadata = {"architecture": "XceptionArtifactDetector", "args": vars(args)}
    fit(model, train_loader, val_loader, optimizer, device, args.output_dir, args.epochs, args.patience, args.accumulation_steps, 0.0, args.grad_clip, not args.no_amp, args.monitor, 1, metadata, test_loader)


if __name__ == "__main__":
    main()
