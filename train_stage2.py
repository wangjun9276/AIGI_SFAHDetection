import argparse
from hybdir.data import create_loader
from hybdir.engine import build_optimizer
from hybdir.models import HybridDetector
from hybdir.trainer import fit
from hybdir.utils import resolve_device, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: train frozen Xception + CLIP-LoRA hybrid detector")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--test-root", default=None)
    parser.add_argument("--stage1-checkpoint", required=True, help="best.pt produced by train_stage1.py")
    parser.add_argument("--clip-path", required=True, help="OpenAI CLIP ViT-L/14 .pt checkpoint")
    parser.add_argument("--output-dir", default="outputs/stage2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--monitor", choices=["ap", "auc", "balanced_accuracy", "accuracy"], default="ap")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=0.5)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.output_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)
    model = HybridDetector(args.stage1_checkpoint, args.clip_path, args.lora_rank, args.lora_alpha, args.lora_dropout).to(device)
    train_loader = create_loader(args.train_root, args.batch_size, train=True, num_workers=args.num_workers)
    val_loader = create_loader(args.val_root, args.batch_size, train=False, num_workers=args.num_workers)
    test_loader = create_loader(args.test_root, args.batch_size, train=False, num_workers=args.num_workers) if args.test_root else None
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    metadata = {"architecture": "HybridDetector", "stage1_checkpoint": args.stage1_checkpoint, "clip_path": args.clip_path, "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout, "args": vars(args)}
    fit(model, train_loader, val_loader, optimizer, device, args.output_dir, args.epochs, args.patience, args.accumulation_steps, args.auxiliary_weight, args.grad_clip, not args.no_amp, args.monitor, 2, metadata, test_loader)


if __name__ == "__main__":
    main()
