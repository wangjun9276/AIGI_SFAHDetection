#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT=${TRAIN_ROOT:?Please set TRAIN_ROOT}
VAL_ROOT=${VAL_ROOT:?Please set VAL_ROOT}
XCEPTION_PRETRAINED=${XCEPTION_PRETRAINED:?Please set XCEPTION_PRETRAINED}
CLIP_PATH=${CLIP_PATH:?Please set CLIP_PATH}
DEVICE=${DEVICE:-cuda:0}

python train_stage1.py --train-root "$TRAIN_ROOT" --val-root "$VAL_ROOT" --xception-pretrained "$XCEPTION_PRETRAINED" --output-dir outputs/stage1 --device "$DEVICE" --batch-size 64 --lr 2e-4 --optimizer adamw
python train_stage2.py --train-root "$TRAIN_ROOT" --val-root "$VAL_ROOT" --stage1-checkpoint outputs/stage1/best.pt --clip-path "$CLIP_PATH" --output-dir outputs/stage2 --device "$DEVICE" --batch-size 32 --lr 5e-4 --optimizer sgd --auxiliary-weight 0.1
