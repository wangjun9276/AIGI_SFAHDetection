#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT=${TEST_ROOT:?Please set TEST_ROOT}
CLIP_PATH=${CLIP_PATH:?Please set CLIP_PATH}
DEVICE=${DEVICE:-cuda:0}
SUBSETS=${SUBSETS:-}

python test.py --stage 2 --checkpoint outputs/stage2/best.pt --stage1-checkpoint outputs/stage1/best.pt --clip-path "$CLIP_PATH" --test-root "$TEST_ROOT" --subsets "$SUBSETS" --output-dir results/stage2 --device "$DEVICE"
