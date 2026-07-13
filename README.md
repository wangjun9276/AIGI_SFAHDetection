# AIGI_SFAHDetection
This is the official code for our work 'Suspicious Frequency Amplified Hybrid Framework for Generalizable AIGC Image Detection'. Code coming soon.

## 1. Overview

### Stage 1：Sobel + NPR + Xception

The input image is first enhanced by Sobel gradient, then the Neighboring Pixel Residual (NPR) is calculated, and then Xception is used to complete the true and false classification. In the first stage, all branches of Xception are trained and `best.pt` is saved according to the verification set indicators.

### Stage 2：freeze Xception + CLIP ViT-L/14 LoRA

The second stage loads Xception from the first stage `best.pt` and always freezes its parameters and BatchNorm statistics. The upper half of the CLIP ViT-L/14 visual Transformer inserts LoRA on Q/K/V; while retaining:
- Xception frequency channel attention;
- CLIP patch-token GMM Transformer;
- Xception/CLIP 512-dimensional projection and additive fusion;
- Main classification loss and four auxiliary classification losses.

The new checkpoint in the second stage only saves trainable parameters, so the first stage checkpoint and the original CLIP ViT-L/14 weights are required for testing; the path will be written into the metadata and can also be overridden in the test command. The test entrance is also compatible with the complete `clip_lora_eeFrozen_DCT4` checkpoint saved in the original project, and the old parameter names will be automatically converted at this time.

## 2. data structure

The code recursively scans images and determines labels based on directory names. Recommended structure:

```text
train/
├── 0_real/
└── 1_fake/
val/
├── 0_real/
└── 1_fake/
test/
├── progan/
│   ├── 0_real/
│   └── 1_fake/
└── stylegan/
    ├── 0_real/
    └── 1_fake/
```

Compatible with both `nature/ai` and `real/fake`. Corrupted images are skipped during batch concatenation, and the test results record the `num_skipped` count.

## 3. Install

```bash
pip install -r requirements.txt
```

Prerequisites:

- Xception ImageNet pre-trained weights (e.g., `xception-b5690688.pth`);
- OpenAI CLIP `ViT-L-14.pt`;
- Paths to training, validation, and test data.

## 4. Stage 1 Training

```bash
python train_stage1.py \
  --train-root /path/to/train \
  --val-root /path/to/val \
  --test-root /path/to/optional_test \
  --xception-pretrained /path/to/xception-b5690688.pth \
  --output-dir outputs/stage1 \
  --device cuda:0 \
  --batch-size 64 \
  --optimizer adamw \
  --lr 2e-4
```


## 5. Stage 2 training

```bash
python train_stage2.py \
  --train-root /path/to/train \
  --val-root /path/to/val \
  --test-root /path/to/optional_test \
  --stage1-checkpoint outputs/stage1/best.pt \
  --clip-path /path/to/ViT-L-14.pt \
  --output-dir outputs/stage2 \
  --device cuda:0 \
  --batch-size 32 \
  --optimizer sgd \
  --lr 5e-4 \
  --auxiliary-weight 0.1
```

The default settings of the second stage are consistent with the original solution: LoRA rank=4, alpha=0.5, QKV, ViT-L/14 half-up blocks.

## 6. Complete test

### First phase of testing

```bash
python test.py \
  --stage 1 \
  --checkpoint outputs/stage1/best.pt \
  --test-root /path/to/test \
  --subsets progan,stylegan,biggan \
  --output-dir results/stage1 \
  --device cuda:0
```

### Second phase of testing

```bash
python test.py \
  --stage 2 \
  --checkpoint outputs/stage2/best.pt \
  --stage1-checkpoint outputs/stage1/best.pt \
  --clip-path /path/to/ViT-L-14.pt \
  --test-root /path/to/test \
  --subsets progan,stylegan,biggan \
  --output-dir results/stage2 \
  --device cuda:0
```

If `--subsets` is empty, `--test-root` will be directly used as a test set. The output includes:

- `metrics.csv` / `metrics.json`;
- Image-by-image predictions `predictions_<subset>.csv` for each subset and aggregated `predictions_all.csv`;
- ACC, balanced ACC, AP, AUC, EER, F1, precision, recall, real accuracy, fake accuracy of each subset, macro mean and pooled overall;
- Number of images found, successfully decoded and skipped.


### Test checkpoint

```bash
python test.py \
  --stage 2 \
  --checkpoint /path/to/legacy_full_stage2.pt \
  --test-root /path/to/test \
  --subsets progan,stylegan,biggan \
  --output-dir results/legacy_stage2 \
  --device cuda:0
```

The pretrained model is available at [this link](https://drive.google.com/file/d/1AVcRZLCW1rETdh68yz_GA25rwrjfqlWc/view?usp=sharing).


### Additional Post-processing Robustness Evaluation
