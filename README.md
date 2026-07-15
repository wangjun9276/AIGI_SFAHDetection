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

### Additional Post-processing Robustness Results

| Method | Metric | Resize 0.7 | Resize 0.5 | Crop 0.8 | Crop 0.5 | Median 3 | Median 5 | Median 7 | Bright 1.5 | Contrast 1.5 | Color 1.5 | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2P-CLIP | ACC | 83.71 | 82.20 | 85.72 | 84.80 | 84.94 | 73.40 | 63.33 | 73.41 | 72.01 | 79.82 | .60 |
| C2P-CLIP | AP | 94.87 | 93.42 | 95.60 | 94.08 | 94.91 | 88.98 | 82.43 | 90.50 | 90.20 | 95.39 | .10 |
| FatFormer | ACC | 82.54 | 80.13 | 85.19 | 86.03 | .00 | .00 | 55.23 | 81.63 | 82.49 | 86.19 | .00 |
| FatFormer | AP | 90.26 | 87.86 | 93.11 | 95.29 | .00 | .00 | 77.17 | 89.23 | 89.76 | 92.84 | .80 |
| SAFE | ACC | 80.77 | 54.31 | 64.10 | 87.53 | 83.55 | 71.57 | 64.25 | 90.13 | 87.64 | 92.12 | .00 |
| SAFE | AP | 86.60 | 58.20 | 73.52 | 95.86 | 88.72 | 77.65 | 69.60 | 95.20 | 94.34 | 95.97 | .00 |
| ForensicsMOE | ACC | .00 | .00 | .00 | .00 | .00 | .00 | .00 | 83.77 | 85.10 | 95.87 | .70 |
| ForensicsMOE | AP | .00 | .00 | .00 | .00 | .00 | .00 | .00 | 92.54 | 92.89 | 99.02 | .90 |
| RINE | ACC | 80.89 | 77.70 | 85.00 | 83.69 | 84.26 | 63.42 | 54.81 | 85.47 | 78.26 | 79.78 | .50 |
| RINE | AP | 88.03 | 83.30 | 91.15 | 91.27 | 91.57 | 83.29 | 61.05 | 93.41 | 88.59 | 88.66 | .40 |
| VIB | ACC | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .80 |
| VIB | AP | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .60 |
| **SFAH** | **ACC** | **89.70** | **80.10** | **.00** | **.00** | **82.30** | **62.20** | **53.80** | **74.30** | **71.00** | **89.80** | **.20** |
| **SFAH** | **AP** | **96.60** | **90.00** | **.00** | **.00** | **97.20** | **89.30** | **80.00** | **86.70** | **84.50** | **97.00** | **.55** |
