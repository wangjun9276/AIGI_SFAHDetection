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
| CNNSpot | ACC | 68.50 | 63.00 | 65.20 | 58.00 | 70.50 | 68.20 | 65.00 | 69.00 | 67.50 | 68.80 | 66.37 |
| CNNSpot | AP | 78.00 | 74.00 | 76.00 | 69.00 | 80.50 | 78.00 | 74.50 | 79.00 | 77.50 | 79.00 | 76.55 |
| FreqDect | ACC | 60.00 | 55.50 | 58.00 | 52.00 | 62.50 | 60.50 | 58.50 | 61.00 | 59.00 | 60.50 | 58.75 |
| FreqDect | AP | 62.00 | 58.00 | 60.00 | 53.00 | 68.00 | 65.00 | 62.00 | 66.00 | 63.50 | 65.50 | 62.30 |
| LGrad | ACC | 70.00 | 65.00 | 68.00 | 60.00 | 73.00 | 70.00 | 66.00 | 72.00 | 69.00 | 71.50 | 68.45 |
| LGrad | AP | 70.00 | 65.00 | 68.00 | 58.00 | 75.00 | 72.00 | 68.00 | 74.00 | 71.00 | 73.50 | 69.45 |
| FreqNet | ACC | 70.00 | 66.00 | 68.00 | 60.00 | 75.00 | 72.00 | 69.00 | 74.00 | 71.00 | 73.00 | 69.80 |
| FreqNet | AP | 76.00 | 72.00 | 74.00 | 65.00 | 82.00 | 79.00 | 75.00 | 81.00 | 78.00 | 80.00 | 76.20 |
| NPR | ACC | 75.00 | 68.00 | 73.00 | 62.00 | 88.00 | 82.00 | 74.00 | 90.00 | 86.00 | 89.00 | 78.70 |
| NPR | AP | 82.00 | 75.00 | 80.00 | 68.00 | 92.00 | 87.00 | 78.00 | 94.00 | 90.00 | 93.00 | 83.90 |
| C2P-CLIP | ACC | 82.00 | 77.00 | 80.00 | 72.00 | 84.00 | 82.00 | 79.00 | 84.00 | 82.00 | 84.00 | 80.60 |
| C2P-CLIP | AP | 95.00 | 91.00 | 94.00 | 86.00 | 96.00 | 94.00 | 91.00 | 95.00 | 94.00 | 95.00 | 93.10 |
| FatFormer | ACC | 83.00 | 78.00 | 82.00 | 74.00 | 86.00 | 84.00 | 80.00 | 85.00 | 83.00 | 85.00 | 82.00 |
| FatFormer | AP | 91.00 | 88.00 | 90.00 | 83.00 | 92.00 | 91.00 | 88.00 | 92.00 | 91.00 | 92.00 | 89.80 |
| SAFE | ACC | 82.00 | 74.00 | 78.00 | 68.00 | 90.00 | 85.00 | 78.00 | 90.13 | 87.64 | 92.12 | 82.30 |
| SAFE | AP | 88.00 | 82.00 | 85.00 | 75.00 | 94.00 | 90.00 | 82.00 | 95.20 | 94.34 | 95.97 | 87.60 |
| ForensicsMOE | ACC | 87.00 | 81.00 | 85.00 | 77.00 | 92.00 | 88.00 | 83.00 | 93.00 | 89.00 | 92.00 | 86.70 |
| ForensicsMOE | AP | 96.00 | 92.00 | 95.00 | 88.00 | 98.00 | 96.00 | 91.00 | 99.00 | 96.00 | 98.00 | 94.90 |
| RINE | ACC | 83.00 | 78.00 | 82.00 | 75.00 | 86.00 | 84.00 | 80.00 | 85.47 | 78.26 | 79.78 | 82.50 |
| RINE | AP | 90.00 | 86.00 | 89.00 | 82.00 | 94.00 | 91.00 | 87.00 | 93.41 | 88.59 | 88.66 | 89.40 |
| CSF | ACC | 76.00 | 70.00 | 74.00 | 66.00 | 82.00 | 78.00 | 72.00 | 84.00 | 80.00 | 83.00 | 76.50 |
| CSF | AP | 78.00 | 72.00 | 76.00 | 66.00 | 84.00 | 80.00 | 74.00 | 86.00 | 82.00 | 85.00 | 78.30 |
| VIB | ACC | 82.00 | 76.00 | 80.00 | 72.00 | 84.00 | 82.00 | 78.00 | 86.00 | 83.00 | 85.00 | 80.80 |
| VIB | AP | 88.00 | 84.00 | 87.00 | 80.00 | 91.00 | 89.00 | 85.00 | 92.00 | 89.00 | 91.00 | 87.60 |
| **SFAH** | **ACC** | **89.00** | **84.00** | **87.00** | **79.00** | **94.00** | **91.00** | **86.00** | **95.00** | **92.00** | **95.00** | **89.20** |
| **SFAH** | **AP** | **96.50** | **94.00** | **96.00** | **90.00** | **99.00** | **98.00** | **96.00** | **99.00** | **98.00** | **99.00** | **96.55** |
