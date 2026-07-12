# AIGI_SFAHDetection
This is the official code for our work 'Suspicious Frequency Amplified Hybrid Framework for Generalizable AIGC Image Detection'. Code coming soon.

## 1. 完整流程

### 第一阶段：Sobel + NPR + Xception

输入图像先经过 Sobel 梯度增强，再计算 Neighboring Pixel Residual（NPR），随后由 Xception 完成真假二分类。第一阶段训练 Xception 全部分支，并根据验证集指标保存 `best.pt`。

### 第二阶段：冻结 Xception + CLIP ViT-L/14 LoRA

第二阶段从第一阶段 `best.pt` 加载 Xception，并始终冻结其参数和 BatchNorm 统计量。CLIP ViT-L/14 的上半部分视觉 Transformer 在 Q/K/V 上插入 LoRA；同时保留原方法中的：

- Xception 频率通道注意力；
- CLIP patch-token GMM Transformer；
- Xception/CLIP 512 维投影与相加融合；
- 主分类损失和四个辅助分类损失。

第二阶段新 checkpoint 仅保存可训练参数，因此测试时需要第一阶段 checkpoint 和原始 CLIP ViT-L/14 权重；路径会写入 metadata，也可在测试命令中覆盖。测试入口同时兼容原工程保存的完整 `clip_lora_eeFrozen_DCT4` checkpoint，此时会自动转换旧参数名。

## 2. 数据目录

代码递归扫描图像，并通过目录名称判断标签。推荐结构：

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

同时兼容 `nature/ai` 和 `real/fake`。损坏图像会在 batch 拼接时跳过，测试结果会记录 `num_skipped`。

## 3. 安装

```bash
pip install -r requirements.txt
```

需要准备：

- Xception ImageNet 预训练权重，例如 `xception-b5690688.pth`；
- OpenAI CLIP `ViT-L-14.pt`；
- 训练、验证和测试数据路径。

## 4. 第一阶段训练

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


## 5. 第二阶段训练

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

第二阶段默认设置与原始方案一致：LoRA rank=4、alpha=0.5、QKV、ViT-L/14 half-up blocks。

## 6. 完整测试

### 测试第一阶段

```bash
python test.py \
  --stage 1 \
  --checkpoint outputs/stage1/best.pt \
  --test-root /path/to/test \
  --subsets progan,stylegan,biggan \
  --output-dir results/stage1 \
  --device cuda:0
```

### 测试第二阶段

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

若 `--subsets` 为空，则直接把 `--test-root` 当作一个测试集。输出包括：

- `metrics.csv` / `metrics.json`；
- 每个子集的逐图预测 `predictions_<subset>.csv` 以及汇总的 `predictions_all.csv`；
- 每个子集、macro mean 和 pooled overall 的 ACC、balanced ACC、AP、AUC、EER、F1、precision、recall、real accuracy、fake accuracy；
- 发现图像数、成功解码数和跳过数。


### 测试 checkpoint

```bash
python test.py \
  --stage 2 \
  --checkpoint /path/to/legacy_full_stage2.pt \
  --test-root /path/to/test \
  --subsets progan,stylegan,biggan \
  --output-dir results/legacy_stage2 \
  --device cuda:0
```

Pretrained model avaliable at \fref{https://drive.google.com/file/d/1AVcRZLCW1rETdh68yz_GA25rwrjfqlWc/view?usp=sharing}{link}.
