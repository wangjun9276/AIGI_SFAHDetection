# 原始 Hybdir 代码梳理与重构说明

## 一、原始代码的实际训练逻辑

### 第一阶段

原始 `run.sh` 中第一阶段计划调用：

```bash
python main.py --method=xception_sobel_pass_npr ...
```

对应模型的数据流为：

```text
RGB image
  -> Sobel gradient magnitude
  -> residual enhancement: image + Sobel(image)
  -> NPR: x - nearest_upsample(nearest_downsample(x))
  -> Xception feature map [B, 2048, 7, 7]
  -> global pooling + binary classifier
  -> fake logit
```

第一阶段应训练 Xception 全部分支。原代码从 `xception-b5690688.pth` 加载不含分类头的 ImageNet 权重。

### 第二阶段

原始第二阶段对应 `clip_lora_eeFrozen_DCT4`：

```text
                             +-> original pooled Xception feature ----+
RGB -> frozen Stage-1 Xception                                      |
                             +-> frequency-attended feature ----------+-> 2048->512

RGB -> CLIP ViT-L/14 + QKV LoRA -> global CLIP feature --------------+
                              +-> 256 patch tokens
                                  -> 2-layer GMM Transformer
                                  -> patch pooled feature ------------+-> 768->512

Xception 512 feature + CLIP 512 feature -> binary classifier
```

融合公式对应原实现：

```text
xception_vector = map(base_xception + freq_scale * attended_xception)
clip_vector = project(global_clip - gmm_scale * patch_clip)
logit = head(xception_vector + clip_vector)
```

训练时还应计算四个辅助分类头：原始 Xception、频率注意 Xception、全局 CLIP、patch CLIP。总损失为主 BCE 加 `0.1 × 四个辅助 BCE 之和`。

## 二、原始测试逻辑

原始 `inference.py` 会：

1. 根据硬编码数据集名拼接测试目录；
2. 为每个生成器创建 DataLoader；
3. 从硬编码 checkpoint 加载模型；
4. 对每张图输出 sigmoid 概率；
5. 以 0.5 阈值计算 ACC、AP、real ACC 和 fake ACC；
6. 将各子集结果写入 CSV。

## 三、发现的关键问题

1. `main.py` 直接把第一阶段模型的 `(pred, feature)` 元组当 tensor 使用。
2. `main_proposal.py` 期望第二阶段返回五个输出，但 `clip_lora_eeFrozen_DCT4.forward()` 实际只返回主预测。
3. 训练期间根据测试集均值做 early stopping，产生测试集泄漏。
4. 验证集错误使用训练增强 `train_augment()`，导致验证结果随机波动。
5. 模型文件 import 时立即在固定 `cuda:1` 加载 CLIP，设备参数失效并占用额外显存。
6. 第一阶段 checkpoint、CLIP 权重、训练集和测试集路径全部硬编码。
7. 第二阶段虽然冻结 Xception 参数，但 `model.train()` 仍会更新其 BatchNorm 统计量。
8. 梯度累积遇到最后不足一个累积周期的 batch 时不会执行 optimizer step。
9. `num_workers=0` 时仍设置 `persistent_workers=True/prefetch_factor`，会触发 DataLoader 参数错误。
10. 测试预处理注释掉小图 resize，小于 224 的图像可能无法形成统一 batch。
11. 损坏图像被替换成全零图并保留原标签，会污染训练和评估指标。
12. 测试代码对单类别子集直接调用 AUC/类别准确率，可能报错。
13. 多个入口脚本、模型变体、可视化和复杂度统计相互复制且配置不一致。
14. 原 `utils/loss.py` 存在 `from __future__` 位置错误，整个工程不能通过完整语法编译。
15. 原第二阶段保存整个冻结 CLIP 和 Xception，checkpoint 体积接近 1 GB，实际只需保存 LoRA 与新模块。

## 四、重构后的文件职责

```text
train_stage1.py          第一阶段训练入口
train_stage2.py          第二阶段训练入口
test.py                  两阶段统一测试入口
hybdir/data.py           数据扫描、增强、安全 collate
hybdir/models.py         两个最终模型
hybdir/xception.py       精简 Xception
hybdir/frequency_attention.py
hybdir/gmm_transformer.py
hybdir/engine.py         训练 epoch、验证、测试和优化器
hybdir/trainer.py        early stopping 与 checkpoint
hybdir/checkpoint.py     新旧权重加载和兼容转换
hybdir/metrics.py        完整二分类指标
clip/model.py            仅保留 CLIP 视觉编码器
loralib/                 仅保留视觉 QKV LoRA
```

## 五、checkpoint 规则

### Stage 1

`best.pt` 保存完整第一阶段模型，且兼容原始 `xception_sobel_pass_npr` raw state dict。

### Stage 2 新格式

只保存：

- CLIP LoRA 参数；
- 频率注意力模块；
- GMM patch Transformer；
- 两个投影层；
- 主/辅助分类头；
- `freq_scale` 和 `gmm_scale`。

冻结的 Xception 和 CLIP 基础权重不重复保存。测试时从 checkpoint metadata 或命令行读取其路径。

### Stage 2 原格式兼容

`test.py` 会自动识别原工程保存的完整 `clip_lora_eeFrozen_DCT4` state dict，并映射：

- `xception_sobel_pass_npr -> xception_branch`；
- `att -> frequency_attention`；
- `transformer.layers -> patch_transformer.blocks`；
- `project/map/head_aux/head -> 新命名`。

因此已有完整第二阶段 checkpoint 可以直接测试，不要求重新训练。

## 六、已完成的验证

- 全工程 `compileall` 通过；
- 第一阶段 224×224 前向输出为 `[B,1]`，特征为 `[B,2048,7,7]`；
- 使用旧 Xception 参数命名构造 checkpoint，严格加载到新模型成功；
- 第一阶段完成一轮真实 CPU 训练、验证、best checkpoint 保存和重新加载；
- 测试入口完成逐图 CSV、损坏图跳过、macro mean 和 pooled overall 指标输出；
- 第二阶段通过伪 CLIP 完成主输出、四辅助输出、反向传播和 checkpoint 加载；
- 新视觉 CLIP 与原视觉 CLIP 的 state-dict key 和 tensor shape 完全一致；
- 精简 LoRA 在零初始化时与原 MHA 输出严格一致，并保留原 checkpoint 参数名；
- 构造旧式完整 Stage-2 checkpoint 后，所有可训练参数可无误映射到新模型。
