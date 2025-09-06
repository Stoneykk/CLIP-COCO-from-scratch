# 🛠️ CLIP from Scratch - MVP 构建任务清单

目标：复现一个最小版本的 CLIP 模型，使用 ResNet-50 和冻结的 DistilBERT，在 COCO Caption 数据集上实现基本图文对比训练流程。

---

## ✅ 阶段 1：项目初始化与基础模块准备

### 任务 1：初始化项目文件夹结构
- 创建完整的项目文件/文件夹结构（参考 architecture.md）
- 输出：空壳结构搭建完毕，可用 `tree` 查看结构是否正确

### 任务 2：实现 `image_encoder.py` 中的 ResNet-50 封装
- 加载 torchvision 的预训练 ResNet-50，去除分类头
- 输入：`[B, 3, 224, 224]`
- 输出：`[B, 2048]`
- 测试：是否能 forward 成功，输出维度是否正确

### 任务 3：实现 `text_encoder.py` 中的冻结 DistilBERT 封装
- 加载 `distilbert-base-uncased`，冻结参数
- 输入：`input_ids`, `attention_mask`
- 输出：`[B, 768]`（CLS向量）
- 测试：是否 forward 成功，参数是否已冻结

### 任务 4：实现 `clip_model.py`，组合图像+文本编码器并加入 projection head
- 拼接 image + text encoder，加上 projection head（线性层）
- 输出：归一化向量 `image_proj`, `text_proj`，形状为 `[B, D]`
- 测试：dummy batch 正常前向传播

---

## 📊 阶段 2：数据准备与验证流通路

### 任务 5：实现 `tokenizer.py`
- 封装 tokenizer 加载与文本编码函数
- 输入：`List[str]`
- 输出：`input_ids`, `attention_mask`
- 测试：输出是否正确，是否自动 padding

### 任务 6：实现 `coco_dataset.py` 的 Dataset 类
- 输入：图像路径 + captions.json
- 输出：图像张量，input_ids，attention_mask
- 测试：`__getitem__` 输出是否可输入模型

### 任务 7：实现 DataLoader（使用 COCO Caption 子集）
- 实例化 Dataset 并构建 DataLoader
- 测试：能否 batch 迭代，输出维度是否正确

---

## 🧠 阶段 3：训练与损失函数构建

### 任务 8：实现 `info_nce_loss()` 函数
- 输入：归一化后的图文向量
- 输出：标量 loss
- 测试：构造正负例验证 loss 正确性

### 任务 9：实现训练前向与 loss 计算逻辑（不含优化器）
- 目标：验证模型与数据通路是否通畅
- 测试：打印 loss 值、输出维度

### 任务 10：添加优化器与训练循环（1 epoch）
- 加入 Adam 优化器 + 简单训练循环
- 测试：loss 是否收敛

---

## 🔁 阶段 4：验证与评估模块构建

### 任务 11：实现 Recall@K 指标（在 `metrics.py`）
- 输入：图文嵌入向量
- 输出：Recall@1, 5, 10
- 测试：toy case 检查结果是否合理

### 任务 12：写测试脚本 `test_forward_pass.py`
- 检查模型、数据、loss 全流程是否打通

---

## 🎯 阶段 5：主入口与可视化

### 任务 13：实现 `main.py`
- 从配置加载模型 + dataset + optimizer + train loop
- 输入：`configs/config.yaml`
- 测试：是否能完整启动训练

### 任务 14：实现 `logger.py`（可选）
- 使用 logging / Tensorboard 记录 loss 与指标

---

## ✅ 最终验收标准

- [ ] 能用 COCO Caption 子集跑通完整训练流程
- [ ] 模型由 ResNet + 冻结 DistilBERT + 投影层构成
- [ ] 使用 InfoNCE loss 收敛
- [ ] 能评估 Recall@K
- [ ] 每个模块单测完整，后续可扩展