# CLIP from Scratch

一个从零开始实现的最小版本CLIP模型，使用ResNet-50和冻结的DistilBERT，在COCO Caption数据集上进行图文对比学习。

## 项目结构

```
clip_from_scratch/
├── data/                        # 存放训练数据（图像 + 文本）
│   └── coco/                    # COCO caption 数据集结构
│       ├── images/             
│       └── captions.json       
├── models/                      # 模型定义
│   ├── image_encoder.py        # ResNet50 图像编码器
│   ├── text_encoder.py         # DistilBERT 文本编码器（冻结）
│   └── clip_model.py           # 整体 CLIP 模型
├── datasets/                    # 数据集处理
│   └── coco_dataset.py         # COCO数据集加载器
├── train/                       # 训练相关
│   └── train_clip.py           # 训练主循环
├── utils/                       # 工具函数
│   ├── tokenizer.py            # 文本tokenizer
│   ├── metrics.py              # 评估指标
│   └── logger.py               # 日志记录
├── configs/                     # 配置文件
│   └── config.yaml             # 超参数配置
├── tests/                       # 测试文件
│   └── test_forward_pass.py    # 单元测试
└── main.py                     # 训练入口
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python main.py
```

## 开发状态

🚧 项目正在开发中，当前已完成项目结构搭建。
