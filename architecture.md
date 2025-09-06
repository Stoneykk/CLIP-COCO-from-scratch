clip_from_scratch/
│
├── data/                        # 存放训练数据（图像 + 文本）
│   └── coco/                    # 比如 COCO caption 数据集结构
│       ├── images/             
│       └── captions.json       
│
├── models/
│   ├── __init__.py
│   ├── image_encoder.py        # ResNet50 图像编码器模块
│   ├── text_encoder.py         # DistilBERT 文本编码器模块（冻结）
│   └── clip_model.py           # 整体 CLIP 模型（拼接 image 和 text encoder）
│
├── datasets/
│   ├── __init__.py
│   └── coco_dataset.py         # 构建 PyTorch Dataset 处理图像 + 文本对
│
├── train/
│   ├── __init__.py
│   └── train_clip.py           # 训练主循环（包含 loss、优化器、日志）
│
├── utils/
│   ├── __init__.py
│   ├── tokenizer.py            # 加载和缓存 DistilBERT tokenizer
│   ├── metrics.py              # 计算 R@1、R@5、R@10、InfoNCE 等
│   └── logger.py               # 可选：日志记录、可视化
│
├── configs/
│   └── config.yaml             # 超参数、路径等统一配置
│
├── tests/
│   └── test_forward_pass.py    # 单元测试：检查模型正向传播是否无误
│
├── main.py                     # 启动训练流程（入口文件）
│
├── requirements.txt            # Python 依赖
└── README.md                   # 项目介绍与使用指南
