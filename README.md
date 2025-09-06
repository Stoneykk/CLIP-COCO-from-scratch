# CLIP-COCO-from-scratch

A minimal CLIP (Contrastive Language-Image Pre-training) implementation from scratch using ResNet-50 and frozen DistilBERT, trained on COCO Caption dataset for multimodal learning.

## 🎯 Project Overview

This project implements a simplified version of CLIP that learns to associate images with their corresponding text descriptions through contrastive learning. The model consists of:

- **Image Encoder**: ResNet-50 (pre-trained on ImageNet)
- **Text Encoder**: DistilBERT (LoRA+PEFT)
- **Projection Head**: Linear layers to map both encoders to a common embedding space
- **Loss Function**: InfoNCE contrastive loss

## 📦 Project Structure

```
CLIP-COCO-from-scratch/
├── data/                        # Training data (images + text)
│   └── coco/                    # COCO caption dataset structure
│       ├── images/             
│       └── captions.json       
├── models/                      # Model definitions
│   ├── image_encoder.py        # ResNet50 image encoder
│   ├── text_encoder.py         # DistilBERT text encoder (frozen)
│   └── clip_model.py           # Complete CLIP model
├── datasets/                    # Dataset processing
│   └── coco_dataset.py         # COCO dataset loader
├── train/                       # Training utilities
│   └── train_clip.py           # Training loop
├── utils/                       # Utility functions
│   ├── tokenizer.py            # Text tokenizer
│   ├── metrics.py              # Evaluation metrics
│   └── logger.py               # Logging utilities
├── configs/                     # Configuration files
│   └── config.yaml             # Hyperparameters
├── tests/                       # Test files
│   └── test_forward_pass.py    # Unit tests
└── main.py                     # Training entry point
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Stoneykk/CLIP-COCO-from-scratch.git
cd CLIP-COCO-from-scratch
pip install -r requirements.txt
```

### Training

```bash
python main.py
```

## 🔧 Development Status

🚧 **Under Development** - Currently completed:

### ✅ Phase 1: Project Initialization & Core Modules (COMPLETED)
- ✅ Project structure setup
- ✅ ResNet-50 image encoder implementation
- ✅ Frozen DistilBERT text encoder implementation  
- ✅ Complete CLIP model with projection heads

### 🔄 Phase 2: Data Preparation & Validation Pipeline (IN PROGRESS)
- ⏳ Tokenizer implementation
- ⏳ COCO dataset loader
- ⏳ DataLoader setup

### ⏳ Phase 3: Training & Loss Functions
- ⏳ InfoNCE loss implementation
- ⏳ Training forward pass
- ⏳ Optimizer and training loop

### ⏳ Phase 4: Validation & Evaluation
- ⏳ Recall@K metrics
- ⏳ Test scripts

### ⏳ Phase 5: Main Entry & Visualization
- ⏳ Main training script
- ⏳ Logging and visualization

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9.0+
- Transformers 4.20.0+
- CUDA (optional, for GPU acceleration)


## 🙏 Acknowledgments

- OpenAI CLIP for the original paper and inspiration
- Hugging Face Transformers for DistilBERT
- PyTorch team for the deep learning framework
- COCO dataset creators
