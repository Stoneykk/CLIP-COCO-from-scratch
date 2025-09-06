"""
训练主循环（包含 loss、优化器、日志）
实现DataLoader创建和训练流程
"""

import torch
from torch.utils.data import DataLoader
import os
import sys
from typing import Dict, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.coco_dataset import COCODataset, create_sample_captions_file
from utils.tokenizer import TextTokenizer
from models.clip_model import CLIPModel


def create_dataloader(
    images_dir: str,
    captions_file: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    max_captions_per_image: int = 5,
    image_size: int = 224,
    split: str = "train"
) -> DataLoader:
    """
    创建COCO数据集的DataLoader
    
    Args:
        images_dir (str): 图像文件夹路径
        captions_file (str): captions.json文件路径
        batch_size (int): 批次大小，默认32
        shuffle (bool): 是否打乱数据，默认True
        num_workers (int): 数据加载进程数，默认0
        max_captions_per_image (int): 每张图像最多使用的caption数量
        image_size (int): 图像尺寸，默认224
        split (str): 数据集分割（train/val/test）
        
    Returns:
        DataLoader: 配置好的数据加载器
    """
    # 创建tokenizer
    tokenizer = TextTokenizer(max_length=77)
    
    # 创建数据集
    dataset = COCODataset(
        images_dir=images_dir,
        captions_file=captions_file,
        tokenizer=tokenizer,
        max_captions_per_image=max_captions_per_image,
        image_size=image_size,
        split=split
    )
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # 丢弃最后一个不完整的batch
    )
    
    return dataloader


def test_dataloader():
    """测试DataLoader功能"""
    print("Testing DataLoader...")
    
    # 创建示例数据
    sample_images_dir = "data/coco/images"
    sample_captions_file = "data/coco/sample_captions.json"
    
    # 确保目录存在
    os.makedirs(sample_images_dir, exist_ok=True)
    
    # 创建示例captions文件（如果不存在）
    if not os.path.exists(sample_captions_file):
        create_sample_captions_file(sample_captions_file, num_samples=20)
    
    # 创建示例图像文件（如果不存在）
    from PIL import Image
    for i in range(20):
        image_path = os.path.join(sample_images_dir, f"sample_{i:06d}.jpg")
        if not os.path.exists(image_path):
            # 创建一个简单的测试图像
            test_image = Image.new('RGB', (640, 480), (i*12, i*12, i*12))
            test_image.save(image_path)
    
    # 创建DataLoader
    batch_size = 4
    dataloader = create_dataloader(
        images_dir=sample_images_dir,
        captions_file=sample_captions_file,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        max_captions_per_image=3
    )
    
    print(f"DataLoader created with batch_size={batch_size}")
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # 测试batch迭代
    print("\nTesting batch iteration...")
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i+1}:")
        print(f"  Images shape: {batch['image'].shape}")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Captions: {batch['caption']}")
        
        # 验证输出维度
        expected_batch_size = min(batch_size, len(dataloader.dataset))
        assert batch['image'].shape == (expected_batch_size, 3, 224, 224), \
            f"Expected image shape ({expected_batch_size}, 3, 224, 224), got {batch['image'].shape}"
        assert batch['input_ids'].shape == (expected_batch_size, 77), \
            f"Expected input_ids shape ({expected_batch_size}, 77), got {batch['input_ids'].shape}"
        assert batch['attention_mask'].shape == (expected_batch_size, 77), \
            f"Expected attention_mask shape ({expected_batch_size}, 77), got {batch['attention_mask'].shape}"
        
        # 只测试前3个batch
        if i >= 2:
            break
    
    # 测试与模型的兼容性
    print("\nTesting model compatibility...")
    model = CLIPModel(embed_dim=256)
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch['image']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # 前向传播
            image_proj, text_proj = model(images, input_ids, attention_mask)
            
            print(f"Model output - Batch {i+1}:")
            print(f"  Image projection shape: {image_proj.shape}")
            print(f"  Text projection shape: {text_proj.shape}")
            
            # 验证模型输出维度
            expected_batch_size = min(batch_size, len(dataloader.dataset))
            assert image_proj.shape == (expected_batch_size, 256), \
                f"Expected image_proj shape ({expected_batch_size}, 256), got {image_proj.shape}"
            assert text_proj.shape == (expected_batch_size, 256), \
                f"Expected text_proj shape ({expected_batch_size}, 256), got {text_proj.shape}"
            
            # 只测试第一个batch
            break
    
    print("\n✅ DataLoader test passed!")
    print("✅ Model compatibility test passed!")
    
    return True


def get_dataloader_info(dataloader: DataLoader) -> Dict[str, Any]:
    """
    获取DataLoader信息
    
    Args:
        dataloader (DataLoader): 数据加载器
        
    Returns:
        Dict[str, Any]: DataLoader信息
    """
    return {
        'dataset_size': len(dataloader.dataset),
        'batch_size': dataloader.batch_size,
        'num_batches': len(dataloader),
        'shuffle': dataloader.sampler is not None and hasattr(dataloader.sampler, 'shuffle'),
        'num_workers': dataloader.num_workers,
        'pin_memory': dataloader.pin_memory,
        'drop_last': dataloader.drop_last
    }


if __name__ == "__main__":
    test_dataloader()
