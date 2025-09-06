"""
构建 PyTorch Dataset 处理图像 + 文本对
支持COCO Caption数据集的加载和预处理
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Any
import random

try:
    from ..utils.tokenizer import TextTokenizer
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.tokenizer import TextTokenizer


class COCODataset(Dataset):
    """
    COCO Caption数据集类
    
    功能：
    - 加载COCO图像和对应的文本描述
    - 图像预处理（resize, normalize等）
    - 文本tokenization
    - 返回可直接输入模型的张量
    """
    
    def __init__(
        self,
        images_dir: str,
        captions_file: str,
        tokenizer: Optional[TextTokenizer] = None,
        max_captions_per_image: int = 5,
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        split: str = "train"
    ):
        """
        初始化COCO数据集
        
        Args:
            images_dir (str): 图像文件夹路径
            captions_file (str): captions.json文件路径
            tokenizer (TextTokenizer, optional): 文本tokenizer
            max_captions_per_image (int): 每张图像最多使用的caption数量
            image_size (int): 图像尺寸，默认224
            transform (transforms.Compose, optional): 图像变换
            split (str): 数据集分割（train/val/test）
        """
        self.images_dir = images_dir
        self.captions_file = captions_file
        self.max_captions_per_image = max_captions_per_image
        self.image_size = image_size
        self.split = split
        
        # 初始化tokenizer
        if tokenizer is None:
            self.tokenizer = TextTokenizer(max_length=77)
        else:
            self.tokenizer = tokenizer
        
        # 设置图像变换
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        
        # 加载数据
        self.data = self._load_data()
        
    def _get_default_transform(self) -> transforms.Compose:
        """获取默认的图像变换"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet标准化
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载COCO数据"""
        print(f"Loading COCO {self.split} data...")
        
        # 加载captions文件
        with open(self.captions_file, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
        
        # 构建数据列表
        data = []
        
        # 创建image_id到captions的映射
        image_id_to_captions = {}
        for annotation in captions_data['annotations']:
            image_id = annotation['image_id']
            caption = annotation['caption']
            
            if image_id not in image_id_to_captions:
                image_id_to_captions[image_id] = []
            image_id_to_captions[image_id].append(caption)
        
        # 创建image_id到image信息的映射
        image_id_to_info = {}
        for image_info in captions_data['images']:
            image_id_to_info[image_info['id']] = image_info
        
        # 构建数据项
        for image_id, captions in image_id_to_captions.items():
            if image_id in image_id_to_info:
                image_info = image_id_to_info[image_id]
                image_path = os.path.join(self.images_dir, image_info['file_name'])
                
                # 检查图像文件是否存在
                if os.path.exists(image_path):
                    # 限制每个图像的caption数量
                    if len(captions) > self.max_captions_per_image:
                        captions = random.sample(captions, self.max_captions_per_image)
                    
                    data.append({
                        'image_id': image_id,
                        'image_path': image_path,
                        'captions': captions,
                        'file_name': image_info['file_name']
                    })
        
        print(f"Loaded {len(data)} image-caption pairs")
        return data
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据项
        
        Args:
            idx (int): 数据索引
            
        Returns:
            Dict[str, torch.Tensor]: 包含图像和文本张量的字典
        """
        item = self.data[idx]
        
        # 加载图像
        image_path = item['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个黑色图像作为fallback
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        
        # 应用图像变换
        image_tensor = self.transform(image)
        
        # 随机选择一个caption
        caption = random.choice(item['captions'])
        
        # 编码文本
        encoded_text = self.tokenizer.encode_single_text(caption)
        
        return {
            'image': image_tensor,
            'input_ids': encoded_text['input_ids'].squeeze(0),  # 移除batch维度
            'attention_mask': encoded_text['attention_mask'].squeeze(0),
            'caption': caption,
            'image_id': item['image_id'],
            'file_name': item['file_name']
        }
    
    def get_item_by_image_id(self, image_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """根据image_id获取数据项"""
        for i, item in enumerate(self.data):
            if item['image_id'] == image_id:
                return self.__getitem__(i)
        return None
    
    def get_captions_by_image_id(self, image_id: int) -> Optional[List[str]]:
        """根据image_id获取所有captions"""
        for item in self.data:
            if item['image_id'] == image_id:
                return item['captions']
        return None


def create_sample_captions_file(output_path: str, num_samples: int = 100):
    """
    创建示例captions文件用于测试
    
    Args:
        output_path (str): 输出文件路径
        num_samples (int): 样本数量
    """
    # 创建示例数据
    sample_data = {
        "images": [],
        "annotations": []
    }
    
    # 创建示例图像信息
    for i in range(num_samples):
        image_id = i + 1
        file_name = f"sample_{i:06d}.jpg"
        
        sample_data["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": 640,
            "height": 480
        })
        
        # 为每个图像创建多个captions
        captions = [
            f"A photo of sample image {i+1}",
            f"Sample picture number {i+1}",
            f"Test image {i+1} for CLIP training"
        ]
        
        for j, caption in enumerate(captions):
            sample_data["annotations"].append({
                "id": i * 3 + j + 1,
                "image_id": image_id,
                "caption": caption
            })
    
    # 保存文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample captions file: {output_path}")


def test_coco_dataset():
    """测试COCO数据集"""
    print("Testing COCODataset...")
    
    # 创建示例数据
    sample_images_dir = "data/coco/images"
    sample_captions_file = "data/coco/sample_captions.json"
    
    # 确保目录存在
    os.makedirs(sample_images_dir, exist_ok=True)
    
    # 创建示例captions文件
    create_sample_captions_file(sample_captions_file, num_samples=10)
    
    # 创建示例图像文件（黑色图像）
    for i in range(10):
        image_path = os.path.join(sample_images_dir, f"sample_{i:06d}.jpg")
        if not os.path.exists(image_path):
            # 创建一个简单的测试图像
            test_image = Image.new('RGB', (640, 480), (i*25, i*25, i*25))
            test_image.save(image_path)
    
    # 创建数据集
    dataset = COCODataset(
        images_dir=sample_images_dir,
        captions_file=sample_captions_file,
        max_captions_per_image=3
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试__getitem__
    if len(dataset) > 0:
        sample = dataset[0]
        
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Attention mask shape: {sample['attention_mask'].shape}")
        print(f"Caption: {sample['caption']}")
        
        # 验证输出维度
        assert sample['image'].shape == (3, 224, 224), f"Expected image shape (3, 224, 224), got {sample['image'].shape}"
        assert sample['input_ids'].shape == (77,), f"Expected input_ids shape (77,), got {sample['input_ids'].shape}"
        assert sample['attention_mask'].shape == (77,), f"Expected attention_mask shape (77,), got {sample['attention_mask'].shape}"
        
        # 测试多个样本
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"Sample {i}: {sample['caption']}")
    
    print("✅ COCODataset test passed!")
    
    return True


if __name__ == "__main__":
    test_coco_dataset()
