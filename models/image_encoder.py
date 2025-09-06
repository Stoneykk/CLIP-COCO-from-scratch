

"""
ResNet-50 图像编码器模块
加载预训练的ResNet-50，去除分类头，输出2048维特征向量
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ImageEncoder(nn.Module):
    """
    ResNet-50图像编码器
    
    输入: [B, 3, 224, 224] - 批次图像
    输出: [B, 2048] - 图像特征向量
    """
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        """
        初始化图像编码器
        
        Args:
            pretrained (bool): 是否使用预训练权重，默认True
            freeze_backbone (bool): 是否冻结backbone参数，默认False
        """
        super(ImageEncoder, self).__init__()
        
        # 加载预训练的ResNet-50
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # 去除分类头，保留特征提取部分
        # ResNet-50的最后一层是fc层，我们只需要到avgpool层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 冻结backbone参数（可选）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 输出特征维度
        self.feature_dim = 2048
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像张量 [B, 3, 224, 224]
            
        Returns:
            torch.Tensor: 图像特征向量 [B, 2048]
        """
        # 通过ResNet backbone
        features = self.backbone(x)  # [B, 2048, 1, 1]
        
        # 展平特征向量
        features = features.view(features.size(0), -1)  # [B, 2048]
        
        return features
    
    def get_feature_dim(self) -> int:
        """返回特征维度"""
        return self.feature_dim


def test_image_encoder():
    """测试图像编码器"""
    print("Testing ImageEncoder...")
    
    # 创建模型
    encoder = ImageEncoder(pretrained=True, freeze_backbone=False)
    
    # 创建测试输入
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Input shape: {test_input.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = encoder(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [{batch_size}, 2048]")
    
    # 验证输出维度
    assert output.shape == (batch_size, 2048), f"Expected shape ({batch_size}, 2048), got {output.shape}"
    
    print("✅ ImageEncoder test passed!")
    print(f"Feature dimension: {encoder.get_feature_dim()}")
    
    return True


if __name__ == "__main__":
    test_image_encoder()
