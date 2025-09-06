"""
整体 CLIP 模型（拼接 image 和 text encoder）
组合ResNet-50图像编码器和DistilBERT文本编码器，加入投影头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

try:
    from .image_encoder import ImageEncoder
    from .text_encoder import TextEncoder
except ImportError:
    from image_encoder import ImageEncoder
    from text_encoder import TextEncoder


class CLIPModel(nn.Module):
    """
    CLIP模型：组合图像和文本编码器，加入投影头
    
    输入: 图像 [B, 3, 224, 224] 和 文本 (input_ids, attention_mask)
    输出: 归一化的图像和文本投影向量 [B, embed_dim]
    """
    
    def __init__(
        self,
        image_encoder: Optional[ImageEncoder] = None,
        text_encoder: Optional[TextEncoder] = None,
        embed_dim: int = 256,
        freeze_image_encoder: bool = False,
        freeze_text_encoder: bool = True
    ):
        """
        初始化CLIP模型
        
        Args:
            image_encoder (ImageEncoder, optional): 图像编码器，默认创建新的
            text_encoder (TextEncoder, optional): 文本编码器，默认创建新的
            embed_dim (int): 投影后的嵌入维度，默认256
            freeze_image_encoder (bool): 是否冻结图像编码器，默认False
            freeze_text_encoder (bool): 是否冻结文本编码器，默认True
        """
        super(CLIPModel, self).__init__()
        
        # 创建或使用提供的编码器
        if image_encoder is None:
            self.image_encoder = ImageEncoder(
                pretrained=True, 
                freeze_backbone=freeze_image_encoder
            )
        else:
            self.image_encoder = image_encoder
            
        if text_encoder is None:
            self.text_encoder = TextEncoder(freeze=freeze_text_encoder)
        else:
            self.text_encoder = text_encoder
        
        # 获取编码器的输出维度
        self.image_dim = self.image_encoder.get_feature_dim()  # 2048
        self.text_dim = self.text_encoder.get_feature_dim()    # 768
        self.embed_dim = embed_dim
        
        # 投影头：将不同维度的特征映射到相同的嵌入空间
        self.image_projection = nn.Linear(self.image_dim, embed_dim)
        self.text_projection = nn.Linear(self.text_dim, embed_dim)
        
        # 初始化投影层权重
        self._init_projection_layers()
        
    def _init_projection_layers(self):
        """初始化投影层权重"""
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.image_projection.weight)
        nn.init.zeros_(self.image_projection.bias)
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.zeros_(self.text_projection.bias)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像
        
        Args:
            images (torch.Tensor): 输入图像 [B, 3, 224, 224]
            
        Returns:
            torch.Tensor: 归一化的图像嵌入 [B, embed_dim]
        """
        # 通过图像编码器
        image_features = self.image_encoder(images)  # [B, 2048]
        
        # 通过投影层
        image_proj = self.image_projection(image_features)  # [B, embed_dim]
        
        # L2归一化
        image_proj = F.normalize(image_proj, p=2, dim=-1)
        
        return image_proj
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        编码文本
        
        Args:
            input_ids (torch.Tensor): 输入token IDs [B, seq_len]
            attention_mask (torch.Tensor): 注意力掩码 [B, seq_len]
            
        Returns:
            torch.Tensor: 归一化的文本嵌入 [B, embed_dim]
        """
        # 通过文本编码器
        text_features = self.text_encoder(input_ids, attention_mask)  # [B, 768]
        
        # 通过投影层
        text_proj = self.text_projection(text_features)  # [B, embed_dim]
        
        # L2归一化
        text_proj = F.normalize(text_proj, p=2, dim=-1)
        
        return text_proj
    
    def forward(
        self, 
        images: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            images (torch.Tensor): 输入图像 [B, 3, 224, 224]
            input_ids (torch.Tensor): 输入token IDs [B, seq_len]
            attention_mask (torch.Tensor): 注意力掩码 [B, seq_len]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (image_proj, text_proj) 归一化的投影向量
        """
        # 编码图像和文本
        image_proj = self.encode_image(images)
        text_proj = self.encode_text(input_ids, attention_mask)
        
        return image_proj, text_proj
    
    def get_embed_dim(self) -> int:
        """返回嵌入维度"""
        return self.embed_dim
    
    def get_image_dim(self) -> int:
        """返回图像编码器输出维度"""
        return self.image_dim
    
    def get_text_dim(self) -> int:
        """返回文本编码器输出维度"""
        return self.text_dim


def test_clip_model():
    """测试CLIP模型"""
    print("Testing CLIPModel...")
    
    # 创建模型
    model = CLIPModel(embed_dim=256)
    
    # 创建测试输入
    batch_size = 4
    seq_len = 32
    
    images = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"Images shape: {images.shape}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # 前向传播
    with torch.no_grad():
        image_proj, text_proj = model(images, input_ids, attention_mask)
    
    print(f"Image projection shape: {image_proj.shape}")
    print(f"Text projection shape: {text_proj.shape}")
    print(f"Expected shape: [{batch_size}, 256]")
    
    # 验证输出维度
    assert image_proj.shape == (batch_size, 256), f"Expected image shape ({batch_size}, 256), got {image_proj.shape}"
    assert text_proj.shape == (batch_size, 256), f"Expected text shape ({batch_size}, 256), got {text_proj.shape}"
    
    # 验证归一化
    image_norm = torch.norm(image_proj, p=2, dim=-1)
    text_norm = torch.norm(text_proj, p=2, dim=-1)
    
    print(f"Image projection norms: {image_norm}")
    print(f"Text projection norms: {text_norm}")
    
    # 检查是否接近1（L2归一化后）
    assert torch.allclose(image_norm, torch.ones_like(image_norm), atol=1e-6), "Image projections should be L2 normalized"
    assert torch.allclose(text_norm, torch.ones_like(text_norm), atol=1e-6), "Text projections should be L2 normalized"
    
    # 测试单独编码
    with torch.no_grad():
        image_emb = model.encode_image(images)
        text_emb = model.encode_text(input_ids, attention_mask)
    
    assert torch.equal(image_emb, image_proj), "encode_image should match forward pass"
    assert torch.equal(text_emb, text_proj), "encode_text should match forward pass"
    
    print("✅ CLIPModel test passed!")
    print(f"Embedding dimension: {model.get_embed_dim()}")
    print(f"Image encoder dimension: {model.get_image_dim()}")
    print(f"Text encoder dimension: {model.get_text_dim()}")
    
    return True


if __name__ == "__main__":
    test_clip_model()
