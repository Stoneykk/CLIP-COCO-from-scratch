"""
DistilBERT 文本编码器模块（冻结）
加载预训练的DistilBERT，冻结所有参数，输出768维CLS向量
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from typing import Optional, Dict, Any


class TextEncoder(nn.Module):
    """
    DistilBERT文本编码器（冻结参数）
    
    输入: input_ids, attention_mask
    输出: [B, 768] - 文本特征向量（CLS token）
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", freeze: bool = True):
        """
        初始化文本编码器
        
        Args:
            model_name (str): DistilBERT模型名称，默认"distilbert-base-uncased"
            freeze (bool): 是否冻结模型参数，默认True
        """
        super(TextEncoder, self).__init__()
        
        # 加载预训练的DistilBERT模型
        self.bert = DistilBertModel.from_pretrained(model_name)
        
        # 冻结所有参数
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 输出特征维度（DistilBERT的隐藏层维度）
        self.feature_dim = self.bert.config.dim  # 768
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids (torch.Tensor): 输入token IDs [B, seq_len]
            attention_mask (torch.Tensor): 注意力掩码 [B, seq_len]
            
        Returns:
            torch.Tensor: 文本特征向量 [B, 768] (CLS token)
        """
        # 通过DistilBERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 获取CLS token的表示（第一个token）
        # last_hidden_state shape: [B, seq_len, 768]
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        
        return cls_embeddings
    
    def get_feature_dim(self) -> int:
        """返回特征维度"""
        return self.feature_dim
    
    def is_frozen(self) -> bool:
        """检查模型参数是否已冻结"""
        return not any(param.requires_grad for param in self.bert.parameters())


def test_text_encoder():
    """测试文本编码器"""
    print("Testing TextEncoder...")
    
    # 创建模型
    encoder = TextEncoder(freeze=True)
    
    # 创建测试输入
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = encoder(input_ids, attention_mask)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [{batch_size}, 768]")
    
    # 验证输出维度
    assert output.shape == (batch_size, 768), f"Expected shape ({batch_size}, 768), got {output.shape}"
    
    # 验证参数是否冻结
    is_frozen = encoder.is_frozen()
    print(f"Model parameters frozen: {is_frozen}")
    assert is_frozen, "Model parameters should be frozen"
    
    print("✅ TextEncoder test passed!")
    print(f"Feature dimension: {encoder.get_feature_dim()}")
    
    return True


if __name__ == "__main__":
    test_text_encoder()
