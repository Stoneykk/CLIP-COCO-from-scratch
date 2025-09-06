"""
加载和缓存 DistilBERT tokenizer
封装文本编码功能，支持批量处理和自动padding
"""

import torch
from transformers import DistilBertTokenizer
from typing import List, Dict, Union, Optional
import os


class TextTokenizer:
    """
    DistilBERT文本tokenizer封装类
    
    功能：
    - 加载和缓存DistilBERT tokenizer
    - 文本编码：List[str] -> input_ids, attention_mask
    - 自动padding和truncation
    """
    
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased",
        max_length: int = 77,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ):
        """
        初始化tokenizer
        
        Args:
            model_name (str): DistilBERT模型名称
            max_length (int): 最大序列长度，默认77（CLIP标准）
            padding (Union[bool, str]): 是否padding，默认True
            truncation (bool): 是否截断，默认True
            return_tensors (str): 返回张量类型，默认"pt"（PyTorch）
        """
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        
        # 加载tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # 添加特殊token（如果需要）
        self._setup_special_tokens()
        
    def _setup_special_tokens(self):
        """设置特殊token"""
        # DistilBERT已经有CLS和SEP token，通常不需要额外设置
        pass
    
    def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        编码文本列表
        
        Args:
            texts (List[str]): 输入文本列表
            
        Returns:
            Dict[str, torch.Tensor]: 包含input_ids和attention_mask的字典
        """
        # 使用tokenizer编码
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length' if self.padding else False,  # 确保padding到max_length
            truncation=self.truncation,
            return_tensors=self.return_tensors,
            return_attention_mask=True
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def encode_single_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        编码单个文本
        
        Args:
            text (str): 输入文本
            
        Returns:
            Dict[str, torch.Tensor]: 包含input_ids和attention_mask的字典
        """
        return self.encode_texts([text])
    
    def decode_tokens(self, input_ids: torch.Tensor) -> List[str]:
        """
        解码token IDs为文本
        
        Args:
            input_ids (torch.Tensor): token IDs
            
        Returns:
            List[str]: 解码后的文本列表
        """
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return self.tokenizer.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """获取特殊token的ID"""
        return {
            'cls_token_id': self.tokenizer.cls_token_id,
            'sep_token_id': self.tokenizer.sep_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'unk_token_id': self.tokenizer.unk_token_id
        }
    
    def get_max_length(self) -> int:
        """获取最大序列长度"""
        return self.max_length


def test_tokenizer():
    """测试tokenizer功能"""
    print("Testing TextTokenizer...")
    
    # 创建tokenizer
    tokenizer = TextTokenizer(max_length=32)
    
    # 测试文本
    test_texts = [
        "A photo of a cat sitting on a table",
        "A dog running in the park",
        "A beautiful sunset over the ocean",
        "A person riding a bicycle"
    ]
    
    print(f"Input texts: {test_texts}")
    print(f"Number of texts: {len(test_texts)}")
    
    # 编码文本
    encoded = tokenizer.encode_texts(test_texts)
    
    print(f"Input IDs shape: {encoded['input_ids'].shape}")
    print(f"Attention mask shape: {encoded['attention_mask'].shape}")
    print(f"Expected shape: [{len(test_texts)}, 32]")
    
    # 验证输出维度
    batch_size = len(test_texts)
    assert encoded['input_ids'].shape == (batch_size, 32), f"Expected input_ids shape ({batch_size}, 32), got {encoded['input_ids'].shape}"
    assert encoded['attention_mask'].shape == (batch_size, 32), f"Expected attention_mask shape ({batch_size}, 32), got {encoded['attention_mask'].shape}"
    
    # 验证padding
    print(f"Input IDs:\n{encoded['input_ids']}")
    print(f"Attention mask:\n{encoded['attention_mask']}")
    
    # 检查是否有padding（pad_token_id）
    pad_token_id = tokenizer.get_special_tokens()['pad_token_id']
    has_padding = (encoded['input_ids'] == pad_token_id).any()
    print(f"Has padding: {has_padding}")
    
    # 验证attention mask
    # attention mask应该在有实际token的位置为1，padding位置为0
    assert torch.all((encoded['attention_mask'] == 0) | (encoded['attention_mask'] == 1)), "Attention mask should only contain 0s and 1s"
    
    # 测试解码
    decoded = tokenizer.decode_tokens(encoded['input_ids'])
    print(f"Decoded texts: {decoded}")
    
    # 测试单个文本编码
    single_encoded = tokenizer.encode_single_text(test_texts[0])
    assert single_encoded['input_ids'].shape == (1, 32), "Single text encoding should have batch size 1"
    
    # 测试tokenizer信息
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: {tokenizer.get_special_tokens()}")
    print(f"Max length: {tokenizer.get_max_length()}")
    
    print("✅ TextTokenizer test passed!")
    
    return True


if __name__ == "__main__":
    test_tokenizer()
