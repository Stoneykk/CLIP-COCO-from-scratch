"""
计算 R@1、R@5、R@10、InfoNCE 等
实现CLIP训练和评估所需的损失函数和指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


def info_nce_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    InfoNCE (Information Noise Contrastive Estimation) 损失函数
    
    CLIP使用的对比学习损失，最大化正样本对的相似度，最小化负样本对的相似度
    
    Args:
        image_features (torch.Tensor): 归一化的图像特征 [B, D]
        text_features (torch.Tensor): 归一化的文本特征 [B, D]
        temperature (float): 温度参数，控制分布的尖锐程度，默认0.07
        reduction (str): 损失缩减方式，'mean'或'sum'，默认'mean'
        
    Returns:
        torch.Tensor: InfoNCE损失值
    """
    # 确保输入是归一化的
    assert torch.allclose(torch.norm(image_features, p=2, dim=-1), torch.ones_like(torch.norm(image_features, p=2, dim=-1)), atol=1e-6), \
        "Image features should be L2 normalized"
    assert torch.allclose(torch.norm(text_features, p=2, dim=-1), torch.ones_like(torch.norm(text_features, p=2, dim=-1)), atol=1e-6), \
        "Text features should be L2 normalized"
    
    batch_size = image_features.shape[0]
    device = image_features.device
    
    # 计算相似度矩阵
    # image_features @ text_features.T -> [B, B]
    logits_per_image = torch.matmul(image_features, text_features.T) / temperature
    logits_per_text = logits_per_image.T  # [B, B]
    
    # 创建标签：对角线为正样本对
    labels = torch.arange(batch_size, device=device)
    
    # 计算交叉熵损失
    # 图像到文本的损失
    loss_i2t = F.cross_entropy(logits_per_image, labels, reduction=reduction)
    # 文本到图像的损失
    loss_t2i = F.cross_entropy(logits_per_text, labels, reduction=reduction)
    
    # 总损失是两个方向损失的平均
    total_loss = (loss_i2t + loss_t2i) / 2
    
    return total_loss


def compute_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    k_values: Tuple[int, ...] = (1, 5, 10)
) -> dict:
    """
    计算检索指标：Recall@K
    
    Args:
        image_features (torch.Tensor): 归一化的图像特征 [B, D]
        text_features (torch.Tensor): 归一化的文本特征 [B, D]
        k_values (Tuple[int, ...]): 要计算的K值，默认(1, 5, 10)
        
    Returns:
        dict: 包含各种Recall@K指标的字典
    """
    batch_size = image_features.shape[0]
    device = image_features.device
    
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(image_features, text_features.T)  # [B, B]
    
    # 创建标签：对角线为正样本
    labels = torch.arange(batch_size, device=device)
    
    metrics = {}
    
    # 计算图像到文本的Recall@K
    max_k = min(max(k_values), batch_size)
    _, indices_i2t = torch.topk(similarity_matrix, max_k, dim=1)
    for k in k_values:
        if k <= batch_size:
            correct_i2t = (indices_i2t[:, :k] == labels.unsqueeze(1)).any(dim=1)
            recall_i2t = correct_i2t.float().mean().item()
            metrics[f'recall_i2t@{k}'] = recall_i2t
    
    # 计算文本到图像的Recall@K
    _, indices_t2i = torch.topk(similarity_matrix.T, max_k, dim=1)
    for k in k_values:
        if k <= batch_size:
            correct_t2i = (indices_t2i[:, :k] == labels.unsqueeze(1)).any(dim=1)
            recall_t2i = correct_t2i.float().mean().item()
            metrics[f'recall_t2i@{k}'] = recall_t2i
    
    # 计算平均Recall@K
    for k in k_values:
        if k <= batch_size:
            metrics[f'recall@{k}'] = (metrics[f'recall_i2t@{k}'] + metrics[f'recall_t2i@{k}']) / 2
    
    return metrics


def test_info_nce_loss():
    """测试InfoNCE损失函数"""
    print("Testing InfoNCE Loss...")
    
    # 测试1：基本功能测试
    print("\n1. Basic functionality test:")
    batch_size = 4
    feature_dim = 256
    
    # 创建归一化的特征向量
    image_features = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=-1)
    text_features = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=-1)
    
    loss = info_nce_loss(image_features, text_features)
    print(f"InfoNCE loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    
    # 测试2：完美对齐的情况（损失应该很小）
    print("\n2. Perfect alignment test:")
    # 使用相同的特征向量
    image_features = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=-1)
    text_features = image_features.clone()  # 完全相同的特征
    
    loss = info_nce_loss(image_features, text_features)
    print(f"Perfect alignment loss: {loss.item():.4f}")
    assert loss.item() < 0.1, "Perfect alignment should have very low loss"
    
    # 测试3：完全不对齐的情况（损失应该很大）
    print("\n3. No alignment test:")
    # 使用正交的特征向量
    image_features = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=-1)
    text_features = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=-1)
    # 确保它们不相关
    while torch.abs(torch.corrcoef(torch.stack([image_features.flatten(), text_features.flatten()]))[0, 1]) > 0.1:
        text_features = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=-1)
    
    loss = info_nce_loss(image_features, text_features)
    print(f"No alignment loss: {loss.item():.4f}")
    assert loss.item() > 1.0, "No alignment should have high loss"
    
    # 测试4：温度参数的影响
    print("\n4. Temperature parameter test:")
    image_features = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=-1)
    text_features = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=-1)
    
    for temp in [0.01, 0.07, 0.1, 1.0]:
        loss = info_nce_loss(image_features, text_features, temperature=temp)
        print(f"Temperature {temp}: loss = {loss.item():.4f}")
    
    # 测试5：不同batch size
    print("\n5. Different batch sizes test:")
    for bs in [2, 8, 16]:
        img_feat = F.normalize(torch.randn(bs, feature_dim), p=2, dim=-1)
        txt_feat = F.normalize(torch.randn(bs, feature_dim), p=2, dim=-1)
        loss = info_nce_loss(img_feat, txt_feat)
        print(f"Batch size {bs}: loss = {loss.item():.4f}")
    
    print("\n✅ InfoNCE Loss test passed!")
    return True


def test_retrieval_metrics():
    """测试检索指标"""
    print("\nTesting Retrieval Metrics...")
    
    batch_size = 8
    feature_dim = 256
    
    # 创建测试数据
    image_features = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=-1)
    text_features = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=-1)
    
    # 计算指标
    metrics = compute_retrieval_metrics(image_features, text_features)
    
    print("Retrieval metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 验证指标范围
    for key, value in metrics.items():
        assert 0 <= value <= 1, f"{key} should be between 0 and 1, got {value}"
    
    # 测试完美检索的情况
    print("\nPerfect retrieval test:")
    image_features = F.normalize(torch.randn(batch_size, feature_dim), p=2, dim=-1)
    text_features = image_features.clone()  # 完全相同的特征
    
    metrics = compute_retrieval_metrics(image_features, text_features)
    print("Perfect retrieval metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
        if 'recall@1' in key:
            assert value == 1.0, f"Perfect retrieval should have recall@1 = 1.0, got {value}"
    
    print("\n✅ Retrieval Metrics test passed!")
    return True


if __name__ == "__main__":
    test_info_nce_loss()
    test_retrieval_metrics()
