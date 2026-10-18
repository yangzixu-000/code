"""
Transformer 模型（ViT-B/16, Swin Transformer）

所有模型统一规范：
    - 使用 self.backbone 统一变量名（与 CNN 模型对齐）
    - 支持 config 配置
    - 使用统一冻结工具
    - 使用统一 MLPHead 分类头
    - 支持分层学习率 (get_param_groups)
    - 支持特征提取 (get_features)
    - 完整的文档注释
    - 统一的日志输出 (🦋)
    - 兼容 torchvision 低版本
"""

import logging
import torch
import torch.nn as nn
from torchvision import models
from .base import BaseClassifier
from .cnn_models import (
    safe_load_weights,
    apply_freeze_by_modules,
    get_freeze_patterns,
    MLPHead,
    init_weights
)

logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================

def _get_mlp_config(config: dict, default_hidden=None, default_dropout=None, default_activation='gelu'):
    """统一读取 MLP 配置"""
    if default_hidden is None:
        default_hidden = [1024, 512]
    if default_dropout is None:
        default_dropout = [0.3, 0.2]
    return (
        config.get('hidden_dims', default_hidden),
        config.get('dropout_rates', default_dropout),
        config.get('activation', default_activation)
    )


def get_transformer_freeze_patterns(model_name: str, freeze_strategy: str = 'frozen'):
    """
    统一冻结规则获取
    
    模型名称映射：
        - 'ViT-B/16' -> 'ViT'（匹配 cnn_models.py 中的 'ViT' 分支）
        - 'Swin Transformer' -> 'Swin Transformer'（保持原样，匹配 cnn_models.py 中的分支）
    """
    name_map = {
        'ViT-B/16': 'ViT',
        'Swin Transformer': 'Swin Transformer',
    }
    normalized_name = name_map.get(model_name, model_name)
    return get_freeze_patterns(normalized_name, freeze_strategy)


def _handle_freeze(backbone, model_name: str, freeze_strategy: str, logger_obj):
    """统一的冻结处理逻辑"""
    if freeze_strategy in ['frozen', 'partial']:
        freeze_patterns, unfreeze_patterns = get_transformer_freeze_patterns(model_name, freeze_strategy)
        frozen_count, trainable_count = apply_freeze_by_modules(backbone, freeze_patterns, unfreeze_patterns)
        logger_obj.info(
            f"  🔒 {model_name} | freeze={freeze_strategy} | "
            f"frozen={frozen_count} | trainable={trainable_count}"
        )
        return frozen_count, trainable_count
    elif freeze_strategy == 'full':
        logger_obj.info(f"  🔓 {model_name} | freeze=full (no freezing)")
        return 0, sum(p.numel() for p in backbone.parameters())
    else:
        logger_obj.warning(f"  ⚠️ Unknown freeze_strategy '{freeze_strategy}', using full as fallback")
        return 0, sum(p.numel() for p in backbone.parameters())


# ============================================================
# ViT-B/16
# ============================================================
class ViTClassifier(BaseClassifier):
    """
    ViT-B/16 分类器
    
    默认输入尺寸：224×224
    Patch 大小：16×16 → 196 个 patch tokens
    特征维度：768
    参数量：~86.6M
    
    get_features() 返回:
        torch.Tensor: shape (B, 197, 768)
            - [:, 0, :] 是 CLS token（全局特征，用于分类）
            - [:, 1:, :] 是 196 个 patch tokens（空间特征，14×14 网格）
    
    冻结策略：
        - frozen: 冻结 backbone，仅训练分类头
        - partial: 解冻后几层
        - full: 全量训练
    """
    
    def __init__(self, num_classes: int, config: dict = None):
        super().__init__(num_classes)
        config = config or {}
        
        # ===== 配置参数 =====
        freeze_strategy = config.get("freeze_strategy", "frozen")
        hidden_dims, dropout_rates, activation = _get_mlp_config(
            config, [1024, 512], [0.3, 0.2], 'gelu'
        )
        self.lr_scale_backbone = config.get("lr_scale_backbone", 0.1)
        self.lr_scale_head = config.get("lr_scale_head", 1.0)
        
        # ===== 安全加载预训练权重 =====
        self.backbone = safe_load_weights(
            models.vit_b_16,
            models.ViT_B_16_Weights.IMAGENET1K_V1,
            "ViT-B/16"
        )
        
        # ===== 统一冻结逻辑 =====
        _handle_freeze(self.backbone, 'ViT-B/16', freeze_strategy, logger)
        
        # ===== 替换分类头 =====
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = MLPHead(
            in_features, num_classes,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
            activation=activation
        )
        self.backbone.heads.head.apply(init_weights)
        
        # 检测 ViT 版本（是否支持 forward_features）
        self._has_forward_features = hasattr(self.backbone, 'forward_features')
        
        logger.info(
            f"🦋 ViT-B/16 | freeze={freeze_strategy} | "
            f"has_forward_features={self._has_forward_features} | "
            f"lr_scale_backbone={self.lr_scale_backbone}"
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_param_groups(self, lr_scale_backbone=None, lr_scale_head=None):
        """
        获取分层学习率参数组
        
        注意：不过滤 requires_grad=False 的参数，确保优化器始终持有所有参数
        与项目其他模型（ResNet/EfficientNet/MobileNet）保持统一规范
        """
        if lr_scale_backbone is None:
            lr_scale_backbone = self.lr_scale_backbone
        if lr_scale_head is None:
            lr_scale_head = self.lr_scale_head
        
        backbone_params, head_params = [], []
        for name, param in self.backbone.named_parameters():
            # 移除了 if not param.requires_grad: continue
            if "heads" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        # 统计日志
        cnt_backbone = sum(p.numel() for p in backbone_params)
        cnt_head = sum(p.numel() for p in head_params)
        logger.info(
            f"📊 ViT-B/16 param groups: backbone={cnt_backbone:,}(lr_scale={lr_scale_backbone}), "
            f"head={cnt_head:,}(lr_scale={lr_scale_head})"
        )
        
        return [
            {"params": backbone_params, "lr_scale": lr_scale_backbone},
            {"params": head_params, "lr_scale": lr_scale_head}
        ]
    
    def get_features(self, x):
        """
        提取 ViT 的 token 特征
        
        Returns:
            torch.Tensor: (B, 197, 768)
                - CLS token 用于分类
                - Patch tokens 保留空间结构（14×14）
        
        兼容处理：
            - torchvision >= 0.15: 使用 forward_features()
            - torchvision < 0.15: 手动提取 encoder 输出
        """
        if self._has_forward_features:
            return self.backbone.forward_features(x)  # (B, 197, 768)
        else:
            # 手动提取 ViT 特征（兼容低版本 torchvision）
            x = self.backbone._process_input(x)
            x = self.backbone.encoder(x)  # (B, 197, 768)
            return x


# ============================================================
# Swin Transformer
# ============================================================
class SwinTransformerClassifier(BaseClassifier):
    """
    Swin Transformer 分类器
    
    默认输入尺寸：224×224
    特征维度：768
    特征图尺寸：7×7
    参数量：~28.3M
    
    get_features() 返回:
        torch.Tensor: shape (B, 768, 7, 7) 4D 特征图
    
    冻结策略：
        - frozen: 冻结 backbone，仅训练分类头
        - partial: 解冻后几层
        - full: 全量训练
    """
    
    def __init__(self, num_classes: int, config: dict = None):
        super().__init__(num_classes)
        config = config or {}
        
        # ===== 配置参数 =====
        freeze_strategy = config.get("freeze_strategy", "frozen")
        hidden_dims, dropout_rates, activation = _get_mlp_config(
            config, [1024, 512], [0.3, 0.2], 'gelu'
        )
        self.lr_scale_backbone = config.get("lr_scale_backbone", 0.1)
        self.lr_scale_head = config.get("lr_scale_head", 1.0)
        
        # ===== 安全加载预训练权重 =====
        self.backbone = safe_load_weights(
            models.swin_t,
            models.Swin_T_Weights.IMAGENET1K_V1,
            "Swin Transformer"
        )
        
        # ===== 统一冻结逻辑 =====
        _handle_freeze(self.backbone, 'Swin Transformer', freeze_strategy, logger)
        
        # ===== 替换分类头 =====
        in_features = self.backbone.head.in_features
        self.backbone.head = MLPHead(
            in_features, num_classes,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
            activation=activation
        )
        self.backbone.head.apply(init_weights)
        
        # Swin 的 features 属性在 torchvision 0.13+ 中始终存在
        self._has_features = hasattr(self.backbone, 'features')
        
        logger.info(
            f"🦋 Swin Transformer | freeze={freeze_strategy} | "
            f"has_features={self._has_features} | "
            f"lr_scale_backbone={self.lr_scale_backbone}"
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_param_groups(self, lr_scale_backbone=None, lr_scale_head=None):
        """
        获取分层学习率参数组
        
        注意：不过滤 requires_grad=False 的参数，确保优化器始终持有所有参数
        与项目其他模型（ResNet/EfficientNet/MobileNet）保持统一规范
        """
        if lr_scale_backbone is None:
            lr_scale_backbone = self.lr_scale_backbone
        if lr_scale_head is None:
            lr_scale_head = self.lr_scale_head
        
        backbone_params, head_params = [], []
        for name, param in self.backbone.named_parameters():
            # 移除了 if not param.requires_grad: continue
            if "head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        # 统计日志
        cnt_backbone = sum(p.numel() for p in backbone_params)
        cnt_head = sum(p.numel() for p in head_params)
        logger.info(
            f"📊 Swin Transformer param groups: backbone={cnt_backbone:,}(lr_scale={lr_scale_backbone}), "
            f"head={cnt_head:,}(lr_scale={lr_scale_head})"
        )
        
        return [
            {"params": backbone_params, "lr_scale": lr_scale_backbone},
            {"params": head_params, "lr_scale": lr_scale_head}
        ]
    
    def get_features(self, x):
        """
        提取 Swin Transformer 的 4D 特征图
        
        Returns:
            torch.Tensor: (B, 768, 7, 7) 特征图
        
        注意：torchvision 0.13+ 中 self.backbone.features 始终存在
        """
        if self._has_features:
            return self.backbone.features(x)
        else:
            # 极低版本 torchvision 的降级方案
            raise RuntimeError(
                "Swin Transformer get_features requires torchvision >= 0.13. "
                "Current version does not have 'features' attribute."
            )