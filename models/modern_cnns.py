"""
现代 CNN 模型（ConvNeXt-Tiny, EfficientNetV2-S, MaxViT-T, RegNetY-4GF）

所有模型统一规范：
    - 支持 config 配置
    - 使用统一冻结工具 (apply_freeze_by_modules)
    - 使用统一 MLPHead 分类头
    - 支持分层学习率 (get_param_groups)
    - 支持特征提取 (get_features)
    - 完整的文档注释
    - 统一的日志输出
"""

import logging
import torch.nn as nn
from torchvision import models
from .base import BaseClassifier
from .cnn_models import (
    safe_load_weights,
    apply_freeze_by_modules,
    get_freeze_patterns,
    MLPHead,
    init_weights,
    get_classifier_in_features,
    replace_last_linear_with_mlp,
)

logger = logging.getLogger(__name__)


def get_model_name_for_freeze(model_name: str) -> str:
    """
    获取冻结规则所需的模型名称映射
    """
    name_map = {
        'ConvNeXt-Tiny': 'ConvNeXtTiny',
        'EfficientNetV2-S': 'EfficientNetV2S',
        'MaxViT-T': 'MaxViT',
        'RegNetY-4GF': 'RegNetY',
        'RegNetY-400MF': 'RegNetY',
    }
    return name_map.get(model_name, model_name)


# ============================================================
# ConvNeXt-Tiny
# ============================================================
class ConvNeXtTinyClassifier(BaseClassifier):
    """
    ConvNeXt-Tiny 分类器
    """
    def __init__(self, num_classes: int, config: dict = None):
        super().__init__(num_classes)
        config = config or {}

        freeze_strategy = config.get("freeze_strategy", "frozen")
        hidden_dims = config.get("hidden_dims", [512, 256])
        dropout_rates = config.get("dropout_rates", [0.2, 0.1])
        activation = config.get("activation", "gelu")
        self.lr_scale_backbone = config.get("lr_scale_backbone", 0.1)
        self.lr_scale_head = config.get("lr_scale_head", 1.0)

        self.backbone = safe_load_weights(
            models.convnext_tiny,
            models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
            "ConvNeXt-Tiny"
        )

        model_key = get_model_name_for_freeze('ConvNeXt-Tiny')
        if freeze_strategy in ['frozen', 'partial']:
            freeze_patterns, unfreeze_patterns = get_freeze_patterns(model_key, freeze_strategy)
            frozen_count, trainable_count = apply_freeze_by_modules(
                self.backbone, freeze_patterns, unfreeze_patterns
            )
            logger.info(f"📸 ConvNeXt-Tiny | freeze={freeze_strategy} | frozen={frozen_count} | trainable={trainable_count}")
        elif freeze_strategy == 'full':
            logger.info(f"📸 ConvNeXt-Tiny | freeze=full")

        # 替换分类头（ConvNeXt 的 classifier 通常是 Sequential，用通用替换）
        replace_last_linear_with_mlp(
            self.backbone.classifier, num_classes,
            hidden_dims=hidden_dims, dropout_rates=dropout_rates, activation=activation,
        )

    def forward(self, x):
        return self.backbone(x)

    def get_param_groups(self, lr_scale_backbone=None, lr_scale_head=None):
        if lr_scale_backbone is None:
            lr_scale_backbone = self.lr_scale_backbone
        if lr_scale_head is None:
            lr_scale_head = self.lr_scale_head
        backbone_params, head_params = [], []
        for name, param in self.backbone.named_parameters():
            if "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {"params": backbone_params, "lr_scale": lr_scale_backbone},
            {"params": head_params, "lr_scale": lr_scale_head},
        ]

    def get_features(self, x):
        return self.backbone.features(x)


# ============================================================
# EfficientNetV2-S
# ============================================================
class EfficientNetV2SClassifier(BaseClassifier):
    """
    EfficientNetV2-S 分类器
    """
    def __init__(self, num_classes: int, config: dict = None):
        super().__init__(num_classes)
        config = config or {}

        freeze_strategy = config.get("freeze_strategy", "frozen")
        hidden_dims = config.get("hidden_dims", [512, 256])
        dropout_rates = config.get("dropout_rates", [0.2, 0.1])
        activation = config.get("activation", "gelu")
        self.lr_scale_backbone = config.get("lr_scale_backbone", 0.1)
        self.lr_scale_head = config.get("lr_scale_head", 1.0)

        self.backbone = safe_load_weights(
            models.efficientnet_v2_s,
            models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
            "EfficientNetV2-S"
        )

        model_key = get_model_name_for_freeze('EfficientNetV2-S')
        if freeze_strategy in ['frozen', 'partial']:
            freeze_patterns, unfreeze_patterns = get_freeze_patterns(model_key, freeze_strategy)
            frozen_count, trainable_count = apply_freeze_by_modules(
                self.backbone, freeze_patterns, unfreeze_patterns
            )
            logger.info(f"📸 EfficientNetV2-S | freeze={freeze_strategy} | frozen={frozen_count} | trainable={trainable_count}")
        elif freeze_strategy == 'full':
            logger.info(f"📸 EfficientNetV2-S | freeze=full")

        # 替换分类头（EfficientNetV2-S 的 classifier 通常是 Sequential(Dropout, Linear)）
        replace_last_linear_with_mlp(
            self.backbone.classifier, num_classes,
            hidden_dims=hidden_dims, dropout_rates=dropout_rates, activation=activation,
        )

    def forward(self, x):
        return self.backbone(x)

    def get_param_groups(self, lr_scale_backbone=None, lr_scale_head=None):
        if lr_scale_backbone is None:
            lr_scale_backbone = self.lr_scale_backbone
        if lr_scale_head is None:
            lr_scale_head = self.lr_scale_head
        backbone_params, head_params = [], []
        for name, param in self.backbone.named_parameters():
            if "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {"params": backbone_params, "lr_scale": lr_scale_backbone},
            {"params": head_params, "lr_scale": lr_scale_head},
        ]

    def get_features(self, x):
        return self.backbone.features(x)


# ============================================================
# MaxViT-T（已修复分类头替换）
# ============================================================
class MaxViTClassifier(BaseClassifier):
    """
    MaxViT-T 分类器
    """
    def __init__(self, num_classes: int, config: dict = None):
        super().__init__(num_classes)
        config = config or {}

        freeze_strategy = config.get("freeze_strategy", "frozen")
        hidden_dims = config.get("hidden_dims", [512, 256])
        dropout_rates = config.get("dropout_rates", [0.2, 0.1])
        activation = config.get("activation", "gelu")
        self.lr_scale_backbone = config.get("lr_scale_backbone", 0.1)
        self.lr_scale_head = config.get("lr_scale_head", 1.0)

        self.backbone = safe_load_weights(
            models.maxvit_t,
            models.MaxVit_T_Weights.IMAGENET1K_V1,
            "MaxViT-T"
        )

        model_key = get_model_name_for_freeze('MaxViT-T')
        if freeze_strategy in ['frozen', 'partial']:
            freeze_patterns, unfreeze_patterns = get_freeze_patterns(model_key, freeze_strategy)
            frozen_count, trainable_count = apply_freeze_by_modules(
                self.backbone, freeze_patterns, unfreeze_patterns
            )
            logger.info(f"📸 MaxViT-T | freeze={freeze_strategy} | frozen={frozen_count} | trainable={trainable_count}")
        elif freeze_strategy == 'full':
            logger.info(f"📸 MaxViT-T | freeze=full")

        # ===== 关键修复：只替换最后一层 Linear，保留池化/展平/归一化/Tanh =====
        # 原生 MaxViT 的 classifier 结构：
        #   AdaptiveAvgPool2d -> Flatten -> LayerNorm -> Linear -> Tanh -> Linear
        # 我们必须保留前 5 层，只把最后的 Linear 替换成 MLPHead。
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = MLPHead(
            in_features, num_classes,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
            activation=activation
        )
        self.backbone.classifier[-1].apply(init_weights)

    def forward(self, x):
        return self.backbone(x)

    def get_param_groups(self, lr_scale_backbone=None, lr_scale_head=None):
        if lr_scale_backbone is None:
            lr_scale_backbone = self.lr_scale_backbone
        if lr_scale_head is None:
            lr_scale_head = self.lr_scale_head
        backbone_params, head_params = [], []
        for name, param in self.backbone.named_parameters():
            if "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {"params": backbone_params, "lr_scale": lr_scale_backbone},
            {"params": head_params, "lr_scale": lr_scale_head},
        ]

    def get_features(self, x):
        return self.backbone.features(x)


# ============================================================
# RegNetY-4GF
# ============================================================
class RegNetY4GFClassifier(BaseClassifier):
    """
    RegNetY-4GF 分类器
    """
    def __init__(self, num_classes: int, config: dict = None):
        super().__init__(num_classes)
        config = config or {}

        freeze_strategy = config.get("freeze_strategy", "frozen")
        hidden_dims = config.get("hidden_dims", [512, 256])
        dropout_rates = config.get("dropout_rates", [0.2, 0.1])
        activation = config.get("activation", "gelu")
        self.lr_scale_backbone = config.get("lr_scale_backbone", 0.1)
        self.lr_scale_head = config.get("lr_scale_head", 1.0)

        # RegNetY-4GF 可能在某些版本不可用，自动降级到 400MF
        try:
            self.backbone = safe_load_weights(
                models.regnet_y_4gf,
                models.RegNet_Y_4GF_Weights.IMAGENET1K_V1,
                "RegNetY-4GF"
            )
            self._actual_model_name = "RegNetY-4GF"
        except AttributeError:
            logger.warning("⚠️ RegNetY-4GF not available, falling back to RegNetY-400MF.")
            self.backbone = safe_load_weights(
                models.regnet_y_400mf,
                models.RegNet_Y_400MF_Weights.IMAGENET1K_V1,
                "RegNetY-400MF"
            )
            self._actual_model_name = "RegNetY-400MF"

        model_key = get_model_name_for_freeze(self._actual_model_name)
        if freeze_strategy in ['frozen', 'partial']:
            freeze_patterns, unfreeze_patterns = get_freeze_patterns(model_key, freeze_strategy)
            frozen_count, trainable_count = apply_freeze_by_modules(
                self.backbone, freeze_patterns, unfreeze_patterns
            )
            logger.info(f"📸 {self._actual_model_name} | freeze={freeze_strategy} | frozen={frozen_count} | trainable={trainable_count}")
        elif freeze_strategy == 'full':
            logger.info(f"📸 {self._actual_model_name} | freeze=full")

        # 替换 fc 层（RegNet 通常是 Linear）
        new_head, _ = replace_last_linear_with_mlp(
            self.backbone.fc, num_classes,
            hidden_dims=hidden_dims, dropout_rates=dropout_rates, activation=activation,
        )
        if isinstance(self.backbone.fc, nn.Linear):
            self.backbone.fc = new_head

    def forward(self, x):
        return self.backbone(x)

    def get_param_groups(self, lr_scale_backbone=None, lr_scale_head=None):
        if lr_scale_backbone is None:
            lr_scale_backbone = self.lr_scale_backbone
        if lr_scale_head is None:
            lr_scale_head = self.lr_scale_head
        backbone_params, head_params = [], []
        for name, param in self.backbone.named_parameters():
            if "fc" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {"params": backbone_params, "lr_scale": lr_scale_backbone},
            {"params": head_params, "lr_scale": lr_scale_head},
        ]

    def get_features(self, x):
        feat = self.backbone.stem(x)
        feat = self.backbone.trunk(feat)
        return feat