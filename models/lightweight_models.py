"""
轻量级模型（MobileNetV3-Large, ShuffleNetV2）
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

class MobileNetV3LargeClassifier(BaseClassifier):
    def __init__(self, num_classes: int, config: dict = None):
        super().__init__(num_classes)
        config = config or {}
        freeze_strategy = config.get("freeze_strategy", "frozen")
        hidden_dims = config.get("hidden_dims", [256])
        dropout_rates = config.get("dropout_rates", [0.2])
        activation = config.get("activation", "relu")

        self.backbone = safe_load_weights(
            models.mobilenet_v3_large,
            models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
            "MobileNetV3-Large"
        )

        freeze_patterns, unfreeze_patterns = get_freeze_patterns("MobileNetV3Large", freeze_strategy)
        frozen_count, trainable_count = apply_freeze_by_modules(
            self.backbone, freeze_patterns, unfreeze_patterns
        )

        in_feat = self.backbone.classifier[0].in_features  # 960
        self.backbone.classifier = MLPHead(
            in_feat, num_classes,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
            activation=activation
        )

        logger.info(f"📦 MobileNetV3-Large init done | freeze={freeze_strategy}")

    def forward(self, x):
        return self.backbone(x)

    def get_param_groups(self, lr_scale_backbone=0.1, lr_scale_head=1.0):
        backbone_params, head_params = [], []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {"params": backbone_params, "lr_scale": lr_scale_backbone},
            {"params": head_params, "lr_scale": lr_scale_head}
        ]

    def get_features(self, x):
        return self.backbone.features(x)


class ShuffleNetV2Classifier(BaseClassifier):
    def __init__(self, num_classes: int, config: dict = None):
        super().__init__(num_classes)
        config = config or {}
        freeze_strategy = config.get("freeze_strategy", "frozen")
        hidden_dims = config.get("hidden_dims", [256])
        dropout_rates = config.get("dropout_rates", [0.2])
        activation = config.get("activation", "relu")

        self.backbone = safe_load_weights(
            models.shufflenet_v2_x1_0,
            models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
            "ShuffleNetV2-x1.0"
        )

        freeze_patterns, unfreeze_patterns = get_freeze_patterns("ShuffleNetV2", freeze_strategy)
        frozen_count, trainable_count = apply_freeze_by_modules(
            self.backbone, freeze_patterns, unfreeze_patterns
        )

        in_feat = self.backbone.fc.in_features
        self.backbone.fc = MLPHead(
            in_feat, num_classes,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
            activation=activation
        )

        logger.info(f"📦 ShuffleNetV2-x1.0 init done | freeze={freeze_strategy}")

    def forward(self, x):
        return self.backbone(x)

    def get_param_groups(self, lr_scale_backbone=0.1, lr_scale_head=1.0):
        backbone_params, head_params = [], []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if "fc" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {"params": backbone_params, "lr_scale": lr_scale_backbone},
            {"params": head_params, "lr_scale": lr_scale_head}
        ]

    def get_features(self, x):
        feat = self.backbone.conv1(x)
        feat = self.backbone.maxpool(feat)
        feat = self.backbone.stage2(feat)
        feat = self.backbone.stage3(feat)
        feat = self.backbone.stage4(feat)
        feat = self.backbone.conv5(feat)
        return feat