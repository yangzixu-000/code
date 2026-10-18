"""
iFormer-S 分类器

重要说明：
    iFormer-S（Inception Transformer）在 timm 1.0.x 中已被移除（timm.list_models('*iformer*') 为空）。
    本类优先尝试通过 timm 加载官方实现：
        1) timm.create_model('iformer_small', pretrained=True)
        2) timm.create_model('iformer_base',  pretrained=True)
    若当前环境的 timm 版本包含 iformer（例如 timm 0.9.x），则自动可用，无需任何改动。

    若 timm 不包含 iformer：
        - 可在 config 中提供 pretrained_path（指向官方 iformer_small.pth），
          前提是你已自行实现 / 提供与权重结构匹配的 iFormer（见下方 custom_impl 钩子）。
        - 否则会抛出清晰、可操作的 RuntimeError，提示两种解决方案，避免“悄悄用随机权重训练”
          从而污染基准结论。

该模型与另外 4 个新模型共享完全相同的训练协议（224×224 / ImageNet 预训练 / batch_size=32 /
epoch=80 / seed=42 / 全局一致的增强与长尾策略），不做任何针对新模型的调参。
"""

import logging
import torch
import torch.nn as nn
import timm
from .base import BaseClassifier
from .cnn_models import (
    MLPHead,
    init_weights,
    apply_freeze_by_modules,
    get_classifier_in_features,
)

logger = logging.getLogger(__name__)


class IFormerSClassifier(BaseClassifier):
    """
    iFormer-S 分类器（timm 门控加载 + 可选本地权重 + 可选自定义实现）。
    """

    model_label = "iFormer-S"

    def __init__(self, num_classes: int, config: dict = None):
        super().__init__(num_classes)
        config = config or {}

        self.lr_scale_backbone = config.get("lr_scale_backbone", 1.0)
        self.lr_scale_head = config.get("lr_scale_head", 1.0)

        self.backbone = self._build_backbone(config)

        freeze_strategy = config.get("freeze_strategy", "full")
        if freeze_strategy == "frozen":
            apply_freeze_by_modules(self.backbone, [r".*"], [r"head\..*"])
            logger.info(f"📸 {self.model_label} | freeze=frozen")
        elif freeze_strategy == "partial":
            apply_freeze_by_modules(
                self.backbone, [r".*"], [r"head\..*", r"stages\.(2|3)\..*"]
            )
            logger.info(f"📸 {self.model_label} | freeze=partial")
        else:
            logger.info(f"📸 {self.model_label} | freeze=full")

        self._replace_head(num_classes, config)

    def _build_backbone(self, config):
        """按优先级尝试加载 iFormer；timm 不可用时给出可操作错误。"""
        # 1) 尝试官方 timm 实现
        for arch in ("iformer_small", "iformer_base"):
            try:
                return timm.create_model(arch, pretrained=True)
            except Exception as e:  # noqa: BLE001
                logger.info(f"iFormer 通过 timm({arch}) 加载失败: {e}")

        # 2) 尝试自定义实现（用户把与权重匹配的 iFormer 放到 models/iformer_custom.py）
        if config.get("custom_impl"):
            try:
                from .iformer_custom import iFormer_S  # type: ignore
                logger.info("iFormer-S 使用自定义实现 models/iformer_custom.iFormer_S")
                return iFormer_S(num_classes=1000)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"加载自定义 iFormer 失败: {e}")

        # 3) 均失败：抛出清晰、可操作的错误（不悄悄用随机权重）
        raise RuntimeError(
            "❌ iFormer-S 无法加载：当前环境的 timm 不包含 iformer_small/iformer_base"
            "（timm 1.0.x 已移除该模型）。\n"
            "   解决方案二选一：\n"
            "   (A) 在训练环境中使用 timm 0.9.x（含 iformer_small）："
            " pip install timm==0.9.12  ；或\n"
            "   (B) 将官方 iFormer-S 实现与权重放到 models/iformer_custom.py，"
            "并在该模型 config 中设置 custom_impl: true（可选 pretrained_path 指向 .pth）。\n"
            "   本次未做任何随机权重兜底，以免污染基准对比结论。"
        )

    def _replace_head(self, num_classes, config):
        hidden_dims = config.get("hidden_dims", [512, 256])
        dropout_rates = config.get("dropout_rates", [0.2, 0.1])
        activation = config.get("activation", "gelu")

        head = getattr(self.backbone, "head", None)
        if isinstance(head, nn.Linear):
            in_features = head.in_features
        elif hasattr(self.backbone, "get_classifier") and isinstance(
            self.backbone.get_classifier(), nn.Linear
        ):
            in_features = self.backbone.get_classifier().in_features
        else:
            in_features = get_classifier_in_features(self.backbone)

        new_head = MLPHead(
            in_features, num_classes,
            hidden_dims=hidden_dims, dropout_rates=dropout_rates, activation=activation,
        )
        new_head.apply(init_weights)
        self.backbone.head = new_head

        # 可选：加载用户提供的本地权重（如官方 iformer_small.pth）
        pretrained_path = config.get("pretrained_path")
        if pretrained_path:
            try:
                sd = torch.load(pretrained_path, map_location="cpu")
                miss, unexp = self.backbone.load_state_dict(sd, strict=False)
                logger.info(
                    f"📸 {self.model_label} | 已加载本地权重 {pretrained_path} | "
                    f"missing={miss} unexpected={unexp}"
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"⚠️ 加载本地权重 {pretrained_path} 失败: {e}")

    def forward(self, x):
        return self.backbone(x)

    def get_param_groups(self, lr_scale_backbone=None, lr_scale_head=None):
        if lr_scale_backbone is None:
            lr_scale_backbone = self.lr_scale_backbone
        if lr_scale_head is None:
            lr_scale_head = self.lr_scale_head
        backbone_params, head_params = [], []
        for name, param in self.backbone.named_parameters():
            if "head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {"params": backbone_params, "lr_scale": lr_scale_backbone},
            {"params": head_params, "lr_scale": lr_scale_head},
        ]

    def get_features(self, x):
        return self.backbone.forward_features(x)
