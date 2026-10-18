"""
基于 timm 的现代视觉模型（InceptionNeXt-Small, ConvNeXt-V2-Tiny, CoAtNet-Tiny, EfficientFormerV2-S）

统一规范（与现有 models/ 下其它模型保持一致）：
    - 使用统一 MLPHead 分类头（替换 timm 原生 head）
    - 使用统一冻结工具 (apply_freeze_by_modules)
    - 支持分层学习率 (get_param_groups)
    - 支持特征提取 (get_features)
    - 支持 ImageNet 预训练权重（safe_load_timm 失败时回退随机初始化，与 cnn_models.safe_load_weights 语义一致）

关键实现说明：
    timm 的 forward_head 对多数架构会把 *4D* 特征 (B,C,H,W) 直接喂给 head（原生 head 内部完成
    全局池化），因此这里用 TimmHeadWrapper 在 MLPHead 之前完成全局平均池化；分类头输入维度
    取自 forward_features 的通道数（而非 get_classifier().in_features，后者对 InceptionNeXt 等
    多尺度拼接头会给出错误维度）。

注意：
    - 这 5 个新模型（连同 iFormer-S）统一使用 224×224 输入、ImageNet 预训练、batch_size=64、
      epoch=80、seed=42，以及全局一致的 data augmentation / 长尾离线增强策略（来自 get_dataloaders）。
    - 不针对新模型做任何调参：freeze_strategy=full、lr_scale_backbone=1.0、hidden_dims=[512,256]、
      dropout_rates=[0.2,0.1]、activation=gelu，与现有 modern_cnns 家族（ConvNeXt-Tiny/MaxViT/RegNetY/EffV2）一致。
"""

import logging
import os
import torch
import torch.nn as nn
import timm
from .base import BaseClassifier
from .cnn_models import (
    MLPHead,
    init_weights,
    apply_freeze_by_modules,
)

logger = logging.getLogger(__name__)


class TimmHeadWrapper(nn.Module):
    """
    适配 timm forward_head：timm 会把 4D 特征 (B,C,H,W) 直接喂给 head，
    原生 head 内部完成全局池化。这里先在 MLPHead 之前做全局平均池化，
    使其兼容任意“先特征后 head”的 timm 架构（InceptionNeXt/ConvNeXt-V2/CoAtNet/EfficientFormerV2）。
    """

    def __init__(self, mlp_head: nn.Module):
        super().__init__()
        self.mlp_head = mlp_head

    def forward(self, x, pre_logits: bool = False, **kwargs):
        if x.dim() == 4:
            x = x.mean(dim=(-2, -1))  # 全局平均池化 (B, C, H, W) -> (B, C)
        if pre_logits:
            return x
        return self.mlp_head(x)


def safe_load_timm(arch: str, model_name: str, num_classes: int,
                   pretrained_path: str = None):
    """
    加载 timm 模型与 ImageNet 预训练骨干权重。
    - 传入 num_classes：timm 会用随机初始化的分类头（维度数 = num_classes），但骨干仍加载 ImageNet 权重；
      这正好符合“骨干用 ImageNet 预训练、分类头重新训练”的基准协议。
    - pretrained_path：给定本地权重文件时跳过在线下载，以随机初始化创建模型（随后由调用方加载本地权重）。
    - 在线下载失败时默认**直接抛错**：基准协议要求 ImageNet 预训练，禁止静默回退随机初始化
      （那会让整轮 80 epoch 训练结果失效）。确要随机初始化必须显式设置环境变量 OBWX_ALLOW_RANDOM_INIT=1。
    """
    if pretrained_path:
        logger.info(f"📸 {model_name} | 检测到本地权重 {pretrained_path}，跳过在线 ImageNet 下载")
        return timm.create_model(arch, pretrained=False, num_classes=num_classes)
    try:
        return timm.create_model(arch, pretrained=True, num_classes=num_classes)
    except Exception as e:  # 无网络 / 权重缺失 / 未知 arch
        if os.environ.get("OBWX_ALLOW_RANDOM_INIT", "").strip() != "1":
            raise RuntimeError(
                f"❌ {model_name} (timm:{arch}) ImageNet 预训练权重加载失败：{e}\n"
                f"   基准协议要求 ImageNet 预训练，已中止训练（不回退随机初始化）。修复方式任选其一：\n"
                f"   (1) 设置 HF 镜像后重试：CMD `set HF_ENDPOINT=https://hf-mirror.com` / "
                f"PowerShell `$env:HF_ENDPOINT='https://hf-mirror.com'` / Linux `export HF_ENDPOINT=...`\n"
                f"   (2) 离线权重：在各模型 config 设 pretrained_path 指向本地 .pth/.safetensors\n"
                f"   (3) 确要随机初始化（将破坏基准协议，结果不可用于论文）：`set OBWX_ALLOW_RANDOM_INIT=1`"
            ) from e
        logger.warning(f"⚠️ 加载 {model_name} (timm:{arch}) ImageNet 权重失败: {e}")
        logger.warning(f"   OBWX_ALLOW_RANDOM_INIT=1 已设置，使用随机初始化继续（结果不具预训练基准效力）。")
        return timm.create_model(arch, pretrained=False, num_classes=num_classes)


def _replace_timm_head(backbone, num_classes, config, head_attr="head"):
    """
    把 timm backbone 的分类头原地替换成 [全局池化 -> MLPHead]。
    in_features 取 forward_features 的通道数（对 4D 输出即 C），保证维度正确。

    head_attr：timm 各架构的分类头属性名并不统一 —— 绝大多数现代架构用 `head`
    （ConvNeXt/CoAtNet/MaxViT/PVT…），而 **ResNet 家族（含 WideResNet / ResNeXt）
    用的是 `fc`**，且根本没有 `head` 属性。若对 ResNet 家族误写 `backbone.head = ...`，
    只会新增一个从未被调用的属性，原生随机初始化的 `fc` 仍参与前向 → 分类头实际未被替换。
    故此处必须按架构传入正确的属性名。
    """
    hidden_dims = config.get("hidden_dims", [512, 256])
    dropout_rates = config.get("dropout_rates", [0.2, 0.1])
    activation = config.get("activation", "gelu")

    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        feats = backbone.forward_features(dummy)
    in_features = feats.shape[1] if feats.dim() >= 2 else feats.shape[-1]

    new_mlp = MLPHead(
        in_features, num_classes,
        hidden_dims=hidden_dims, dropout_rates=dropout_rates, activation=activation,
    )
    new_mlp.apply(init_weights)
    if not hasattr(backbone, head_attr):
        raise AttributeError(
            f"timm backbone {type(backbone).__name__} 没有分类头属性 '{head_attr}'；"
            f"请为该架构设置正确的 TimmClassifier.head_attr。"
        )
    setattr(backbone, head_attr, TimmHeadWrapper(new_mlp))
    return in_features


class TimmClassifier(BaseClassifier):
    """
    timm 模型通用基类。子类只需设置 arch（timm 模型名）与 model_label（论文显示名）。
    """

    arch: str = ""
    model_label: str = ""
    # 分类头属性名：多数 timm 架构为 "head"，ResNet 家族（WideResNet/ResNeXt）为 "fc"
    head_attr: str = "head"

    def __init__(self, num_classes: int, config: dict = None):
        super().__init__(num_classes)
        config = config or {}

        self.lr_scale_backbone = config.get("lr_scale_backbone", 1.0)
        self.lr_scale_head = config.get("lr_scale_head", 1.0)

        self.backbone = safe_load_timm(self.arch, self.model_label, num_classes,
                                       pretrained_path=config.get("pretrained_path"))

        head_pat = rf"{self.head_attr}\..*"
        freeze_strategy = config.get("freeze_strategy", "full")
        if freeze_strategy == "frozen":
            # 冻结全部，仅保留分类头可训练（timm head 参数名形如 head.mlp_head.* / head_dist.*）
            apply_freeze_by_modules(self.backbone, [r".*"], [head_pat, r"head_dist\..*"])
            logger.info(f"📸 {self.model_label} | freeze=frozen | backbone frozen, {self.head_attr} trainable")
        elif freeze_strategy == "partial":
            # 冻结全部，解冻分类头与最后两个 stage（如 backbone 采用 stage 结构）
            apply_freeze_by_modules(
                self.backbone,
                [r".*"],
                [head_pat, r"head_dist\..*", r"stages\.(2|3)\..*", r"blocks\.(8|9|10|11)\..*"],
            )
            logger.info(f"📸 {self.model_label} | freeze=partial")
        else:  # full
            logger.info(f"📸 {self.model_label} | freeze=full (全部参数参与训练)")

        _replace_timm_head(self.backbone, num_classes, config, head_attr=self.head_attr)

        # 可选：从本地权重文件加载（覆盖 ImageNet 权重，例如官方 iFormer/自定义权重）
        pretrained_path = config.get("pretrained_path")
        if pretrained_path:
            try:
                sd = torch.load(pretrained_path, map_location="cpu")
                miss, unexp = self.backbone.load_state_dict(sd, strict=False)
                logger.info(f"📸 {self.model_label} | 已加载本地权重 {pretrained_path} | missing={miss} unexpected={unexp}")
            except Exception as e:
                logger.warning(f"⚠️ 加载本地权重 {pretrained_path} 失败: {e}")

    def forward(self, x):
        return self.backbone(x)

    def get_param_groups(self, lr_scale_backbone=None, lr_scale_head=None):
        if lr_scale_backbone is None:
            lr_scale_backbone = self.lr_scale_backbone
        if lr_scale_head is None:
            lr_scale_head = self.lr_scale_head
        backbone_params, head_params = [], []
        prefix = self.head_attr + "."
        for name, param in self.backbone.named_parameters():
            if name == self.head_attr or name.startswith(prefix):
                head_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {"params": backbone_params, "lr_scale": lr_scale_backbone},
            {"params": head_params, "lr_scale": lr_scale_head},
        ]

    def get_features(self, x):
        return self.backbone.forward_features(x)


class InceptionNeXtSmallClassifier(TimmClassifier):
    """InceptionNeXt-Small 分类器 (timm: inception_next_small)"""
    arch = "inception_next_small"
    model_label = "InceptionNeXt-Small"


class ConvNeXtV2TinyClassifier(TimmClassifier):
    """ConvNeXt-V2-Tiny 分类器 (timm: convnextv2_tiny)"""
    arch = "convnextv2_tiny"
    model_label = "ConvNeXt-V2-Tiny"


class CoAtNetTinyClassifier(TimmClassifier):
    """CoAtNet-Tiny 分类器 (timm: coatnet_0_224，CoAtNet 最小变体即 Tiny)"""
    arch = "coatnet_0_224"
    model_label = "CoAtNet-Tiny"


class EfficientFormerV2SClassifier(TimmClassifier):
    """EfficientFormerV2-S 分类器 (timm: efficientformerv2_s2，论文中的 S 变体)"""
    arch = "efficientformerv2_s2"
    model_label = "EfficientFormerV2-S"


class MobileViTV2SClassifier(TimmClassifier):
    """MobileViT-V2-S 分类器 (timm: mobilevitv2_100)

    注：MobileViT-V2 官方按宽度 0.5/0.75/1.0… 命名，并无 “S” 变体；
        经确认以 1.0（7.80M）作为 “S” 的映射（与基准中 S 级模型量级一致）。
        forward_features 通道数 = 512（head_in）。
    """
    arch = "mobilevitv2_100"
    model_label = "MobileViT-V2-S"


class SwinV2TinyClassifier(TimmClassifier):
    """Swin-V2-Tiny 分类器 (timm: swinv2_cr_tiny_224)

    注：swinv2_tiny_window{8,16}_256 在 224 输入下会触发
        “Input height (224) doesn't match model (256)” 断言（timm 强制），
        故统一 224×224 协议下必须使用 224 原生 checkpoint。
        forward_features 通道数 = 768（head_in）。
    """
    arch = "swinv2_cr_tiny_224"
    model_label = "Swin-V2-Tiny"


class RepViTM11Classifier(TimmClassifier):
    """RepViT-M1.1 分类器 (timm: repvit_m1_1，forward_features 通道数 = 512）"""
    arch = "repvit_m1_1"
    model_label = "RepViT-M1.1"


class EdgeNeXtSmallRWClassifier(TimmClassifier):
    """EdgeNeXt-Small-RW 分类器 (timm: edgenext_small_rw)

    注：用户原写 edgnext_small_rw，timm 实际前缀为 edgenext（无第二个 g 后少一个 n 的拼写），
        修正为 edgenext_small_rw（10.00M，224 原生，forward_features 通道数 = 384）。
        “_rw” 为 Ross Wightman 自训练权重变体，与普通 edgenext_small 权重来源不同。
    """
    arch = "edgenext_small_rw"
    model_label = "EdgeNeXt-Small-RW"


class PVTv2B2Classifier(TimmClassifier):
    """PVTv2-B2 分类器 (timm: pvt_v2_b2)

    用途：替代用户原计划的 CMT-Small——CMT 从未被 timm 收录（0.9.2 / 0.9.12 /
          1.0.29 / main 四个版本源码目录均无 cmt 实现），故改选同层级替代。
          PVTv2-B2 为金字塔结构 + 卷积下采样 + SR 注意力，多尺度特性与 CMT 最接近。
          224 原生，forward_features 通道数 = 512。
    """
    arch = "pvt_v2_b2"
    model_label = "PVTv2-B2"


class WideResNet50Classifier(TimmClassifier):
    """WideResNet-50-2 分类器 (timm: wide_resnet50_2)

    补齐论文 14 基线中缺失的 WideResNet-50。ResNet 家族在 timm 中分类头属性为 `fc`
    （无 `head`），故 head_attr="fc"。224 原生，forward_features 通道数 = 2048。

    参数量校验：backbone 66.83M + 协议 MLPHead(2048→512→256→6650) 2.89M = 69.73M，
    与论文报告 model_metrics_14.json 的 params_M=69.73 **完全一致**（差 0.00M），
    可确认该 wrapper 与论文原始实现等价。
    """
    arch = "wide_resnet50_2"
    model_label = "WideResNet-50"
    head_attr = "fc"


class ResNeXt50Classifier(TimmClassifier):
    """ResNeXt-50 (32×4d) 分类器 (timm: resnext50_32x4d)

    补齐论文 14 基线中缺失的 ResNeXt-50。同属 ResNet 家族 → head_attr="fc"。
    224 原生，forward_features 通道数 = 2048。

    参数量校验：backbone 22.98M + 协议 MLPHead 2.89M = 25.87M，
    与论文报告的 params_M=25.87 **完全一致**（差 0.00M）。
    """
    arch = "resnext50_32x4d"
    model_label = "ResNeXt-50"
    head_attr = "fc"
