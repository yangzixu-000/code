"""
CNN 模型定义（ResNet18, EfficientNet-B3, DenseNet121）
所有模型统一使用 apply_freeze_by_modules 进行冻结
支持分层学习率配置
"""

import re
import logging
import torch
import torch.nn as nn
from torchvision import models
from .base import BaseClassifier

logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================

def safe_load_weights(model_builder, weight_enum, model_name):
    """安全加载预训练权重，失败时返回随机初始化"""
    try:
        return model_builder(weights=weight_enum)
    except Exception as e:
        logger.warning(f"⚠️ 加载 {model_name} 预训练权重失败: {e}")
        logger.warning(f"   将使用随机初始化权重继续训练。")
        return model_builder(weights=None)


def apply_freeze_by_modules(model, freeze_patterns, unfreeze_patterns=None):
    """
    按模块名冻结/解冻参数
    """
    if unfreeze_patterns is None:
        unfreeze_patterns = []

    freeze_compiled = [re.compile(p) for p in freeze_patterns]
    unfreeze_compiled = [re.compile(p) for p in unfreeze_patterns]

    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        should_freeze = any(p.search(name) for p in freeze_compiled)
        should_unfreeze = any(p.search(name) for p in unfreeze_compiled)

        if should_freeze and not should_unfreeze:
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True
            trainable_count += 1

    logger.info(f"  🔒 Frozen: {frozen_count} params, Trainable: {trainable_count} params")
    return frozen_count, trainable_count


def get_freeze_patterns(model_name: str, freeze_strategy: str = 'frozen'):
    """
    根据模型名和冻结策略返回冻结规则
    """
    if freeze_strategy == 'full':
        return [], []

    if freeze_strategy == 'partial':
        if model_name == 'MobileNetV3Large':
            return ([r'features\.(0|1|2|3|4|5|6|7)\..*'],
                    [r'features\.(8|9|10|11|12|13|14|15|16)\..*', r'classifier\..*'])
        elif model_name == 'ShuffleNetV2':
            return ([r'stage2\..*'],
                    [r'stage3\..*', r'stage4\..*', r'conv5\..*', r'fc\..*'])

    if model_name in ['ResNet18', 'ResNet50']:
        return ([r'.*'], [r'layer4\..*', r'fc\..*'])

    elif model_name == 'EfficientNet-B3':
        return ([r'.*'], [r'features\.(10|11|12)\..*', r'classifier\..*'])

    elif model_name == 'DenseNet121':
        return ([r'features\..*'], [r'features\.denseblock4\..*', r'features\.transition3\..*', r'classifier\..*'])

    elif model_name == 'MobileNetV3Large':
        return ([r'features\..*'], [r'features\.16\..*', r'classifier\..*'])

    elif model_name == 'ShuffleNetV2':
        return ([r'.*'], [r'stage4\..*', r'conv5\..*', r'fc\..*'])

    elif model_name in ['ViT-B/16', 'ViT']:
        return (
            [r'encoder\.layers\.encoder_layer_\d+\..*'],
            [r'encoder\.layers\.encoder_layer_(6|7|8|9|10|11)\..*', r'heads\..*']
        )

    elif model_name == 'Swin Transformer':
        return (
            [r'features\..*'],
            [r'features\.(4|5|6|7)\..*', r'head\..*']
        )

    else:
        return ([r'.*'], [r'classifier\..*', r'fc\..*'])


def init_weights(m):
    """He/Kaiming 初始化"""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def _find_first_linear(module):
    """DFS 查找模块树中第一个 nn.Linear（用于提取分类头输入维度）。"""
    if isinstance(module, nn.Linear):
        return module
    for child in module.children():
        hit = _find_first_linear(child)
        if hit is not None:
            return hit
    return None


def get_classifier_in_features(backbone, hints=None):
    """
    鲁棒地从预训练 backbone 中提取「分类头线性层的输入维度」。
    """
    if hints is None:
        hints = ['fc', 'classifier', 'head', 'heads.head']

    for path in hints:
        cur = backbone
        ok = True
        for attr in path.split('.'):
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if not ok:
            continue
        linear = _find_first_linear(cur)
        if linear is not None:
            return linear.in_features

    last_linear = None
    for m in backbone.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        return last_linear.in_features

    raise AttributeError(
        "Cannot infer classifier in_features from backbone. "
        f"Tried attr paths: {hints}, and also scanned all nn.Linear children. "
        f"Please specify in_features explicitly for this backbone."
    )


def _find_last_linear(module, _container=None):
    """
    返回 (父容器或 module 自身, attr_or_index, Linear 实例)，以便原地替换。
    """
    last_info = None

    if isinstance(module, nn.Sequential):
        for idx, child in enumerate(module):
            if isinstance(child, nn.Linear):
                last_info = (module, idx, child)
            sub = _find_last_linear(child)
            if sub is not None:
                last_info = sub
        return last_info

    if isinstance(module, nn.Module) and not isinstance(module, nn.Linear):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                last_info = (module, name, child)
            sub = _find_last_linear(child)
            if sub is not None:
                last_info = sub
    return last_info


def replace_last_linear_with_mlp(
    head_module, num_classes, hidden_dims, dropout_rates, activation='relu'
):
    """
    在一个「分类头子模块」内部定位最后一个 nn.Linear，把它**原地替换**成 MLPHead。
    """
    # Case 1: head 自己就是 Linear
    if isinstance(head_module, nn.Linear):
        in_features = head_module.in_features
        new_mlp = MLPHead(
            in_features, num_classes,
            hidden_dims=hidden_dims, dropout_rates=dropout_rates, activation=activation,
        )
        new_mlp.apply(init_weights)
        return new_mlp, in_features

    # Case 2: head 是 Sequential / 容器模块
    info = _find_last_linear(head_module)
    if info is None:
        raise AttributeError(
            "Cannot find nn.Linear inside head_module to replace. "
            f"Module type: {type(head_module).__name__}."
        )
    container, key, linear_layer = info
    in_features = linear_layer.in_features
    new_mlp = MLPHead(
        in_features, num_classes,
        hidden_dims=hidden_dims, dropout_rates=dropout_rates, activation=activation,
    )
    new_mlp.apply(init_weights)
    if isinstance(key, int):
        container[key] = new_mlp
    else:
        setattr(container, key, new_mlp)
    return head_module, in_features


class MLPHead(nn.Module):
    """
    通用 MLP 分类头
    """
    def __init__(self, in_features, num_classes, hidden_dims=None, dropout_rates=None, activation='relu'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512]
        if dropout_rates is None:
            dropout_rates = [0.4, 0.2]

        act_map = {
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True),
        }
        act_fn = act_map.get(activation, nn.ReLU(inplace=True))
        assert len(hidden_dims) == len(dropout_rates)

        layers = []
        prev_dim = in_features
        for h_dim, d_rate in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn)
            if d_rate > 0:
                layers.append(nn.Dropout(d_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        return self.mlp(x)


# ============================================================
# 模型定义
# ============================================================

class ResNet18Classifier(BaseClassifier):
    """ResNet18 分类器"""
    def __init__(self, num_classes: int, config: dict = None):
        super().__init__(num_classes)
        config = config or {}
        self.lr_scale_backbone = config.get('lr_scale_backbone', 0.1)
        self.lr_scale_head = config.get('lr_scale_head', 1.0)

        self.backbone = safe_load_weights(
            models.resnet18,
            models.ResNet18_Weights.IMAGENET1K_V1,
            "ResNet18"
        )
        freeze_strategy = config.get('freeze_strategy', 'frozen')
        freeze_patterns, unfreeze_patterns = get_freeze_patterns('ResNet18', freeze_strategy)
        apply_freeze_by_modules(self.backbone, freeze_patterns, unfreeze_patterns)

        hidden_dims = config.get('hidden_dims', [512])
        dropout_rates = config.get('dropout_rates', [0.2])
        activation = config.get('activation', 'relu')

        # 替换 fc
        in_features = self.backbone.fc.in_features
        self.backbone.fc = MLPHead(in_features, num_classes, hidden_dims, dropout_rates, activation)

    def forward(self, x):
        return self.backbone(x)

    def get_param_groups(self, lr_scale_backbone=None, lr_scale_head=None):
        if lr_scale_backbone is None:
            lr_scale_backbone = self.lr_scale_backbone
        if lr_scale_head is None:
            lr_scale_head = self.lr_scale_head
        backbone_params, head_params = [], []
        for name, param in self.backbone.named_parameters():
            if 'fc' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {'params': backbone_params, 'lr_scale': lr_scale_backbone},
            {'params': head_params, 'lr_scale': lr_scale_head},
        ]


class EfficientNetB3Classifier(BaseClassifier):
    """EfficientNet-B3 分类器"""
    def __init__(self, num_classes: int, config: dict = None):
        super().__init__(num_classes)
        config = config or {}
        self.lr_scale_backbone = config.get('lr_scale_backbone', 0.1)
        self.lr_scale_head = config.get('lr_scale_head', 1.0)

        self.backbone = safe_load_weights(
            models.efficientnet_b3,
            models.EfficientNet_B3_Weights.IMAGENET1K_V1,
            "EfficientNet-B3"
        )
        freeze_strategy = config.get('freeze_strategy', 'frozen')
        freeze_patterns, unfreeze_patterns = get_freeze_patterns('EfficientNet-B3', freeze_strategy)
        apply_freeze_by_modules(self.backbone, freeze_patterns, unfreeze_patterns)

        hidden_dims = config.get('hidden_dims', [512, 256])
        dropout_rates = config.get('dropout_rates', [0.2, 0.1])
        activation = config.get('activation', 'silu')

        # 替换 classifier 中的最后 Linear
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
            if 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {'params': backbone_params, 'lr_scale': lr_scale_backbone},
            {'params': head_params, 'lr_scale': lr_scale_head},
        ]


class DenseNet121Classifier(BaseClassifier):
    """DenseNet121 分类器"""
    def __init__(self, num_classes: int, config: dict = None):
        super().__init__(num_classes)
        config = config or {}
        self.lr_scale_backbone = config.get('lr_scale_backbone', 0.1)
        self.lr_scale_head = config.get('lr_scale_head', 1.0)

        self.base = safe_load_weights(
            models.densenet121,
            models.DenseNet121_Weights.IMAGENET1K_V1,
            "DenseNet121"
        )
        self.features = self.base.features

        freeze_strategy = config.get('freeze_strategy', 'frozen')
        freeze_patterns, unfreeze_patterns = get_freeze_patterns('DenseNet121', freeze_strategy)
        apply_freeze_by_modules(self.features, freeze_patterns, unfreeze_patterns)

        attention_channels = config.get('attention_channels', 1024)
        attention_dropout = config.get('attention_dropout', 0.0)
        self.attention = nn.Sequential(
            nn.Conv2d(attention_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(attention_dropout),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(attention_dropout),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention.apply(init_weights)

        hidden_dims = config.get('hidden_dims', [512, 256])
        dropout_rates = config.get('dropout_rates', [0.2, 0.1])
        activation = config.get('activation', 'relu')
        self.classifier = MLPHead(1024, num_classes, hidden_dims, dropout_rates, activation)

    def forward(self, x):
        features = self.features(x)
        attn = self.attention(features)
        weighted_features = features * attn
        pooled = weighted_features.mean(dim=[2, 3])
        return self.classifier(pooled)

    def get_param_groups(self, lr_scale_backbone=None, lr_scale_head=None):
        if lr_scale_backbone is None:
            lr_scale_backbone = self.lr_scale_backbone
        if lr_scale_head is None:
            lr_scale_head = self.lr_scale_head
        backbone_params, head_params = [], []
        for name, param in self.named_parameters():
            if 'classifier' in name or 'attention' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {'params': backbone_params, 'lr_scale': lr_scale_backbone},
            {'params': head_params, 'lr_scale': lr_scale_head},
        ]