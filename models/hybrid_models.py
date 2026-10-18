"""
混合 CNN-Transformer 模型（完整修复版）

设计思路：
    1. CNN 提取 2D 特征图（保留空间结构）
    2. 特征图展平为 token 序列
    3. 添加 CLS Token 和动态位置编码
    4. Transformer 编码器建模全局空间依赖
    5. CLS Token 或均值池化后分类

支持骨干：
    - resnet18: 输出 512×7×7，特征图尺寸 7×7
    - resnet50: 输出 2048×7×7，特征图尺寸 7×7
    - efficientnet_b0: 输出 1280×7×7，特征图尺寸 7×7

默认输入尺寸：224×224（特征图尺寸 7×7）
支持动态输入尺寸：位置编码在每次前向时动态生成
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
    init_weights,
    MLPHead
)

logger = logging.getLogger(__name__)


class EfficientNetWrapper(nn.Module):
    """EfficientNet 包装类，使 forward 输出 4D 特征图"""
    def __init__(self, backbone):
        super().__init__()
        self.features = backbone.features

    def forward(self, x):
        return self.features(x)


class ResNetFeatureExtractor(nn.Module):
    """只提取 ResNet 的 4D 特征图，跳过 avgpool 和 fc，避免 flatten"""
    def __init__(self, backbone):
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # (B, C, H, W)


class HybridCNNTransformerClassifier(BaseClassifier):
    """
    混合 CNN-Transformer 分类器

    默认输入尺寸：224×224，特征图尺寸：7×7
    参数量：取决于 CNN 骨干和 Transformer 配置

    支持骨干（cnn_backbone）：
        - 'resnet18': 输出 512×7×7，轻量快速
        - 'resnet50': 输出 2048×7×7，更强特征
        - 'efficientnet_b0': 输出 1280×7×7，高效

    配置参数：
        cnn_backbone: 骨干名称，默认 'resnet18'
        transformer_dim: Transformer 编码维度，默认 512
        num_heads: 注意力头数，默认 8
        num_layers: 编码器层数，默认 2
        dropout: Dropout 概率，默认 0.1
        use_cls_token: 是否使用 CLS Token，默认 True
        pool_type: 池化方式，'cls' 或 'mean'，默认 'cls'
        freeze_strategy: 'frozen' / 'partial' / 'full'，默认 'frozen'
        hidden_dims: 分类头隐藏层维度，默认 [512]
        dropout_rates: 分类头 Dropout 率，默认 [0.2]
        activation: 分类头激活函数，默认 'gelu'
    """

    def __init__(self, num_classes: int, config: dict = None):
        super().__init__(num_classes)
        config = config or {}

        # ===== 配置参数 =====
        cnn_backbone = config.get('cnn_backbone', 'resnet18')
        transformer_dim = config.get('transformer_dim', 512)
        num_heads = config.get('num_heads', 8)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.1)
        use_cls_token = config.get('use_cls_token', True)
        pool_type = config.get('pool_type', 'cls')
        freeze_strategy = config.get('freeze_strategy', 'frozen')
        hidden_dims = config.get('hidden_dims', [512])
        dropout_rates = config.get('dropout_rates', [0.2])
        activation = config.get('activation', 'gelu')

        self.lr_scale_cnn = config.get('lr_scale_cnn', 0.1)
        self.lr_scale_trans = config.get('lr_scale_transformer', 0.5)
        self.lr_scale_head = config.get('lr_scale_head', 1.0)

        # 1. 构建 CNN 骨干（输出 4D 特征图）
        self.cnn, cnn_out_dim, feat_h, feat_w = self._build_cnn_backbone(cnn_backbone)

        # 2. 冻结策略
        backbone_key_map = {
            'resnet18': 'ResNet18',
            'resnet50': 'ResNet50',
            'efficientnet_b0': 'EfficientNet-B0',
        }
        backbone_key = backbone_key_map.get(cnn_backbone, cnn_backbone.capitalize())

        if freeze_strategy in ['frozen', 'partial']:
            freeze_patterns, unfreeze_patterns = get_freeze_patterns(backbone_key, freeze_strategy)
            frozen_cnt, train_cnt = apply_freeze_by_modules(self.cnn, freeze_patterns, unfreeze_patterns)
            logger.info(
                f"  🔒 Hybrid-CNN-Transformer: CNN backbone '{cnn_backbone}' {freeze_strategy}, "
                f"frozen_params={frozen_cnt}, trainable_params={train_cnt}"
            )
        elif freeze_strategy == 'full':
            logger.info(f"  🔓 Hybrid-CNN-Transformer: full training (no freezing)")
        else:
            logger.warning(f"  ⚠️ Unknown freeze_strategy '{freeze_strategy}', using full training as fallback")

        # 3. Token 序列化参数
        self.seq_len = feat_h * feat_w
        self.cnn_out_dim = cnn_out_dim

        # 4. 投影层
        self.proj = nn.Sequential(
            nn.Linear(cnn_out_dim, transformer_dim),
            nn.Dropout(dropout)
        )

        # 5. CLS Token
        self.use_cls_token = use_cls_token
        self.pool_type = pool_type
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim) * 0.02)

        # 6. Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer.apply(init_weights)

        # 7. 分类头
        self.classifier = MLPHead(
            transformer_dim, num_classes,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
            activation=activation
        )

        logger.info(
            f"📐 Hybrid-CNN-Transformer: backbone={cnn_backbone}, "
            f"seq_len={self.seq_len}, dim={transformer_dim}, "
            f"heads={num_heads}, layers={num_layers}, use_cls={use_cls_token}, "
            f"freeze={freeze_strategy}"
        )

    def _create_sinusoidal_position_encoding(self, seq_len: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, dim, device=self.proj[0].weight.device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=pe.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float, device=pe.device) *
            (-torch.log(torch.tensor(10000.0, device=pe.device)) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _build_cnn_backbone(self, backbone_name: str):
        if backbone_name == 'resnet18':
            backbone = safe_load_weights(
                models.resnet18,
                models.ResNet18_Weights.IMAGENET1K_V1,
                "ResNet18"
            )
            backbone = ResNetFeatureExtractor(backbone)   # 关键修复
            return backbone, 512, 7, 7

        elif backbone_name == 'resnet50':
            backbone = safe_load_weights(
                models.resnet50,
                models.ResNet50_Weights.IMAGENET1K_V1,
                "ResNet50"
            )
            backbone = ResNetFeatureExtractor(backbone)
            return backbone, 2048, 7, 7

        elif backbone_name == 'efficientnet_b0':
            backbone = safe_load_weights(
                models.efficientnet_b0,
                models.EfficientNet_B0_Weights.IMAGENET1K_V1,
                "EfficientNet-B0"
            )
            return EfficientNetWrapper(backbone), 1280, 7, 7

        else:
            supported = ['resnet18', 'resnet50', 'efficientnet_b0']
            raise ValueError(f"Unsupported CNN backbone: '{backbone_name}'. Supported: {supported}")

    def _extract_feature_sequence(self, x):
        B = x.shape[0]

        cnn_feat = self.cnn(x)           # (B, C, H, W)
        B, C, H, W = cnn_feat.shape

        tokens = cnn_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        current_seq_len = H * W

        proj_feat = self.proj(tokens)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            feat_with_cls = torch.cat([cls_tokens, proj_feat], dim=1)
            total_seq_len = current_seq_len + 1
        else:
            feat_with_cls = proj_feat
            total_seq_len = current_seq_len

        pos_enc = self._create_sinusoidal_position_encoding(total_seq_len, self.proj[0].out_features)
        feat_with_pos = feat_with_cls + pos_enc

        encoded = self.transformer(feat_with_pos)

        if self.use_cls_token and self.pool_type == 'cls':
            pooled = encoded[:, 0, :]
        else:
            pooled = encoded.mean(dim=1)

        return {
            'cnn_feat': cnn_feat,
            'tokens': tokens,
            'proj_feat': proj_feat,
            'encoded': encoded,
            'pooled': pooled
        }

    def forward(self, x):
        features = self._extract_feature_sequence(x)
        return self.classifier(features['pooled'])

    def get_features(self, x):
        return self._extract_feature_sequence(x)

    def get_param_groups(self, lr_scale_cnn=None, lr_scale_transformer=None, lr_scale_head=None):
        if lr_scale_cnn is None:
            lr_scale_cnn = self.lr_scale_cnn
        if lr_scale_transformer is None:
            lr_scale_transformer = self.lr_scale_trans
        if lr_scale_head is None:
            lr_scale_head = self.lr_scale_head

        cnn_params, proj_cls_params, transformer_core_params, head_params = [], [], [], []
        for name, param in self.named_parameters():
            if name.startswith('cnn.'):
                cnn_params.append(param)
            elif name.startswith('classifier.'):
                head_params.append(param)
            elif name.startswith('transformer.'):
                transformer_core_params.append(param)
            elif name in ['cls_token'] or name.startswith('proj.'):
                proj_cls_params.append(param)
            else:
                transformer_core_params.append(param)

        param_groups = []
        if cnn_params:
            param_groups.append({'params': cnn_params, 'lr_scale': lr_scale_cnn})
        if proj_cls_params:
            param_groups.append({'params': proj_cls_params, 'lr_scale': lr_scale_transformer * 0.8})
        if transformer_core_params:
            param_groups.append({'params': transformer_core_params, 'lr_scale': lr_scale_transformer})
        if head_params:
            param_groups.append({'params': head_params, 'lr_scale': lr_scale_head})

        return param_groups