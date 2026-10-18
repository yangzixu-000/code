import os
import sys
import json
import re
import argparse
import torch
import torch.distributed as dist
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 路径引导（ModuleNotFoundError 兜底）
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

def _ensure_project_root_on_syspath():
    roots_to_try = [_PROJECT_ROOT, os.path.abspath(os.getcwd())]
    for root in list(dict.fromkeys(roots_to_try)):
        if not root or not os.path.isdir(root):
            continue
        has_pkg = all(
            os.path.isdir(os.path.join(root, p)) and
            os.path.exists(os.path.join(root, p, '__init__.py'))
            for p in ('data', 'models', 'utils')
        )
        if not has_pkg:
            continue
        if root not in sys.path:
            sys.path.insert(0, root)
        os.environ.setdefault('OBWX_PROJECT_ROOT', root)
        return root
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    return _PROJECT_ROOT

PROJECT_ROOT = _ensure_project_root_on_syspath()

from utils.config import get_config
from utils.seed import set_seed
from data.dataset import get_dataloaders
from models import (
    ResNet18Classifier,
    EfficientNetB3Classifier,
    DenseNet121Classifier,
    ViTClassifier,
    SwinTransformerClassifier,
    ConvNeXtTinyClassifier,
    EfficientNetV2SClassifier,
    MaxViTClassifier,
    RegNetY4GFClassifier,
    MobileNetV3LargeClassifier,
    ShuffleNetV2Classifier,
    HybridCNNTransformerClassifier,
    InceptionNeXtSmallClassifier,
    ConvNeXtV2TinyClassifier,
    CoAtNetTinyClassifier,
    EfficientFormerV2SClassifier,
    MobileViTV2SClassifier,
    SwinV2TinyClassifier,
    RepViTM11Classifier,
    EdgeNeXtSmallRWClassifier,
    PVTv2B2Classifier,
    IFormerSClassifier,
    WideResNet50Classifier,
    ResNeXt50Classifier
)
from training.train import train_model, is_main_process, get_safe_name, get_rank, get_world_size, is_distributed


MODELS_CONFIG = {
    # ============================================================
    # 经典 CNN（全量微调，保持原有合理配置）
    # ============================================================
    "ResNet18": {
        "class": ResNet18Classifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [512],
            "dropout_rates": [0.2],
            "activation": "relu",
            "lr_scale_backbone": 0.2,      # 原有配置
            "lr_scale_head": 1.0
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 0.2,      # 关键：必须在 optimizer 中显式声明
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },
    "EfficientNet-B3": {
        "class": EfficientNetB3Classifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "silu",
            "lr_scale_backbone": 0.15,     # 原有配置
            "lr_scale_head": 1.0
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "lr_scale_backbone": 0.15,     # 显式声明
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-6, "per_batch": False}
    },
    "DenseNet121": {
        "class": DenseNet121Classifier,
        "config": {
            "freeze_strategy": "frozen",
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "relu",
            "attention_dropout": 0.0,
            "lr_scale_backbone": 0.1,      # 原有配置
            "lr_scale_head": 1.0
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 0.1,      # 显式声明
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": True}
    },

    # ============================================================
    # Transformer
    # ============================================================
    "ViT-B/16": {
        "class": ViTClassifier,
        "config": {
            "freeze_strategy": "frozen",
            "hidden_dims": [1024, 512],
            "dropout_rates": [0.3, 0.2],
            "activation": "gelu",
            "lr_scale_backbone": 0.1,      # 原有配置
            "lr_scale_head": 1.0
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 0.1,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "cosine", "T_max": 80, "eta_min": 1e-6, "per_batch": False}
    },
    "Swin Transformer": {
        "class": SwinTransformerClassifier,
        "config": {
            "freeze_strategy": "frozen",   # 保持冻结（已在 cnn_models 中解冻部分 stage）
            "hidden_dims": [1024, 512],
            "dropout_rates": [0.3, 0.2],
            "activation": "gelu",
            "lr_scale_backbone": 0.1,
            "lr_scale_head": 1.0
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 0.1,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "cosine", "T_max": 80, "eta_min": 1e-6, "per_batch": False}
    },

    # ============================================================
    # 现代 CNN
    # ============================================================
    "ConvNeXt-Tiny": {
        "class": ConvNeXtTinyClassifier,
        "config": {
            "freeze_strategy": "full",    
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,      # 显式声明，确保优化器读取
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 8, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },
    "EfficientNetV2-S": {
        "class": EfficientNetV2SClassifier,
        "config": {
            "freeze_strategy": "full",      # 全量微调
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },
    "MaxViT-T": {
        "class": MaxViTClassifier,
        "config": {
            "freeze_strategy": "full",      # 全量微调
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 8, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },
    "RegNetY-4GF": {
        "class": RegNetY4GFClassifier,
        "config": {
            "freeze_strategy": "full",      # 全量微调（即使回退到 400MF 也适用）
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 3e-4,                     # 提高学习率
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },

    # ---- 补齐论文 14 基线中缺失的两个模型（R2）----
    # WideResNet-50 / ResNeXt-50 的 seed=42 结果已存在于 model_metrics_14.json，
    # 此处补齐 wrapper 使其可训练，用于多种子补充实验的新种子臂。
    # 配置遵循新增模型统一协议：224×224 / ImageNet 预训练 / freeze=full /
    # hidden_dims=[512,256] / dropout=[0.2,0.1] / gelu / lr_scale_backbone=1.0，未做任何针对性调参。
    "WideResNet-50": {
        "class": WideResNet50Classifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },
    "ResNeXt-50": {
        "class": ResNeXt50Classifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },

    # ============================================================
    # 轻量级模型（全量微调）
    # ============================================================
    "MobileNetV3-Large": {
        "class": MobileNetV3LargeClassifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [256, 128],
            "dropout_rates": [0.2, 0.1],
            "activation": "relu",
            "lr_scale_backbone": 0.2,
            "lr_scale_head": 1.0
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 0.2,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 8, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },
    "ShuffleNetV2": {
        "class": ShuffleNetV2Classifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [256, 128],
            "dropout_rates": [0.2, 0.1],
            "activation": "relu",
            "lr_scale_backbone": 0.2,
            "lr_scale_head": 1.0
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 0.2,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 8, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },

    # ============================================================
    # 混合模型
    # ============================================================
    "Hybrid-CNN-Transformer": {
        "class": HybridCNNTransformerClassifier,
        "config": {
            "cnn_backbone": "resnet18",
            "transformer_dim": 512,
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.1,
            "freeze_strategy": "frozen",
            "use_cls_token": True,
            "pool_type": "cls",
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_cnn": 0.1,
            "lr_scale_transformer": 0.5,
            "lr_scale_head": 1.0
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_cnn": 0.1,
            "lr_scale_transformer": 0.5,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 8, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },

    # ============================================================
    # 新增现代架构（统一协议：224×224 / ImageNet 预训练 / batch_size=64 /
    # epoch=80 / seed=42 / 全局一致增强与长尾策略；不针对新模型调参，
    # 配置沿用现有 modern_cnns 家族：freeze=full, lr_scale_backbone=1.0）
    # ============================================================
    "InceptionNeXt-Small": {
        "class": InceptionNeXtSmallClassifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0,
            "batch_size": 64
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },
    "ConvNeXt-V2-Tiny": {
        "class": ConvNeXtV2TinyClassifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0,
            "batch_size": 64
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },
    "CoAtNet-Tiny": {
        "class": CoAtNetTinyClassifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0,
            "batch_size": 64
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },
    "EfficientFormerV2-S": {
        "class": EfficientFormerV2SClassifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0,
            "batch_size": 64
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },
    "iFormer-S": {
        "class": IFormerSClassifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0,
            "batch_size": 64
            # 若训练环境 timm 不含 iformer，可二选一启用：
            #   custom_impl: true            # 需提供 models/iformer_custom.py 且结构匹配权重
            #   pretrained_path: "path/to/iformer_small.pth"  # 加载官方权重（strict=False）
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },
    "MobileViT-V2-S": {
        "class": MobileViTV2SClassifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0,
            "batch_size": 64
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },
    "Swin-V2-Tiny": {
        "class": SwinV2TinyClassifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0,
            "batch_size": 64
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },
    "RepViT-M1.1": {
        "class": RepViTM11Classifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0,
            "batch_size": 64
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },
    "EdgeNeXt-Small-RW": {
        "class": EdgeNeXtSmallRWClassifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0,
            "batch_size": 64
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    },
    # 替代 CMT-Small（CMT 从未被 timm 收录）；同层级、多尺度混合结构
    "PVTv2-B2": {
        "class": PVTv2B2Classifier,
        "config": {
            "freeze_strategy": "full",
            "hidden_dims": [512, 256],
            "dropout_rates": [0.2, 0.1],
            "activation": "gelu",
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0,
            "batch_size": 64
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lr_scale_backbone": 1.0,
            "lr_scale_head": 1.0
        },
        "scheduler": {"type": "CosineWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-7, "per_batch": False}
    }
}


def get_trained_models(config):
    """检测已训练的模型"""
    trained = set()
    model_dir = config['output']['model_save_path']
    if not os.path.exists(model_dir):
        return trained
    saved_files = [f for f in os.listdir(model_dir) 
                   if f.startswith('best_') and f.endswith('.pth')]
    for model_key in MODELS_CONFIG.keys():
        safe_name = get_safe_name(model_key)
        if f"best_{safe_name}.pth" in saved_files:
            trained.add(model_key)
            if is_main_process():
                print(f"  ✅ {model_key} already trained")
    return trained


def generate_report(results, config):
    """生成报告（仅主进程执行）"""
    if not results or not is_main_process():
        return

    results_path = config['output']['results_path']
    results_sorted = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)

    df = pd.DataFrame([{
        'Model': r['model_name'],
        'Test_Accuracy': f"{r['test_accuracy']:.2f}",
        'Best_Val_Accuracy': f"{r['best_val_accuracy']:.2f}",
        'Precision': f"{r['precision']:.4f}",
        'Recall': f"{r['recall']:.4f}",
        'F1_Score': f"{r['f1_score']:.4f}"
    } for r in results_sorted])

    df.to_csv(os.path.join(results_path, 'benchmark_summary.csv'), index=False)

    detailed = {
        'dataset': config['dataset']['name'],
        'dataset_version': config['dataset']['version'],
        'model_results': results,
        'training_config': config['training']
    }
    with open(os.path.join(results_path, 'detailed_results.json'), 'w') as f:
        json.dump(detailed, f, indent=2)

    print("\n" + "=" * 60)
    print("Model Ranking")
    print("=" * 60)
    for i, r in enumerate(results_sorted):
        print(f"{i + 1}. {r['model_name']}: {r['test_accuracy']:.2f}% (F1: {r['f1_score']:.4f})")

    plt.figure(figsize=(14, 10))
    colors = plt.cm.tab20.colors

    for i, r in enumerate(results):
        if r.get('val_accuracies'):
            label = r['model_name']
            color = colors[i % len(colors)]
            plt.plot(r['val_accuracies'], label=label, linewidth=2, color=color)

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Validation Accuracy (%)', fontsize=14)
    plt.title('Validation Accuracy Curves - All Models', fontsize=16)

    plt.legend(
        bbox_to_anchor=(0.98, 0.02),
        loc='lower right',
        ncol=2,
        fontsize=11,
        framealpha=0.92,
        edgecolor='gray'
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'validation_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nResults saved to: {results_path}")


def run_benchmark(models_to_run=None, skip_trained=True):
    """运行基准测试"""
    # 在创建 DataLoader 前初始化分布式
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')

    rank = get_rank()
    world_size = get_world_size()
    is_dist = is_distributed()
    is_main = is_main_process()
    
    if is_main:
        print("=" * 60)
        if is_dist:
            print(f"OBWX Benchmark (DDP Mode, rank={rank}, world_size={world_size})")
        else:
            print("OBWX Benchmark (Single GPU Mode)")
        print("=" * 60)

    config = get_config()
    set_seed(config['random_seed'])

    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        config, rank=rank, world_size=world_size
    )

    if is_main:
        print(f"Classes: {num_classes}, Epochs: {config['training']['epochs']}")
        print(f"Steps per epoch: {len(train_loader)}")

    all_models = list(MODELS_CONFIG.keys())
    if models_to_run:
        all_models = [m for m in all_models if m in models_to_run]
    if skip_trained:
        trained = get_trained_models(config)
        all_models = [m for m in all_models if m not in trained]

    if not all_models:
        if is_main:
            print("\n✅ All models already trained. Nothing to do.")
        return

    if is_main:
        print(f"\n📋 Models to train ({len(all_models)}):")
        for m in all_models:
            print(f"  🏃 {m}")

    results = []
    last_bs = config['training']['batch_size']  # 预循环已按全局 bs 构建 DataLoader
    for model_name in tqdm(all_models, desc="Training", disable=not is_main):
        cfg = MODELS_CONFIG[model_name]

        # 每模型 batch_size：优先取模型 config 中的 batch_size，否则回退全局
        bs = cfg.get("config", {}).get("batch_size", config['training']['batch_size'])
        if bs != last_bs:
            # 仅在 batch_size 变化时重建 DataLoader，避免对原模型（bs=64）造成影响
            train_loader, val_loader, test_loader, num_classes = get_dataloaders(
                config, rank=rank, world_size=world_size, batch_size=bs
            )
            last_bs = bs
            if is_main:
                print(f"  🔁 DataLoaders rebuilt with batch_size={bs} (for {model_name})")

        model = cfg["class"](num_classes, config=cfg.get("config", {}))

        result = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            optimizer_config=cfg["optimizer"],
            scheduler_config=cfg.get("scheduler")
        )

        if result and is_main:
            results.append(result)
            result_path = os.path.join(
                config['output']['results_path'],
                f"result_{get_safe_name(model_name)}.json"
            )
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)

    if is_main:
        generate_report(results, config)
        print("\n🎉 Benchmark completed!")


def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='OBWX Benchmark')
        parser.add_argument('--models', nargs='+', help='Models to train')
        parser.add_argument('--retrain', action='store_true', help='Retrain all')
        parser.add_argument('--list', action='store_true', help='List available models')
        args = parser.parse_args()

        if args.list and is_main_process():
            print("\nAvailable models:")
            for i, name in enumerate(MODELS_CONFIG.keys(), 1):
                print(f"  {i}. {name}")
            return

        run_benchmark(models_to_run=args.models, skip_trained=not args.retrain)
    else:
        if is_main_process():
            print("\n" + "=" * 60)
            print("🔧 No command-line arguments. Using default model list.")
            print("   To customize: run with --models ModelName1 ModelName2")
            print("   To force retrain: run with --retrain")
            print("=" * 60)

        default_models = list(MODELS_CONFIG.keys())
        run_benchmark(models_to_run=default_models, skip_trained=True)


if __name__ == "__main__":
    main()