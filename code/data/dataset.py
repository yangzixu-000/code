"""
OBWX 数据集类和数据加载
所有模型共享此文件中的数据增强策略

增强策略说明（古文字安全版）：
  - 无水平翻转：古文字具有方向性
  - 旋转 ±5°（30% 概率），旋转填充白色背景
  - 亮度/对比度抖动 ±10%（30% 概率）
  - 训练阶段应用 ImageNet 归一化
  - 支持 DDP 分布式训练（DistributedSampler）
"""
import cv2
import os
import glob
import json
import logging
import platform
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.config import validate_processed_root

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_distributed():
    """判断是否处于分布式环境"""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """获取当前进程的 rank"""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size():
    """获取分布式总进程数"""
    if is_distributed():
        return dist.get_world_size()
    return 1


def worker_init_fn(worker_id):
    """DataLoader worker 随机种子固定"""
    import random
    import numpy as np
    base_seed = 42
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


class OBWXDataset(Dataset):
    def __init__(self, root, split, class_to_idx, transform=None, image_extensions=None, image_size=224):
        self.root = root
        self.split = split
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.image_size = image_size
        
        if image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        else:
            self.image_extensions = image_extensions
        
        self.samples = self._load_samples_optimized()
        logger.info(f"{self.split}: loaded {len(self.samples)} samples, {len(set(c for _, c in self.samples))} classes")
    
    def _load_samples_optimized(self):
        """优化版本：单次遍历目录，减少 glob 调用"""
        samples = []
        split_dir = os.path.join(self.root, self.split)
        if not os.path.exists(split_dir):
            logger.warning(f"Split directory {split_dir} does not exist")
            return samples
        
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            if class_name not in self.class_to_idx:
                logger.warning(f"Class '{class_name}' in {self.split} not in class_to_idx, skipping")
                continue
            
            class_idx = self.class_to_idx[class_name]
            for fname in os.listdir(class_path):
                ext = os.path.splitext(fname)[1]
                if ext.lower() in ['.jpg', '.jpeg', '.png']:
                    fpath = os.path.join(class_path, fname)
                    if os.path.getsize(fpath) > 0:
                        samples.append((fpath, class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """跳过损坏图片，返回有效样本"""
        if idx >= len(self.samples):
            raise RuntimeError("No valid samples left after skipping corrupt images.")
        
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed["image"]
            return img, label
        except Exception as e:
            logger.warning(f"Corrupt image: {img_path} (error: {e}), skipping to next index")
            new_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(new_idx)


def get_transform(train=True, image_size=224):
    """
    获取数据变换
    训练阶段：
        - Resize 到 224×224
        - 随机旋转 ±5°（30% 概率），填充白色（255）以避免黑边
        - 亮度/对比度抖动 ±10%（30% 概率）
        - 无水平翻转
        - ImageNet 归一化
    验证/测试：
        - Resize + 归一化
    """
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if train:
        augmentations = [
            A.Resize(image_size, image_size),
            A.Rotate(limit=5, p=0.3, fill=255, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
        ]
        # 可选增强（模拟拓片污渍），默认注释
        # augmentations.extend([
        #     A.GaussNoise(var_limit=(3.0, 12.0), p=0.2),
        #     A.CoarseDropout(max_holes=3, max_height=8, max_width=8, fill_value=255, p=0.2)
        # ])
        augmentations.extend([normalize, ToTensorV2()])
        return A.Compose(augmentations)
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            normalize,
            ToTensorV2()
        ])


def get_dataloaders(config, rank=None, world_size=None, batch_size=None):
    """
    创建训练、验证、测试 DataLoader
    完整支持 DDP 分布式训练

    Args:
        config: 配置字典
        rank: 当前进程 rank（分布式时由外部传入）
        world_size: 总进程数（分布式时由外部传入）
        batch_size: 可选的每模型覆盖值；为 None 时回退到 config['training']['batch_size']
            （用于让不同模型使用不同 batch_size，例如新增模型统一用 32，原模型保持 64）

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    processed_root = config['data']['processed_root']
    # 每模型覆盖优先；否则使用全局 batch_size
    batch_size = batch_size if batch_size is not None else config['training']['batch_size']
    num_workers = config['training']['num_workers']
    pin_memory = config['training']['pin_memory'] and torch.cuda.is_available()
    image_size = config['augmentation']['train']['image_size']
    image_extensions = config.get('image_extensions', None)
    
    # 强制检查 class_mapping.json 存在
    mapping_path = os.path.join(processed_root, "class_mapping.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"Missing class mapping file {mapping_path}. "
            "Please run the preprocessing script first to generate the dataset."
        )
    
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)

    # ===== R1 防呆：确认数据根是完整且符合 §3.2 协议的构建（不匹配直接抛错，不静默回退）=====
    validate_processed_root(processed_root, config['data'].get('expected'))

    class_to_idx = class_mapping["class_to_idx"]
    # 子集探针支持：仅保留前 N 个类别（按类别名排序），用于多种子稳定性探针等场景
    max_classes = config['data'].get('max_classes')
    if max_classes:
        class_to_idx = dict(
            sorted(class_to_idx.items(), key=lambda kv: kv[0])[:int(max_classes)]
        )
    num_classes = len(class_to_idx)
    
    # Windows 平台兼容：默认 num_workers=0（避免 spawn 多进程的兼容问题）。
    # 逃生门：设置环境变量 OBWX_FORCE_NUM_WORKERS=N 可显式指定（如 8）以启用多进程数据加载。
    # 实测（28 核，224px + albumentations）：nw=0 → 837 img/s，nw=4 → 1,976 img/s，
    # nw=8 → 2,776 img/s。GPU 机器上数据加载常成为瓶颈，故提供该开关；默认值不变（0）。
    force_nw = os.environ.get("OBWX_FORCE_NUM_WORKERS")
    if force_nw is not None:
        num_workers = int(force_nw)
        logger.info(f"OBWX_FORCE_NUM_WORKERS={num_workers} (显式覆盖平台默认值)")
    elif platform.system() == "Windows":
        num_workers = 0
        logger.info("Windows detected: setting num_workers=0 to avoid multiprocessing issues.")
    
    # 创建变换
    train_transform = get_transform(train=True, image_size=image_size)
    val_transform = get_transform(train=False, image_size=image_size)
    test_transform = get_transform(train=False, image_size=image_size)
    
    # 创建数据集
    train_dataset = OBWXDataset(
        processed_root, "train", class_to_idx, train_transform, image_extensions, image_size
    )
    val_dataset = OBWXDataset(
        processed_root, "val", class_to_idx, val_transform, image_extensions, image_size
    )
    test_dataset = OBWXDataset(
        processed_root, "test", class_to_idx, test_transform, image_extensions, image_size
    )
    
    # ===== DDP 分布式采样器 =====
    is_dist = is_distributed()
    
    if is_dist:
        if rank is None or world_size is None:
            raise ValueError("DDP mode requires rank and world_size to be provided.")
        
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=42
        )
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=42
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=worker_init_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=worker_init_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, sampler=test_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=worker_init_fn
        )
        logger.info(f"DDP mode: rank={rank}, world_size={world_size}, batch_size={batch_size}")
    else:
        # 单卡模式
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=worker_init_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=worker_init_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=worker_init_fn
        )
    
    if get_rank() == 0:
        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)} | Classes: {num_classes}")
    
    return train_loader, val_loader, test_loader, num_classes


def get_test_dataloader(config, batch_size=None, rank=None, world_size=None):
    """
    单独获取测试集 DataLoader，避免加载训练/验证数据
    """
    processed_root = config['data']['processed_root']
    if batch_size is None:
        batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    pin_memory = config['training']['pin_memory'] and torch.cuda.is_available()
    image_size = config['augmentation']['train']['image_size']
    image_extensions = config.get('image_extensions', None)
    
    mapping_path = os.path.join(processed_root, "class_mapping.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Missing class mapping file {mapping_path}")
    with open(mapping_path, 'r') as f:
        class_to_idx = json.load(f)["class_to_idx"]
    max_classes = config['data'].get('max_classes')
    if max_classes:
        class_to_idx = dict(
            sorted(class_to_idx.items(), key=lambda kv: kv[0])[:int(max_classes)]
        )
    num_classes = len(class_to_idx)
    
    if platform.system() == "Windows":
        num_workers = 0
    
    test_transform = get_transform(train=False, image_size=image_size)
    test_dataset = OBWXDataset(
        processed_root, "test", class_to_idx, test_transform, image_extensions, image_size
    )
    
    is_dist = is_distributed()
    if is_dist:
        if rank is None or world_size is None:
            raise ValueError("DDP mode requires rank and world_size to be provided.")
        sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=42
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=worker_init_fn
        )
    else:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=worker_init_fn
        )
    
    return test_loader, num_classes