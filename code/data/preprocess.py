import os
import sys
import shutil
import json
import random
import logging
import hashlib
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from tqdm import tqdm
import albumentations as A

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import get_config


# ============================================================
# 常量定义（消除魔法数字）
# ============================================================
DEFAULT_GLOBAL_THRESHOLD = 100
DEFAULT_CORNER_THRESHOLD = 128
DEFAULT_BG_CLEAN_THRESHOLD = 220
DEFAULT_BG_DARK_THRESHOLD = 190
DEFAULT_MIN_IMAGES = 15
DEFAULT_IMAGE_SIZE = 224
JPEG_QUALITY = 95
AUGMENT_ROTATE_RANGE = (-1, 1)   # 离线增强旋转 ±1°
AUGMENT_SCALE_RANGE = (0.98, 1.02)
AUGMENT_SHEAR_RANGE = (-0.5, 0.5)


# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 图片标准化类（统一白底黑字，只反转一次）
# ============================================================
class AncientTextNormalizer:
    
    @staticmethod
    def normalize_to_white_black(img_array):
        """
        将图片标准化为白底黑字。
        使用全局均值和四角均值综合判断，只反转一次，避免冲突。
        对于“白底占少数、中间黑底白字”的混合图片，文字部分会得到正确处理。
        """
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        h, w = gray.shape
        corner_h, corner_w = max(5, h // 20), max(5, w // 20)
        
        # 四角区域灰度均值
        corners = [
            gray[:corner_h, :corner_w],
            gray[:corner_h, -corner_w:],
            gray[-corner_h:, :corner_w],
            gray[-corner_h:, -corner_w:]
        ]
        corner_mean = np.mean([np.mean(c) for c in corners])
        global_mean = np.mean(gray)
        
        # 判断：背景偏暗则反转，统一为白底黑字（只反转一次）
        # 条件1：全局均值低于阈值（整图偏暗）
        # 条件2：四角均值低于阈值（边缘也偏暗，典型拓片）
        if global_mean < DEFAULT_GLOBAL_THRESHOLD or corner_mean < DEFAULT_CORNER_THRESHOLD:
            gray = 255 - gray
        
        return gray
    
    @staticmethod
    def enhance_text_clarity(img_array):
        """轻微增强文字清晰度（对比度+锐化）"""
        img_pil = Image.fromarray(img_array)
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Sharpness(img_pil)
        img_pil = enhancer.enhance(1.1)
        return np.array(img_pil)
    
    @staticmethod
    def clean_background(img_array):
        """
        清理背景为纯白，同时保留文字细节。
        策略：先用固定阈值（安全），若背景偏暗则用自适应阈值辅助并取并集，避免丢失细笔画。
        """
        # 1. 固定阈值（主要方法，安全，不会丢失文字）
        result = img_array.copy()
        white_mask = result > DEFAULT_BG_CLEAN_THRESHOLD
        result[white_mask] = 255
        
        # 2. 检测背景亮度（四角均值）
        h, w = result.shape
        corner_h, corner_w = max(10, h // 20), max(10, w // 20)
        corners = [
            result[:corner_h, :corner_w],
            result[:corner_h, -corner_w:],
            result[-corner_h:, :corner_w],
            result[-corner_h:, -corner_w:]
        ]
        corner_mean = np.mean([np.mean(c) for c in corners])
        
        # 3. 如果背景明显偏暗（纸张泛黄、拓片底色深），使用自适应阈值二次清理并取并集
        if corner_mean < DEFAULT_BG_DARK_THRESHOLD:
            # 使用保守参数（blockSize=15, C=3）避免擦除文字
            binary = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 3
            )
            # 取并集：只要任一结果认为是文字（黑色），就保留
            # 固定阈值结果和 binary 中，黑色 = 0，白色 = 255
            # 取最小值可以保留黑色区域（文字）
            result = np.minimum(result, binary)
        
        return result


# ============================================================
# 离线增强流水线（仅几何变换，无翻转，无归一化）
# ============================================================
def get_conservative_augmentation(image_size=DEFAULT_IMAGE_SIZE):
    """保守的离线增强，用于扩充小样本类别（仅几何变换）"""
    return A.Compose([
        A.LongestMaxSize(max_size=256, interpolation=cv2.INTER_LANCZOS4),
        A.PadIfNeeded(
            min_height=256, min_width=256,
            border_mode=cv2.BORDER_CONSTANT, value=255, p=1.0
        ),
        A.RandomCrop(height=image_size, width=image_size, p=1.0),
        A.Affine(
            scale=AUGMENT_SCALE_RANGE,
            translate_percent=(0.01, 0.01),
            rotate=AUGMENT_ROTATE_RANGE,
            shear=AUGMENT_SHEAR_RANGE,
            p=1.0
        ),
        # 无 Normalize，无 ToTensorV2，保持 uint8 输出
    ])


def get_basic_transform(image_size=DEFAULT_IMAGE_SIZE):
    """基础变换（仅尺寸调整，无增强，无归一化）"""
    return A.Compose([
        A.LongestMaxSize(max_size=256, interpolation=cv2.INTER_LANCZOS4),
        A.PadIfNeeded(
            min_height=256, min_width=256,
            border_mode=cv2.BORDER_CONSTANT, value=255, p=1.0
        ),
        A.CenterCrop(height=image_size, width=image_size),
        # 无 Normalize，无 ToTensorV2
    ])


# ============================================================
# 全局 Transform 缓存（支持尺寸动态刷新）
# ============================================================
_BASIC_TRANSFORM = None
_CONSERVATIVE_AUG = None
_CURRENT_IMAGE_SIZE = None


def get_global_transforms(image_size=DEFAULT_IMAGE_SIZE):
    """获取 Transform 单例，尺寸变化时自动重建"""
    global _BASIC_TRANSFORM, _CONSERVATIVE_AUG, _CURRENT_IMAGE_SIZE
    
    if _BASIC_TRANSFORM is None or _CURRENT_IMAGE_SIZE != image_size:
        _BASIC_TRANSFORM = get_basic_transform(image_size)
        _CONSERVATIVE_AUG = get_conservative_augmentation(image_size)
        _CURRENT_IMAGE_SIZE = image_size
    
    return _BASIC_TRANSFORM, _CONSERVATIVE_AUG


# ============================================================
# 单张图片处理函数（返回状态信息）
# ============================================================
def process_single_image(img_path, do_augment=False, image_size=DEFAULT_IMAGE_SIZE):
    """
    处理单张图片：标准化 + 可选增强，返回白底黑字 uint8 图片。
    
    Returns:
        (success: bool, image_array: np.ndarray or None, error_msg: str or None)
    """
    try:
        img = Image.open(img_path)
        
        # 处理透明 PNG（填充白色背景）
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        else:
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # 1. 标准化为白底黑字
        normalized = AncientTextNormalizer.normalize_to_white_black(img_array)
        
        # 2. 增强文字清晰度
        enhanced = AncientTextNormalizer.enhance_text_clarity(normalized)
        
        # 3. 清理背景（不丢文字）
        cleaned = AncientTextNormalizer.clean_background(enhanced)
        
        # 4. 应用几何变换
        img_3ch = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        
        basic_transform, conservative_aug = get_global_transforms(image_size)
        
        if do_augment:
            transformed = conservative_aug(image=img_3ch)
        else:
            transformed = basic_transform(image=img_3ch)
        
        # transformed["image"] 是 numpy 数组 (H,W,C) uint8
        return True, transformed["image"], None
        
    except Exception as e:
        return False, None, str(e)


# ============================================================
# 小样本类别补齐函数
# ============================================================
def augment_images_to_target(image_paths, target_count, image_size=DEFAULT_IMAGE_SIZE):
    """
    将图片列表增强到目标数量，每次从原图重新增强，保证多样性。
    
    Returns:
        (processed_images, success_count, failed_count)
    """
    processed = []
    success_count = 0
    failed_count = 0
    
    # 1. 先处理所有原始图片（不增强）
    for img_path in image_paths:
        success, img_arr, err = process_single_image(
            img_path, do_augment=False, image_size=image_size
        )
        if success and img_arr is not None:
            processed.append(img_arr)
            success_count += 1
        else:
            failed_count += 1
    
    if not processed:
        return [], success_count, failed_count
    
    if len(processed) >= target_count:
        return processed[:target_count], success_count, failed_count
    
    # 2. 通过增强补齐到目标数量（每次从原图重新增强）
    needed = target_count - len(processed)
    for _ in range(needed):
        img_path = random.choice(image_paths)
        success, img_arr, err = process_single_image(
            img_path, do_augment=True, image_size=image_size
        )
        if success and img_arr is not None:
            processed.append(img_arr)
            success_count += 1
        else:
            failed_count += 1
    
    return processed[:target_count], success_count, failed_count


# ============================================================
# 单类别划分与增强
# ============================================================
def split_and_augment_class(original_images, class_name, config, failed_log=None):
    """
    对单个类别进行划分和增强。
    
    Returns:
        (train_images, val_images, test_images, stats_info)
        stats_info 包含各项计数和失败文件列表
    """
    original_count = len(original_images)
    min_images = config.get('split', {}).get('min_images_per_class', DEFAULT_MIN_IMAGES)
    image_size = config.get('augmentation', {}).get('train', {}).get('image_size', DEFAULT_IMAGE_SIZE)
    train_ratio = config.get('split', {}).get('train_ratio', 0.70)
    val_ratio = config.get('split', {}).get('val_ratio', 0.15)
    
    stats_info = {
        'original_count': original_count,
        'train_success': 0,
        'train_failed': 0,
        'val_success': 0,
        'val_failed': 0,
        'test_success': 0,
        'test_failed': 0,
        'is_all_train': False,
        'failed_files': []
    }
    
    # 原始图片 ≤ 3 张：全部放训练集
    if original_count <= 3:
        logger.info(f"  ⚠️  {class_name}: {original_count} images, all to train set")
        train_imgs, succ, fail = augment_images_to_target(
            original_images, min_images, image_size
        )
        stats_info['train_success'] = succ
        stats_info['train_failed'] = fail
        stats_info['is_all_train'] = True
        return train_imgs, [], [], stats_info
    
    # 原始图片 ≥ 4 张：按比例划分
    random.shuffle(original_images)
    n_total = original_count
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_test = n_total - n_train - n_val
    
    # 边界保护：确保每个集合至少 1 张
    if n_test < 1:
        n_train -= 1
        n_test = 1
    if n_train < 1:
        n_train = 1
        n_val = max(1, n_total - n_train - 1)
        n_test = n_total - n_train - n_val
        if n_test < 1:
            n_val -= 1
            n_test = 1
    
    train_originals = original_images[:n_train]
    val_originals = original_images[n_train:n_train + n_val]
    test_originals = original_images[n_train + n_val:]
    
    # ----- 训练集 -----
    if len(train_originals) < min_images:
        train_imgs, succ, fail = augment_images_to_target(
            train_originals, min_images, image_size
        )
    else:
        train_imgs = []
        succ = 0
        fail = 0
        for p in train_originals:
            success, img_arr, err = process_single_image(p, do_augment=False, image_size=image_size)
            if success and img_arr is not None:
                train_imgs.append(img_arr)
                succ += 1
            else:
                fail += 1
    stats_info['train_success'] = succ
    stats_info['train_failed'] = fail
    
    # ----- 验证集（纯原始，不增强） -----
    val_imgs = []
    succ = 0
    fail = 0
    for p in val_originals:
        success, img_arr, err = process_single_image(p, do_augment=False, image_size=image_size)
        if success and img_arr is not None:
            val_imgs.append(img_arr)
            succ += 1
        else:
            fail += 1
            stats_info['failed_files'].append((p, err))
    stats_info['val_success'] = succ
    stats_info['val_failed'] = fail
    
    # ----- 测试集（纯原始，不增强） -----
    test_imgs = []
    succ = 0
    fail = 0
    for p in test_originals:
        success, img_arr, err = process_single_image(p, do_augment=False, image_size=image_size)
        if success and img_arr is not None:
            test_imgs.append(img_arr)
            succ += 1
        else:
            fail += 1
            stats_info['failed_files'].append((p, err))
    stats_info['test_success'] = succ
    stats_info['test_failed'] = fail
    
    return train_imgs, val_imgs, test_imgs, stats_info


# ============================================================
# 主预处理流程
# ============================================================
def preprocess():
    config = get_config()
    
    # 安全读取配置
    raw_root = config.get('data', {}).get('raw_root', './data/obwx_raw')
    processed_root = config.get('data', {}).get('processed_root', './data/obwx_processed')
    image_extensions = config.get('image_extensions', ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'])
    image_size = config.get('augmentation', {}).get('train', {}).get('image_size', DEFAULT_IMAGE_SIZE)
    
    print("=" * 60)
    print("OBWX Data Preprocessing (Final Fix)")
    print("=" * 60)
    print(f"Raw data:   {raw_root}")
    print(f"Output:     {processed_root}")
    print(f"Image size: {image_size}")
    print("Processing: white background + black text (only one inversion)")
    print("           : fixed threshold + adaptive threshold (union) to preserve text")
    print("=" * 60)
    
    # 清空输出目录（彻底清除旧文件）
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(processed_root, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir)
    
    if not os.path.exists(raw_root):
        raise FileNotFoundError(f"Raw data not found: {raw_root}")
    
    # 获取类别列表
    class_names = [d for d in os.listdir(raw_root)
                   if os.path.isdir(os.path.join(raw_root, d))]
    class_names.sort()
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    
    # 统计信息
    stats = {
        'total_classes': len(class_names),
        'total_train': 0,
        'total_val': 0,
        'total_test': 0,
        'classes_all_train': 0,
        'classes_with_val': 0,
        'classes_with_test': 0,
        'total_train_failed': 0,
        'total_val_failed': 0,
        'total_test_failed': 0,
        'empty_classes': 0,
    }
    all_failed_files = {}   # 记录所有失败图片
    class_stats = {}        # 每个类别的详细统计
    
    for class_name in tqdm(class_names, desc="Processing"):
        class_path = os.path.join(raw_root, class_name)
        
        # 收集该类别所有图片
        # 注意：image_extensions 同时列了 ".jpg" 与 ".JPG"（.png/.PNG、
        # .jpeg/.JPEG 同理），下方 f.lower().endswith(ext.lower()) 会把同一
        # 物理文件匹配两次，导致每张图在池里出现两份副本；随后
        # split_and_augment_class 对池子 shuffle 后切 70/15/15，两份副本约
        # 70% 概率落入不同子集，造成 train/test 字节级重复（实测 71.16% 测试
        # 样本与训练集相同——典型数据泄漏）。因此对收集结果做两级去重：
        #   1) 按归一化绝对路径去重（消除扩展名大小写重复列举产生的同路径副本）
        #   2) 按文件内容 md5 去重（消除不同路径下的同图副本）
        # 保证划分建立在唯一图像之上，从根上杜绝跨子集泄漏。
        raw_files = []
        for ext in image_extensions:
            for f in os.listdir(class_path):
                if f.lower().endswith(ext.lower()):
                    raw_files.append(os.path.join(class_path, f))

        # 第 1 级：路径去重
        _by_path = {}
        for p in raw_files:
            _key = os.path.normcase(os.path.abspath(p))
            if _key not in _by_path:
                _by_path[_key] = p
        n_path_dup = len(raw_files) - len(_by_path)
        image_files = list(_by_path.values())

        # 第 2 级：内容去重（流式 md5，避免大文件一次性读入内存）
        _seen_md5 = set()
        _deduped = []
        n_content_dup = 0
        for p in image_files:
            try:
                _h = hashlib.md5()
                with open(p, 'rb') as _fh:
                    for _chunk in iter(lambda: _fh.read(1 << 20), b''):
                        _h.update(_chunk)
                _digest = _h.hexdigest()
            except OSError:
                # 读失败则退回路径去重，避免丢图
                _digest = os.path.normcase(os.path.abspath(p))
            if _digest not in _seen_md5:
                _seen_md5.add(_digest)
                _deduped.append(p)
            else:
                n_content_dup += 1
        image_files = _deduped
        if n_path_dup or n_content_dup:
            logger.info(
                f"  🔍 {class_name}: 去重 路径重复={n_path_dup} "
                f"内容重复={n_content_dup} → 唯一图像={len(image_files)}"
            )

        if not image_files:
            logger.warning(f"  ⚠️  {class_name}: no images found, skipping")
            stats['empty_classes'] += 1
            continue
        
        # 划分并增强
        train_imgs, val_imgs, test_imgs, info = split_and_augment_class(
            image_files, class_name, config
        )
        
        # 记录失败文件
        if info['failed_files']:
            all_failed_files[class_name] = info['failed_files']
        
        # 保存类别统计
        class_stats[class_name] = {
            'original': info['original_count'],
            'train_success': info['train_success'],
            'train_failed': info['train_failed'],
            'val_success': info['val_success'],
            'val_failed': info['val_failed'],
            'test_success': info['test_success'],
            'test_failed': info['test_failed'],
        }
        
        stats['total_train_failed'] += info['train_failed']
        stats['total_val_failed'] += info['val_failed']
        stats['total_test_failed'] += info['test_failed']
        
        if info['is_all_train']:
            stats['classes_all_train'] += 1
        
        # ---- 保存训练集 ----
        if train_imgs:
            train_dir = os.path.join(processed_root, "train", class_name)
            os.makedirs(train_dir, exist_ok=True)
            for i, img in enumerate(train_imgs):
                if img is not None:
                    Image.fromarray(img).save(
                        os.path.join(train_dir, f"{i:04d}.jpg"), quality=JPEG_QUALITY
                    )
            stats['total_train'] += len(train_imgs)
        
        # ---- 保存验证集 ----
        if val_imgs:
            val_dir = os.path.join(processed_root, "val", class_name)
            os.makedirs(val_dir, exist_ok=True)
            for i, img in enumerate(val_imgs):
                if img is not None:
                    Image.fromarray(img).save(
                        os.path.join(val_dir, f"{i:04d}.jpg"), quality=JPEG_QUALITY
                    )
            stats['total_val'] += len(val_imgs)
            stats['classes_with_val'] += 1
        
        # ---- 保存测试集 ----
        if test_imgs:
            test_dir = os.path.join(processed_root, "test", class_name)
            os.makedirs(test_dir, exist_ok=True)
            for i, img in enumerate(test_imgs):
                if img is not None:
                    Image.fromarray(img).save(
                        os.path.join(test_dir, f"{i:04d}.jpg"), quality=JPEG_QUALITY
                    )
            stats['total_test'] += len(test_imgs)
            stats['classes_with_test'] += 1
    
    # ---- 保存类别映射 ----
    mapping_path = os.path.join(processed_root, "class_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({
            "dataset_name": config.get('dataset', {}).get('name', 'OBWX'),
            "dataset_version": config.get('dataset', {}).get('version', 'v1.0'),
            "class_to_idx": class_to_idx,
            "idx_to_class": {v: k for k, v in class_to_idx.items()},
            "total_classes": stats['total_classes'],
            "total_train": stats['total_train'],
            "total_val": stats['total_val'],
            "total_test": stats['total_test']
        }, f, indent=2)
    
    # ---- 保存详细统计 ----
    stats_path = os.path.join(processed_root, "preprocess_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            'summary': stats,
            'class_stats': class_stats,
            'failed_files': all_failed_files
        }, f, indent=2)
    
    # ---- 如果有失败图片，单独保存文本文件方便查看 ----
    failed_txt_path = None
    if all_failed_files:
        failed_txt_path = os.path.join(processed_root, "failed_files.txt")
        with open(failed_txt_path, "w") as f:
            f.write("===== Failed Images Log =====\n\n")
            for class_name, files in all_failed_files.items():
                f.write(f"Class: {class_name} ({len(files)} failed)\n")
                for file_path, err in files:
                    f.write(f"  {file_path}: {err}\n")
                f.write("\n")
        logger.warning(f"Failed files logged to: {failed_txt_path}")
    
    # ---- 最终报告 ----
    print("\n" + "=" * 60)
    print("✅ Preprocessing completed!")
    print("=" * 60)
    print(f"Total classes:           {stats['total_classes']}")
    print(f"Training images:         {stats['total_train']}")
    print(f"Validation images:       {stats['total_val']}")
    print(f"Test images:             {stats['total_test']}")
    print(f"Classes all in train:    {stats['classes_all_train']}")
    print(f"Classes with val:        {stats['classes_with_val']}")
    print(f"Classes with test:       {stats['classes_with_test']}")
    print(f"Train failed images:     {stats['total_train_failed']}")
    print(f"Val failed images:       {stats['total_val_failed']}")
    print(f"Test failed images:      {stats['total_test_failed']}")
    print(f"Empty classes:           {stats['empty_classes']}")
    if failed_txt_path:
        print(f"Failed files logged:     {failed_txt_path}")
    print("=" * 60)


if __name__ == "__main__":
    # 固定所有随机种子
    random.seed(42)
    np.random.seed(42)
    #A.random.seed(42)
    preprocess()