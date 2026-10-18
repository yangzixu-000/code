"""
随机种子工具
保证实验完全可复现（涵盖 Python、NumPy、Torch、CUDA、DataLoader workers）

核心策略：
    1. 固定 Python 内置 hash 随机（PYTHONHASHSEED）
    2. 固定 random / numpy / torch 种子
    3. 固定 CUDA 随机种子（所有 GPU）
    4. 设置 cuDNN 确定性模式
    5. DataLoader workers 独立种子（包含 torch）
    6. 支持确定性模式切换
"""

import os
import random
import sys
import numpy as np
import torch


def _set_python_hash_seed(seed: int = 42):
    """
    固定 Python 内置 hash() 随机性
    影响：dict 遍历顺序、set 顺序等
    必须在 Python 解释器启动时设置，运行时设置仅影响后续行为
    """
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_seed(seed: int = 42):
    """
    固定所有随机种子，保证实验完全可复现
    
    Args:
        seed: 随机种子值
    
    覆盖范围：
        - Python 内置 hash 随机
        - Python random 模块
        - NumPy 随机
        - PyTorch CPU 随机
        - PyTorch CUDA 随机（所有 GPU）
        - cuDNN 确定性（牺牲少量性能）
    
    注意：
        - 某些 CUDA 算子（如某些 reduction）在极端情况下仍可能非确定
        - 如需最大化性能，可使用 set_deterministic_mode(False)
    """
    # 1. Python 内置 hash 随机
    _set_python_hash_seed(seed)
    
    # 2. Python random
    random.seed(seed)
    
    # 3. NumPy
    np.random.seed(seed)
    
    # 4. PyTorch CPU
    torch.manual_seed(seed)
    
    # 5. PyTorch CUDA（所有 GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 确定性模式（牺牲少量性能换取完全确定性）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 额外：限制 CUDA 随机算法（可选，某些算子需额外设置）
        # 注意：torch.use_deterministic_algorithms(True) 在 PyTorch 1.8+ 可用
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError as e:
                # 某些操作可能不支持确定性，仅警告不中断
                print(f"⚠️ Warning: {e}", file=sys.stderr)
        
        print(f"🔒 Random seed set to {seed} (deterministic mode, CUDA)")
    else:
        print(f"🔒 Random seed set to {seed} (CPU mode)")


def worker_init_fn(worker_id: int):
    """
    DataLoader worker 的随机种子初始化函数
    
    用法:
        train_loader = DataLoader(..., worker_init_fn=worker_init_fn)
    
    说明：
        - 每个 worker 获得独立种子（base_seed + worker_id）
        - 覆盖 random, numpy, torch 三种随机源
        - 保证数据加载和增强的完全可复现
    
    注意：
        - 此函数会在每个 worker 子进程中执行
        - 基础种子通过闭包从外部传入（此处固定为 42）
        - 如需自定义基础种子，使用 worker_init_fn_factory
    """
    base_seed = 42
    worker_seed = base_seed + worker_id
    
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    # 如果 worker 内部使用 CUDA（通常不建议），也固定
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)


def worker_init_fn_factory(base_seed: int = 42):
    """
    创建绑定指定基础种子的 worker 初始化函数
    
    Args:
        base_seed: 基础种子值，应与 set_seed 保持一致
    
    Returns:
        worker_init_fn: 绑定了基础种子的 worker 函数
    
    用法:
        from utils.seed import worker_init_fn_factory
        worker_fn = worker_init_fn_factory(config['random_seed'])
        train_loader = DataLoader(..., worker_init_fn=worker_fn)
    """
    def _worker_init_fn(worker_id: int):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)
    
    return _worker_init_fn


def set_deterministic_mode(enable: bool = True):
    """
    切换 PyTorch cuDNN 确定性模式
    
    Args:
        enable: True 启用确定性（牺牲性能），False 启用性能模式
    
    影响：
        - 确定性模式：结果完全可复现，速度稍慢
        - 性能模式：速度最快，结果可能略有波动
    """
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping cuDNN settings.")
        return
    
    torch.backends.cudnn.deterministic = enable
    torch.backends.cudnn.benchmark = not enable
    
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(enable)
        except RuntimeError as e:
            print(f"⚠️ Warning: {e}", file=sys.stderr)
    
    mode = "deterministic" if enable else "performance"
    print(f"🔧 CUDNN set to {mode} mode")


def reset_seed(seed: int = 42):
    """
    重置所有随机种子（同 set_seed，用于实验中途重新初始化）
    """
    set_seed(seed)