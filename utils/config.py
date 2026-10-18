"""
配置加载工具
支持：
    - 统一使用 pathlib API
    - 健壮的异常捕获（YAMLError, KeyError, PermissionError）
    - 多路径查找（支持项目结构调整）
    - 友好的错误提示
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


def _find_config_file(config_path: Optional[Path] = None) -> Path:
    """
    查找配置文件，支持多种路径
    
    查找顺序：
        1. 用户指定的路径
        2. 当前文件所在目录的父目录（默认）
        3. 当前工作目录
        4. 当前工作目录的父目录
    """
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            return config_path
    
    # 默认：项目根目录（utils/ 的父目录）
    default_path = Path(__file__).parent.parent / "config.yaml"
    if default_path.exists():
        return default_path
    
    # 兜底：当前工作目录
    cwd_path = Path.cwd() / "config.yaml"
    if cwd_path.exists():
        return cwd_path
    
    # 再兜底：当前工作目录的父目录
    parent_path = Path.cwd().parent / "config.yaml"
    if parent_path.exists():
        return parent_path
    
    raise FileNotFoundError(
        f"Configuration file 'config.yaml' not found.\n"
        f"Searched locations:\n"
        f"  - {default_path}\n"
        f"  - {cwd_path}\n"
        f"  - {parent_path}\n"
        f"  - (user specified: {config_path})"
    )


def _safe_get(config: Dict[str, Any], keys: list, default=None):
    """
    安全读取嵌套字典键，返回默认值或抛出友好错误
    """
    for key in keys:
        if not isinstance(config, dict):
            return default
        if key not in config:
            raise KeyError(
                f"Missing required configuration key: '{key}'\n"
                f"Full path: {' -> '.join(keys)}\n"
                f"Available keys at this level: {list(config.keys()) if isinstance(config, dict) else 'not a dict'}"
            )
        config = config[key]
    return config


def validate_processed_root(processed_root: str, expected: Optional[Dict[str, Any]] = None) -> None:
    """
    R1 防呆校验：确认 processed_root 指向完整、符合协议的数据构建。

    背景：历史上 config.yaml 的 processed_root 曾指向不完整的中间构建
    （train 仅 241 类），且目录中存在多个历史构建（_corrupt / _vA_backup /
    _partial / _biased 等），直接跑 benchmark 会静默训在错误的数据上。
    本函数在构建 DataLoader 前做 fail-fast 校验，不匹配即抛出，绝不静默回退。

    Args:
        processed_root: data.processed_root
        expected: data.expected（total_classes / test_classes / min_test_samples），
                  缺省或未给出具体项时跳过对应检查。

    Raises:
        FileNotFoundError: 缺少 class_mapping.json 或 test 目录
        ValueError: 与期望值不符
    """
    expected = expected or {}
    root = Path(processed_root)

    mapping_path = root / "class_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"[R1 guard] {processed_root} 下缺少 class_mapping.json —— 该目录不是有效的数据构建。"
        )
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    total_classes = mapping.get("total_classes")
    if total_classes is None:
        total_classes = len(mapping.get("class_to_idx", {}))

    exp_total = expected.get("total_classes")
    if exp_total is not None and int(total_classes) != int(exp_total):
        raise ValueError(
            f"[R1 guard] 数据构建不匹配：{processed_root} 共 {total_classes} 类，"
            f"期望 {exp_total} 类。请检查 config.yaml 的 data.processed_root "
            f"（当前目录下可能残留 _partial / _biased / _corrupt 等历史构建）。"
        )

    test_dir = root / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"[R1 guard] 缺少测试集目录：{test_dir}")

    test_classes = sum(1 for p in test_dir.iterdir() if p.is_dir())
    exp_test_classes = expected.get("test_classes")
    if exp_test_classes is not None and test_classes != int(exp_test_classes):
        raise ValueError(
            f"[R1 guard] 测试集类别数不匹配：{test_dir} 含 {test_classes} 类，"
            f"期望 {exp_test_classes} 类（§3.2 协议：原始样本不足 3 张的类别全部归入训练集）。"
        )

    min_samples = expected.get("min_test_samples")
    if min_samples is not None:
        n = sum(len(list(c.iterdir())) for c in test_dir.iterdir() if c.is_dir())
        if n < int(min_samples):
            raise ValueError(
                f"[R1 guard] 测试集样本过少：{test_dir} 仅 {n} 张，期望 ≥ {min_samples} 张。"
            )

    # 空文件抽查：2026-10-02 曾出现整份重建的 train/val 全为 0 字节空文件，
    # 目录结构与元数据都正常、唯独文件内容为空——只看文件数无法发现，故抽查实际字节数。
    probe_n = expected.get("train_nonempty_probe_classes")
    if probe_n:
        train_dir = root / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"[R1 guard] 缺少训练集目录：{train_dir}")
        class_dirs = sorted(p for p in train_dir.iterdir() if p.is_dir())
        step = max(1, len(class_dirs) // int(probe_n))
        probed = class_dirs[::step][: int(probe_n)]
        empty = []
        for cd in probed:
            files = [f for f in cd.iterdir() if f.is_file()]
            if not files or all(f.stat().st_size == 0 for f in files):
                empty.append(cd.name)
        if empty:
            raise ValueError(
                f"[R1 guard] 训练集存在空文件类别（抽查 {len(probed)} 个类别，"
                f"{len(empty)} 个为空）：{empty[:5]} ... "
                f"该数据构建不可用，请改用 ./data/obwx_processed_clean 或重新预处理。"
            )

    print(f"✅ [R1 guard] 数据根校验通过：{processed_root} "
          f"({total_classes} 类 / 测试 {test_classes} 类)")


def get_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    加载配置文件并创建输出目录
    
    Args:
        config_path: 配置文件路径，默认自动查找
    
    Returns:
        dict: 配置字典
    
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: 配置文件格式错误
        KeyError: 配置缺少必要字段
        PermissionError: 无法创建输出目录
        OSError: 其他系统错误
    """
    try:
        config_path = _find_config_file(config_path)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    
    # 加载 YAML
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(
            f"❌ YAML parsing error in {config_path}:\n"
            f"   {e}\n"
            f"   Please check the syntax of your config.yaml file.",
            file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading config file {config_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    if config is None:
        print(f"❌ Config file {config_path} is empty.", file=sys.stderr)
        sys.exit(1)
    
    # 安全读取必要配置并创建目录
    try:
        # 使用安全读取获取路径
        output_paths = {
            'model_save_path': _safe_get(config, ['output', 'model_save_path']),
            'results_path': _safe_get(config, ['output', 'results_path']),
            'logs_path': _safe_get(config, ['output', 'logs_path']),
        }
        data_paths = {
            'raw_root': _safe_get(config, ['data', 'raw_root']),
            'processed_root': _safe_get(config, ['data', 'processed_root']),
        }
        
        # 创建所有目录
        for name, path in {**output_paths, **data_paths}.items():
            if path:
                try:
                    Path(path).mkdir(parents=True, exist_ok=True)
                except PermissionError as e:
                    print(
                        f"❌ Permission denied while creating directory: {path}\n"
                        f"   {e}\n"
                        f"   Please check write permissions for this path.",
                        file=sys.stderr
                    )
                    sys.exit(1)
                except OSError as e:
                    print(
                        f"❌ Failed to create directory: {path}\n"
                        f"   {e}",
                        file=sys.stderr
                    )
                    sys.exit(1)
    
    except KeyError as e:
        print(
            f"❌ {e}\n"
            f"   Please ensure your config.yaml has the required structure.\n"
            f"   Required sections:\n"
            f"     - output.model_save_path\n"
            f"     - output.results_path\n"
            f"     - output.logs_path\n"
            f"     - data.raw_root\n"
            f"     - data.processed_root",
            file=sys.stderr
        )
        sys.exit(1)
    
    # 打印配置加载信息
    print(f"✅ Config loaded from: {config_path}")
    
    return config