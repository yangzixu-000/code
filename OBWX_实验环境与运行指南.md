# OBWX 实验环境与运行指南 / OBWX Experimental Environment & Run Guide

> 本文档说明 OBWX 古文字分类基准的**实验环境、项目结构、以及从数据预处理到模型训练与评测的完整执行步骤与命令**，供论文复现与第三方使用。  
> This document describes the **experimental environment, project structure, and the complete execution steps and commands** for the OBWX ancient-ideograph classification benchmark — from data preprocessing through model training and evaluation — for reproducibility.

---

## 1. 实验环境 / Experimental Environment


### 1.1 硬件 / Hardware

| 项目 / Item | 建议配置 / Recommended                      | 说明 / Notes                                                                                                                                                                                                     |
| --------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPU       | NVIDIA GPU，支持 CUDA 11.x/12.x，显存 ≥ 16 GB | 大模型（ViT‑B/16、Swin、MaxViT、ConvNeXt 系列等）建议 ≥ 16 GB 显存；显存不足时降低 `batch_size`。Code auto-falls back to CPU if no CUDA device is present, but full 14‑model training on CPU is impractical (single model takes days). |
| CPU       | ≥ 8 核（本项目预处理实测 28 核）                    | 用于数据预处理与（Windows 下默认的）单进程 DataLoader。                                                                                                                                                                          |
| 内存 / RAM  | ≥ 32 GB                                 | 6,650 类、7 万+ 图像预处理阶段峰值占用较高。                                                                                                                                                                                    |
| 磁盘 / Disk | ≥ 50 GB 可用空间                            | 原始数据 `data/obwx_raw` + 预处理产物 `data/obwx_processed` + 模型权重。                                                                                                                                                     |

> ⚠️ 若运行环境无 GPU，`train.py` 会退化为 CPU 模式（见 `training/train.py` 中 `device` 选择逻辑）。论文结论基于 **GPU 训练**得出，请勿用 CPU 跑完整基准。  
> If no GPU is available, `train.py` falls back to CPU. The paper's results were obtained on GPU; do **not** run the full benchmark on CPU.

### 1.2 软件依赖 / Software Dependencies

Python ≥ 3.8（PyTorch 2.x 要求）。核心依赖见 `requirements.txt`：

```text
torch>=2.0.0
torchvision>=0.15.0
albumentations>=1.3.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
pyyaml>=6.0
# 可选（噪声鲁棒性 / 配对 t 检验分析用）：scipy
```


### 1.3 关键环境变量 / Key Environment Variables

| 变量 / Variable            | 作用 / Purpose                                             | 默认值 / Default                                                |
| ------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ |
| `HF_ENDPOINT`            | 下载 timm 预训练权重镜像（国内可达）。`https://hf-mirror.com`            | 官方 `huggingface.co`（可能不可达）                                   |
| `OBWX_FORCE_NUM_WORKERS` | 显式设置 DataLoader `num_workers`（覆盖 Windows 强制 0 的行为），如 `8` | Windows 下默认 `0`；其他平台取 `config.yaml` 的 `training.num_workers` |
| `OBWX_ALLOW_RANDOM_INIT` | 设为 `1` 时允许权重下载失败后用随机初始化继续（否则 fail‑fast 报错）               | 未设置 → 权重缺失即报错退出                                              |

> 权重下载：所有骨干网络使用 **ImageNet 预训练权重**，由 `timm` 自动下载。若 `hf.co` 不可达，请先执行 `export HF_ENDPOINT=https://hf-mirror.com`（Linux/macOS）或 `set HF_ENDPOINT=https://hf-mirror.com`（Windows）。  
> All backbones use **ImageNet‑pretrained weights** fetched by `timm`. If `hf.co` is unreachable, set `HF_ENDPOINT=https://hf-mirror.com` first.

### 1.4 数据集 / Dataset

- **OBWX**：收录甲骨文、金文、战国文字、小篆四个历史时期的古文字字形图像，**原始 74,047 张 / 6,650 个类别**（论文口径）。
- 原始数据放置于 `data/obwx_raw/`，每个子目录对应一个类别。
- 经 `data/preprocess.py` 处理后生成 `data/obwx_processed/`，含 `train/`、`val/`、`test/` 三个划分及 `class_mapping.json`。


### 1.5 训练协议（论文 §3，统一基准） / Training Protocol (paper §3, unified benchmark)

| 配置 / Setting                | 取值 / Value                                                    |
| --------------------------- | ------------------------------------------------------------- |
| 输入尺寸 / Input size           | 224 × 224                                                     |
| 预训练 / Pretraining           | ImageNet                                                      |
| Batch size                  | 64                                                            |
| Epochs                      | 80                                                            |
| 随机种子 / Seed                 | 42（全局固定）                                                      |
| 优化器 / Optimizer             | AdamW，标签平滑 `label_smoothing=0.1`                              |
| 学习率策略 / LR schedule         | CosineWarmRestarts（`T_0=10, T_mult=2`）                        |
| 微调策略 / Fine‑tuning          | 14 个统一协议基线为 **全量微调**（`freeze_strategy="full"`）                |
| 分类头 / Head                  | `hidden_dims=[512, 256]`，`dropout_rates=[0.2, 0.1]`，激活 `gelu` |
| 骨干学习率缩放 / Backbone LR scale | `lr_scale_backbone=1.0`（统一协议基线）                               |

> 注：注册表中 5 个早期模型（ResNet18、EfficientNet‑B3、DenseNet121、ViT‑B/16、Swin Transformer）保留各自调好的 `lr_scale_backbone`（0.1–0.2）；论文报告的 **14 个基线**均遵循上述统一协议。  
> Note: 5 legacy models keep their own tuned `lr_scale_backbone`; the **14 baselines** reported in the paper follow the unified protocol above.

---


## 2. 项目结构 / Project Structure

```text
OBWX/
├── config.yaml              # 全局配置：数据路径、训练超参、增强、模型专项配置、random_seed 等
├── requirements.txt         # Python 依赖
├── README.md                # 快速上手
├── data/
│   ├── obwx_raw/           # 原始数据集（每子目录一个类别）；OBWX.zip 为原始压缩包
│   ├── obwx_processed/     # 预处理产物：train/ val/ test/ + class_mapping.json + preprocess_stats.json
│   ├── dataset.py          # OBWXDataset + get_dataloaders()（统一增强、DDP 支持、R1 防呆校验）
│   └── preprocess.py       # 数据预处理与划分（白底黑字标准化、去重、长尾增强、70/15/15 划分）
├── models/
│   ├── __init__.py         # 模型注册表（24 个可训练分类器）
│   ├── cnn_models.py / modern_cnns.py / transformer_models.py /
│   │   hybrid_models.py / lightweight_models.py / timm_models.py / iformer_models.py
├── training/
│   ├── benchmark.py        # 主入口：批量训练全部/指定模型，生成排名与曲线
│   ├── train.py            # 单模型训练循环（含 DDP/DP、优化器/调度器、测试评测）
│   ├── multi_seed.py       # 多种子重复训练 + 配对 t 检验（论文 §4.7）
│   ├── analyze_best.py     # 最优模型误差结构分析（频段/混淆对，论文 §4.4）
│   └── complexity.py       # 参数量 / MACs 统计
├── utils/
│   ├── config.py           # 配置加载 + validate_processed_root()（R1 数据根防呆校验）
│   └── seed.py             # 全局随机种子固定（Python/NumPy/Torch/CUDA/cuDNN/worker）
├── output/
│   ├── models/             # 训练权重 best_<safe_name>.pth
│   ├── results/            # benchmark_summary.csv / detailed_results.json / result_<model>.json / validation_curves.png
│   └── logs/
└── tools_*.py              # 论文落稿/分析辅助脚本（绘图、噪声鲁棒性、引用核查等）
```

---

## 3. 详细执行步骤与命令 / Detailed Execution Steps & Commands

> 所有命令均**从项目根目录 `OBWX/` 运行**（项目中 `training/*` 会自动把根目录加入 `sys.path`，但从根目录运行是既定规范）。  
> Run all commands **from the project root `OBWX/`** (the scripts auto‑resolve the root, but running from root is the prescribed convention).

### 步骤 1 · 环境准备 / Step 1 · Environment Setup

```bash
# 创建并激活虚拟环境（推荐）/ create & activate a venv (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 安装依赖 / install dependencies
pip install -r requirements.txt

# （可选）设置权重镜像，避免 hf.co 不可达 / optional: set weight mirror
export HF_ENDPOINT=https://hf-mirror.com     # Windows: set HF_ENDPOINT=https://hf-mirror.com
```

### 步骤 2 · 数据预处理 / Step 2 · Data Preprocessing

将原始数据放入 `data/obwx_raw/`（如仅有 `OBWX.zip`，先解压）。然后运行：

```bash
python data/preprocess.py
```

**该脚本会执行 / The script will**：

1. 将图像统一标准化为「白底黑字」（全局均值或四角均值偏暗则反转一次，仅一次）。
2. 对扩展名大小写与文件**内容（md5）两级去重**，杜绝 train/val/test 跨子集泄漏。
3. 按类别划分（见 §4 规则），小样本类别离线几何增强至 ≥ 15 张。
4. 输出 `data/obwx_processed/{train,val,test}/<class>/` 与 `class_mapping.json`、`preprocess_stats.json`。

### 步骤 3 · 基准训练 / Step 3 · Benchmark Training

```bash
# 训练注册表中全部 24 个模型（已训练过的会自动跳过）/ train all 24 registered models (skips trained)
CUDA_LAUNCH_BLOCKING=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3 python training/benchmark.py

# 仅训练指定模型 / train specific models
CUDA_LAUNCH_BLOCKING=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3 python training/benchmark.py --models "EfficientNetV2-S" "MaxViT-T" "ConvNeXt-Tiny"

# 强制重训全部 / force retrain all
python training/benchmark.py --retrain

# 列出所有可训练模型 / list all models
python training/benchmark.py --list
```

- 启动即执行 **R1 防呆校验**（`validate_processed_root`）：核对 `data/obwx_processed` 的类别数（6,650）、测试集类别数（3,447）、测试集样本下限（≥12,000）及训练集空文件抽查；不匹配直接报错，绝不静默训在错误构建上。
- 训练完成后在 `output/results/` 生成 `benchmark_summary.csv`、`detailed_results.json`、`result_<model>.json` 与 `validation_curves.png`。

### 步骤 4 · 多种子实验 / Step 4 · Multi‑seed Experiments

```bash
# 默认：4 个核心模型 × 种子 {42, 123, 2024} / default: 4 core models × {42,123,2024}
python training/multi_seed.py

# 单模型冒烟测试（3 种子、2 epoch）/ smoke test a single model
python training/multi_seed.py --models "EfficientNetV2-S" --epochs 2

# 自定义种子 / custom seeds
python training/multi_seed.py --seeds 42 123 2024 7 2026
```

每个 `(模型, 种子)` 组合独立重建模型与 DataLoader（保证不同种子数据顺序确实不同），权重按种子分目录保存，并在模型间做配对 t 检验（种子数 ≥ 2 可用，但 3 种子功效有限，论文中已如实说明）。  
Each `(model, seed)` pair rebuilds the model and DataLoader independently; checkpoints are saved per‑seed; paired t‑tests are reported between models.

### 步骤 5 · 误差结构与噪声鲁棒性分析 / Step 5 · Error‑structure & Noise‑robustness Analysis

```bash
# 最优模型误差结构（频段性能 + Top‑10 易混淆字头对，论文 §4.4）
python training/analyze_best.py                         # 默认 EfficientNetV2-S
python training/analyze_best.py --model "MaxViT-T" --no-figures

# 噪声鲁棒性评测（论文 §4.5；在 CPU 上运行，自动筛选 6650 类检查点）
python tools_noise_robustness.py --limit 400           # 校验 + 计时
python tools_noise_robustness.py                       # 全量后台运行
```

`tools_noise_robustness.py` 对发布的全量检查点施加高斯 / 椒盐 / 笔画侵蚀三类受控扰动，量化准确率退化；同一 `(noise, level)` 下所有模型共享同一随机种子以保证噪声图案可比。

### 步骤 6 · 结果输出位置 / Step 6 · Output Locations

| 产物 / Artifact            | 路径 / Path                                                                   |
| ------------------------ | --------------------------------------------------------------------------- |
| 训练权重 / Weights           | `output/models/best_<safe_name>.pth`                                        |
| 汇总排名 / Summary           | `output/results/benchmark_summary.csv`                                      |
| 详细结果 / Detailed          | `output/results/detailed_results.json`、`output/results/result_<model>.json` |
| 验证曲线 / Curves            | `output/results/validation_curves.png`                                      |
| 数据映射 / Mapping           | `data/obwx_processed/class_mapping.json`                                    |
| 预处理统计 / Preprocess stats | `data/obwx_processed/preprocess_stats.json`                                 |

---

## 4. 重要规则与注意事项 / Important Rules & Notes

### 4.1 随机种子与确定性 / Random Seed & Determinism

- 全局种子 `random_seed: 42`（`config.yaml`），由 `utils/seed.py:set_seed()` 统一固定 Python / NumPy / Torch / CUDA / cuDNN。
- `torch.backends.cudnn.deterministic=True`、`benchmark=False`，并启用 `use_deterministic_algorithms`（可行时）。
- DataLoader worker 由 `worker_init_fn` 按 `base_seed(42)+worker_id` 独立播种。
- 论文多种子实验通过 `set_seed(seed)` 在每轮前重置，确保可复现。

### 4.2 数据划分阈值（务必与论文一致） / Data‑split Threshold (must match the paper)

- **原始样本数 ≤ 3 张的类别：全部归入训练集**（代码 `preprocess.py` 中为 `original_count <= 3`，含 3 张）。
- 原始样本数 ≥ 4 张的类别：按 **70 / 15 / 15** 随机划分 train/val/test；小样本训练集离线增强至 ≥ 15 张；val/test 仅用原始图、不做增强。
- 论文表述为「at most three / 至多 3 张」，不可写为「不足 3 张 / fewer than three」。

### 4.3 数据增强 / Data Augmentation

- **在线增强（仅训练集）**：Resize 224×224 → 随机旋转 ±5°（`p=0.3`，白色填充避免黑边）→ 亮度/对比度抖动 ±10%（`p=0.3`）→ **无水平翻转**（古文字具方向性）→ ImageNet 归一化。
- **离线增强（preprocess 阶段，仅用于补齐小样本类别）**：`LongestMaxSize(256)→Pad(256, 白)→RandomCrop(224)` + `Affine(scale 0.98–1.02, translate 1%, rotate ±1°, shear ±0.5°, p=1.0)`，无翻转、无归一化。
- 验证/测试：**仅 Resize + ImageNet 归一化**，无任何增强。

### 4.4 Windows 与 num_workers / Windows & num_workers

- Windows 下 `num_workers` 强制为 `0`（规避多进程 spawn 兼容问题）；可用 `OBWX_FORCE_NUM_WORKERS=N` 显式覆盖以加速数据加载（GPU 机器上常是瓶颈）。
- 若未开启该变量，则增广随机采样在种子间真实变化，§4.7「增广的随机采样」表述成立。

### 4.5 防呆与失败快速退出 / Fail‑fast Guards

- `validate_processed_root()`：数据根类别数/测试集类别数/样本下限/空文件任一不符即抛错。
- 权重下载失败默认 fail‑fast；需随机初始化兜底时设 `OBWX_ALLOW_RANDOM_INIT=1`（前几 epoch 个位数准确率 = 随机初始化征兆）。
- 单图损坏时在 `OBWXDataset.__getitem__` 中自动跳到下一张，不中断训练。

### 4.6 模型注册表 / Model Registry

- `models/__init__.py` 注册 **24 个**可训练分类器；`training/benchmark.py:MODELS_CONFIG` 给出各自的 `freeze_strategy`、分类头、优化器与调度器。
- 论文主实验报告 **14 个基线**（EffNetV2‑S、MaxViT‑T、Swin、PVTv2‑B2、EffFormerV2‑S、EfficientNet‑B3、ViT‑B/16、InceptionNeXt、ConvNeXt‑V2、MobileNetV3‑L、MobileViT‑V2、RepViT、WideResNet‑50、ResNeXt‑50）。
- 注：CMT 从未被 timm 收录，已用 PVTv2‑B2 替代；iFormer 仅 timm `main` 分支提供；CoAtNet‑Tiny 无预训练权重已被移出 14 名单。

---


## 5. 一键复现建议流程 / Suggested One‑shot Reproduction Flow

```bash
# 0) 环境 / environment
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com

# 1) 解压原始数据（如仅有 zip）/ unpack raw data if only zip present
#    unzip data/OBWX.zip -d data/obwx_raw   (确保 data/obwx_raw/<class>/ 结构)

# 2) 预处理 / preprocess
python data/preprocess.py

# 3) 基准训练（14 基线，可分批用 --models 指定）/ benchmark (14 baselines; use --models to batch)
python training/benchmark.py --models "EfficientNetV2-S" "MaxViT-T" "Swin Transformer" \
    "PVTv2-B2" "EfficientFormerV2-S" "EfficientNet-B3" "ViT-B/16" "InceptionNeXt-Small" \
    "ConvNeXt-V2-Tiny" "MobileNetV3-Large" "MobileViT-V2-S" "RepViT-M1.1" \
    "WideResNet-50" "ResNeXt-50"

# 4) 多种子 / multi-seed
python training/multi_seed.py

# 5) 分析 / analysis
python training/analyze_best.py --model "EfficientNetV2-S"
python tools_noise_robustness.py
```

> 上述流程覆盖论文 §3–§4 的全部主实验（基准排名、统计显著性、误差结构、噪声鲁棒性、多种子稳定性）。  
> The above covers all main experiments in paper §3–§4 (benchmark ranking, statistical significance, error structure, noise robustness, multi‑seed stability).
