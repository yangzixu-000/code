# OBWX: A Benchmark for Classification of Ancient Chinese Ideographs from Four Historical Periods

> Training and evaluation code for the OBWX benchmark — deep-learning classification of ancient Chinese ideographs across four historical periods (oracle bone, bronze, warring-states, small seal). This repository reproduces all main experiments in the companion paper.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Experimental Environment](#experimental-environment)
  - [Hardware](#hardware)
  - [Software Dependencies](#software-dependencies)
  - [Key Environment Variables](#key-environment-variables)
  - [Training Protocol](#training-protocol-paper-section-3-unified-benchmark)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Execution Steps](#execution-steps)
  - [Step 1 — Environment Setup](#step-1--environment-setup)
  - [Step 2 — Data Preprocessing](#step-2--data-preprocessing)
  - [Step 3 — Benchmark Training](#step-3--benchmark-training)
  - [Step 4 — Multi-seed Experiments](#step-4--multi-seed-experiments)
  - [Step 5 — Error-structure & Noise-robustness Analysis](#step-5--error-structure--noise-robustness-analysis)
  - [Step 6 — Output Locations](#step-6--output-locations)
- [Important Rules & Notes](#important-rules--notes)
- [Suggested One-shot Reproduction Flow](#suggested-one-shot-reproduction-flow)
- [Citation](#citation)
- [License](#license)

---

## Overview

OBWX is a large-scale benchmark for classifying ancient Chinese ideographs (古文字) into 6,650 character classes spanning four historical periods. The codebase provides:

- A reproducible preprocessing pipeline (white-background/black-foreground normalization, dedup, long-tail augmentation, 70/15/15 split).
- A unified training entry point (`training/benchmark.py`) covering 24 registered models, with 14 reported as the main baselines.
- Multi-seed training with paired t-tests (paper §4.7).
- Error-structure analysis (frequency-band performance, Top-10 confusing ideograph pairs; §4.4) and controlled noise-robustness evaluation (§4.5).

All commands are run from the project root `OBWX/`.

---

## Dataset

- **OBWX** collects ancient-ideograph glyph images from four historical periods (oracle bone, bronze, warring-states, small seal): **74,047 images across 6,650 classes** (paper count).
- Raw data is placed in `data/obwx_raw/`, with one subdirectory per class.
- Running `data/preprocess.py` produces `data/obwx_processed/` containing the `train/`, `val/`, `test/` splits and `class_mapping.json`.

---

## Experimental Environment

### Hardware

| Item | Recommended | Notes |
|------|-------------|-------|
| GPU | NVIDIA GPU, CUDA 11.x/12.x, VRAM ≥ 16 GB | Large backbones (ViT-B/16, Swin, MaxViT, ConvNeXt family, etc.) benefit from ≥ 16 GB VRAM; reduce `batch_size` if VRAM is insufficient. Code auto-falls back to CPU when no CUDA device is present, but full 14-model training on CPU is impractical (a single model takes days). |
| CPU | ≥ 8 cores (28 cores used in our preprocessing) | Used for data preprocessing and (by default on Windows) single-process DataLoader. |
| RAM | ≥ 32 GB | Peak memory is high during preprocessing of 6,650 classes and 70k+ images. |
| Disk | ≥ 50 GB free | Raw data `data/obwx_raw` + processed `data/obwx_processed` + model weights. |

> ⚠️ **If no GPU is available, `train.py` falls back to CPU.** The paper's results were obtained on GPU; do **not** run the full benchmark on CPU.

### Software Dependencies

Python ≥ 3.8 (required by PyTorch 2.x). Core dependencies are listed in `requirements.txt`:

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
# optional (noise robustness / paired t-test analysis): scipy
```

### Key Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `HF_ENDPOINT` | Mirror for downloading timm pretrained weights (reachable in China). Use `https://hf-mirror.com` | Official `huggingface.co` (may be unreachable) |
| `OBWX_FORCE_NUM_WORKERS` | Explicitly set DataLoader `num_workers` (overrides the Windows forced-0 behaviour), e.g. `8` | `0` on Windows; elsewhere from `config.yaml` `training.num_workers` |
| `OBWX_ALLOW_RANDOM_INIT` | Set to `1` to allow training to continue with random init if weight download fails (otherwise fail-fast) | Unset → missing weights abort with an error |

> All backbones use **ImageNet-pretrained weights** fetched by `timm`. If `hf.co` is unreachable, set `HF_ENDPOINT=https://hf-mirror.com` first (`export` on Linux/macOS, `set` on Windows).

### Training Protocol (paper Section 3, unified benchmark)

| Setting | Value |
|---------|-------|
| Input size | 224 × 224 |
| Pretraining | ImageNet |
| Batch size | 64 |
| Epochs | 80 |
| Seed | 42 (globally fixed) |
| Optimizer | AdamW, `label_smoothing=0.1` |
| LR schedule | CosineWarmRestarts (`T_0=10, T_mult=2`) |
| Fine-tuning | 14 unified-protocol baselines use **full fine-tuning** (`freeze_strategy="full"`) |
| Head | `hidden_dims=[512, 256]`, `dropout_rates=[0.2, 0.1]`, activation `gelu` |
| Backbone LR scale | `lr_scale_backbone=1.0` (unified-protocol baselines) |

> Note: 5 legacy models keep their own tuned `lr_scale_backbone`; the **14 baselines** reported in the paper follow the unified protocol above.

---

## Project Structure

```text
OBWX/
├── config.yaml              # Global config: data paths, training hyper-params, augmentation,
│                           #   model-specific config, random_seed, etc.
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── data/
│   ├── obwx_raw/           # Raw dataset (one subdir per class); OBWX.zip is the raw archive
│   ├── obwx_processed/     # Preprocess output: train/ val/ test/ + class_mapping.json + preprocess_stats.json
│   ├── dataset.py          # OBWXDataset + get_dataloaders() (unified augmentation, DDP, R1 guard)
│   └── preprocess.py       # Preprocessing & split (white-bg/black-fg normalization, dedup, long-tail aug, 70/15/15)
├── models/
│   ├── __init__.py         # Model registry (24 trainable classifiers)
│   ├── cnn_models.py / modern_cnns.py / transformer_models.py /
│   │   hybrid_models.py / lightweight_models.py / timm_models.py / iformer_models.py
├── training/
│   ├── benchmark.py        # Entry point: batch-train all/specified models, generate ranking & curves
│   ├── train.py            # Single-model training loop (DDP/DP, optimizer/scheduler, test eval)
│   ├── multi_seed.py       # Multi-seed repeat training + paired t-test (paper Section 4.7)
│   ├── analyze_best.py     # Best-model error-structure analysis (bands / confusion pairs, Section 4.4)
│   └── complexity.py       # Parameter count / MACs statistics
├── utils/
│   ├── config.py           # Config loading + validate_processed_root() (R1 data-root guard)
│   └── seed.py             # Global random-seed fixing (Python/NumPy/Torch/CUDA/cuDNN/worker)
├── output/
│   ├── models/             # Trained weights best_<safe_name>.pth
│   ├── results/            # benchmark_summary.csv / detailed_results.json / result_<model>.json / validation_curves.png
│   └── logs/
└── tools_*.py              # Paper-writing / analysis aids (plotting, noise robustness, reference checks, etc.)
```

---

## Installation

```bash
# create & activate a venv (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# (optional) set weight mirror to avoid hf.co unreachable
export HF_ENDPOINT=https://hf-mirror.com     # Windows: set HF_ENDPOINT=https://hf-mirror.com
```

---

## Execution Steps

### Step 1 — Environment Setup

```bash
# create & activate a venv (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# (optional) set weight mirror to avoid hf.co unreachable
export HF_ENDPOINT=https://hf-mirror.com     # Windows: set HF_ENDPOINT=https://hf-mirror.com
```

### Step 2 — Data Preprocessing

Place the raw data into `data/obwx_raw/` (if only `OBWX.zip` is available, unzip it first). Then run:

```bash
python data/preprocess.py
```

The script will:

1. Normalize every image to white-background/black-foreground (invert once if globally or corner-mean dark).
2. Deduplicate by two levels: filename-extension case and file content (md5) to prevent train/val/test leakage.
3. Split by class (see [Data-split Threshold](#42-data-split-threshold-must-match-the-paper)); offline geometric augmentation lifts small-sample classes to ≥ 15 images.
4. Output `data/obwx_processed/{train,val,test}/<class>/` plus `class_mapping.json` and `preprocess_stats.json`.

### Step 3 — Benchmark Training

```bash
# train all 24 registered models (already-trained ones are skipped automatically)
python training/benchmark.py

# train only specified models
python training/benchmark.py --models "EfficientNetV2-S" "MaxViT-T" "ConvNeXt-Tiny"

# force retrain all
python training/benchmark.py --retrain

# list all trainable models
python training/benchmark.py --list
```

- On launch it runs the **R1 data-root guard** (`validate_processed_root`): verifies class count (6,650), test-class count (3,447), test-sample lower bound (≥12,000) and a spot-check for empty training files; mismatches abort immediately rather than silently training on a wrong build.
- After training, `output/results/` contains `benchmark_summary.csv`, `detailed_results.json`, `result_<model>.json` and `validation_curves.png`.

### Step 4 — Multi-seed Experiments

```bash
# default: 4 core models x seeds {42, 123, 2024}
python training/multi_seed.py

# smoke-test a single model (3 seeds, 2 epochs)
python training/multi_seed.py --models "EfficientNetV2-S" --epochs 2

# custom seeds
python training/multi_seed.py --seeds 42 123 2024 7 2026
```

Each `(model, seed)` pair rebuilds the model and DataLoader independently (ensuring the data order genuinely differs across seeds); checkpoints are saved per seed; paired t-tests are reported between models. (With ≥ 2 seeds the test is applicable, but 3 seeds offer limited power, as stated honestly in the paper.)

### Step 5 — Error-structure & Noise-robustness Analysis

```bash
# best-model error structure (band performance + Top-10 confusing ideograph pairs, Section 4.4)
python training/analyze_best.py                         # default EfficientNetV2-S
python training/analyze_best.py --model "MaxViT-T" --no-figures

# noise-robustness evaluation (Section 4.5; runs on CPU, auto-selects 6650-class checkpoints)
python tools_noise_robustness.py --limit 400           # validation + timing
python tools_noise_robustness.py                       # full background run
```

`tools_noise_robustness.py` applies three controlled perturbations (Gaussian / salt-and-pepper / stroke erosion) to the released full checkpoints and quantifies accuracy degradation; within a given `(noise, level)` all models share the same random seed so noise patterns are comparable.

### Step 6 — Output Locations

| Artifact | Path |
|----------|------|
| Weights | `output/models/best_<safe_name>.pth` |
| Summary ranking | `output/results/benchmark_summary.csv` |
| Detailed results | `output/results/detailed_results.json`, `output/results/result_<model>.json` |
| Validation curves | `output/results/validation_curves.png` |
| Class mapping | `data/obwx_processed/class_mapping.json` |
| Preprocess stats | `data/obwx_processed/preprocess_stats.json` |

---

## Important Rules & Notes

### 4.1 Random Seed & Determinism

- Global seed `random_seed: 42` (`config.yaml`), fixed uniformly across Python / NumPy / Torch / CUDA / cuDNN by `utils/seed.py:set_seed()`.
- `torch.backends.cudnn.deterministic=True`, `benchmark=False`, and `use_deterministic_algorithms` is enabled where feasible.
- DataLoader workers are seeded independently via `worker_init_fn` using `base_seed(42) + worker_id`.
- The paper's multi-seed experiments reset via `set_seed(seed)` before each run to ensure reproducibility.

### 4.2 Data-split Threshold (must match the paper)

- Classes with **original sample count ≤ 3**: the ENTIRE class goes to the training set (code in `preprocess.py` is `original_count <= 3`, inclusive of 3).
- Classes with ≥ 4 original samples: randomly split train/val/test by **70 / 15 / 15**; small-sample training sets are offline-augmented to ≥ 15 images; val/test use only original images with no augmentation.
- The paper states **"at most three"** — do NOT write "fewer than three".

### 4.3 Data Augmentation

- **Online augmentation (training set only)**: Resize 224×224 → random rotation ±5° (`p=0.3`, white-fill to avoid black borders) → brightness/contrast jitter ±10% (`p=0.3`) → **NO horizontal flip** (ideographs are orientation-sensitive) → ImageNet normalization.
- **Offline augmentation** (preprocess stage, only to fill small-sample classes): `LongestMaxSize(256)→Pad(256, white)→RandomCrop(224)` + `Affine(scale 0.98–1.02, translate 1%, rotate ±1°, shear ±0.5°, p=1.0)`; no flip, no normalization.
- **Validation/test**: Resize + ImageNet normalization only, no augmentation.

### 4.4 Windows & num_workers

- On Windows, `num_workers` is forced to `0` (to avoid multiprocessing spawn issues); use `OBWX_FORCE_NUM_WORKERS=N` to explicitly override and speed up data loading (often the bottleneck on GPU machines).
- If this variable is not set, the augmentation random sampling genuinely varies across seeds, so the §4.7 "stochastic augmentation draw" statement holds.

### 4.5 Fail-fast Guards

- `validate_processed_root()`: aborts immediately if data-root class count / test-class count / sample lower bound / empty files disagree.
- Weight download failure is fail-fast by default; set `OBWX_ALLOW_RANDOM_INIT=1` only if random-init fallback is intended (single-digit accuracy in early epochs is a symptom of random init).
- A corrupt image is auto-skipped to the next one inside `OBWXDataset.__getitem__`, so training is not interrupted.

### 4.6 Model Registry

- `models/__init__.py` registers **24** trainable classifiers; `training/benchmark.py:MODELS_CONFIG` gives each one's `freeze_strategy`, head, optimizer and scheduler.
- The paper's main experiment reports **14 baselines** (EffNetV2-S, MaxViT-T, Swin, PVTv2-B2, EffFormerV2-S, EfficientNet-B3, ViT-B/16, InceptionNeXt, ConvNeXt-V2, MobileNetV3-L, MobileViT-V2, RepViT, WideResNet-50, ResNeXt-50).
- Note: CMT was never included in timm and is replaced by PVTv2-B2; iFormer is only on timm `main`; CoAtNet-Tiny has no pretrained weights and was removed from the 14-model list.

---

## Suggested One-shot Reproduction Flow

```bash
# 0) environment
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com

# 1) unpack raw data if only zip present
#    unzip data/OBWX.zip -d data/obwx_raw   (ensure data/obwx_raw/<class>/ structure)

# 2) preprocess
python data/preprocess.py

# 3) benchmark (14 baselines; use --models to batch)
python training/benchmark.py --models "EfficientNetV2-S" "MaxViT-T" "Swin Transformer" \
    "PVTv2-B2" "EfficientFormerV2-S" "EfficientNet-B3" "ViT-B/16" "InceptionNeXt-Small" \
    "ConvNeXt-V2-Tiny" "MobileNetV3-Large" "MobileViT-V2-S" "RepViT-M1.1" \
    "WideResNet-50" "ResNeXt-50"

# 4) multi-seed
python training/multi_seed.py

# 5) analysis
python training/analyze_best.py --model "EfficientNetV2-S"
python tools_noise_robustness.py
```

> The above covers all main experiments in paper §3–§4: benchmark ranking, statistical significance, error structure, noise robustness, and multi-seed stability.

---

## Citation

If you use OBWX in your research, please cite the companion paper:

```bibtex
@misc{ge2026obwx,
  title        = {OBWX: A Benchmark for Classification of Ancient Chinese Ideographs from Four Historical Periods},
  author       = {Ge, Yanqiang and Yang, Zixu and Ge, Ziqi and Zhang, Zhan and Wang, Haoyu},
  year         = {2026},
  howpublished = {Under review},
  note         = {Dataset: 10.5281/zenodo.20037584}
}
```

> Replace the above BibTeX with the final citation once the paper is published. The OBWX dataset is archived at DOI [10.5281/zenodo.20037584](https://doi.org/10.5281/zenodo.20037584).

---

## License

Code is released under the MIT License (see `LICENSE` for details). The OBWX dataset is distributed under its own license terms; please consult the dataset documentation before redistribution.
