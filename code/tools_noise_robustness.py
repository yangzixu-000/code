# -*- coding: utf-8 -*-
"""
OBWX 噪声鲁棒性评测 (P0-5 实证)
================================
对发布的 6650 类检查点施加受控噪声扰动，量化准确率退化。

当前发布的 6650 类检查点仅有 2 个（EfficientNetV2-S、MaxViT-T，位于
output/models 根目录）；其余 seed 目录下的检查点为 100 类试点，无对应
6650 类测试切分，故本脚本自动定位「头层 out_features == 6650」的检查点，
在其上完成噪声鲁棒性评测。评测在 clean test split (obwx_processed_clean/test)
上进行。

噪声类型（与论文 §4.5 计划一致）：
  - gaussian       : 加性高斯噪声，sigma ∈ {15,30,50}（0-255 像素尺度）
  - salt_pepper    : 椒盐噪声，比例 p ∈ {0.03,0.07,0.12}
  - stroke_erosion : 笔画侵蚀（模拟笔画缺失/墨迹洇散），核大小 k ∈ {2,3,5}

同一 (noise,level) 条件下所有模型共享同一随机种子，保证噪声图案一致、可比。

用法：
  python tools_noise_robustness.py --limit 400            # 校验 + 计时
  python tools_noise_robustness.py                       # 全量后台运行
"""
import os
import re
import sys
import json
import time
import argparse
import collections
import random

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.benchmark import MODELS_CONFIG          # noqa: E402

DEVICE = torch.device("cpu")
NUM_CLASSES = 6650
IMAGE_SIZE = 224
SEED = 42
BATCH_SIZE = 64

# 候选 6650 类检查点目录（根目录存了 2 个全量检查点）
CKPT_CANDIDATE_DIRS = [
    os.path.join(PROJECT_ROOT, "output", "models"),
    os.path.join(PROJECT_ROOT, "output", "models", "seed42"),
    os.path.join(PROJECT_ROOT, "output", "models", "seed123"),
    os.path.join(PROJECT_ROOT, "output", "models", "seed2024"),
]
TEST_ROOT = os.path.join(PROJECT_ROOT, "data", "obwx_processed")
CLASS_MAP = os.path.join(TEST_ROOT, "class_mapping.json")

# 4 个代表性架构；脚本自动筛出其中「6650 类」的检查点
MODEL_CANDIDATES = [
    "Swin Transformer", "ConvNeXt-Tiny", "EfficientNetV2-S", "MaxViT-T",
]
FAMILY = {
    "Swin Transformer": "Hierarchical Transformer",
    "ConvNeXt-Tiny": "Modern CNN",
    "EfficientNetV2-S": "Modern CNN",
    "MaxViT-T": "Hybrid CNN-Transformer",
}

NOISE_LEVELS = {
    "gaussian": [15, 30, 50],
    "salt_pepper": [0.03, 0.07, 0.12],
    "stroke_erosion": [2, 3, 5],
}


def safe_name(m):
    return re.sub(r"[^a-z0-9]+", "_", m.lower()).strip("_")


def apply_noise(img, noise_type, level, rng):
    if noise_type == "clean":
        return img
    if noise_type == "gaussian":
        sigma = float(level)
        noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if noise_type == "salt_pepper":
        p = float(level)
        out = img.copy()
        mask = rng.random(img.shape[:2])
        out[mask < p / 2] = (0, 0, 0)
        out[mask > 1 - p / 2] = (255, 255, 255)
        return out
    if noise_type == "stroke_erosion":
        k = int(level)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, bin_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((k, k), np.uint8)
        eroded = cv2.erode(bin_img, kernel, iterations=1)
        out = (255 - eroded).astype(np.uint8)
        return np.stack([out, out, out], axis=2)
    raise ValueError(noise_type)


class NoiseTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, class_to_idx, noise_type, level, rng, limit=None):
        self.samples = []
        split_dir = os.path.join(root, "test")
        for cls in os.listdir(split_dir):
            cp = os.path.join(split_dir, cls)
            if not os.path.isdir(cp) or cls not in class_to_idx:
                continue
            ci = class_to_idx[cls]
            for fn in os.listdir(cp):
                if os.path.splitext(fn)[1].lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((os.path.join(cp, fn), ci))
        if limit and limit < len(self.samples):
            by_cls = collections.defaultdict(list)
            for s in self.samples:
                by_cls[s[1]].append(s)
            r = random.Random(SEED)
            per = max(1, limit // max(1, len(by_cls)))
            picked = []
            for ci, lst in by_cls.items():
                r.shuffle(lst)
                picked.extend(lst[:per])
            if len(picked) > limit:
                picked = picked[:limit]
            self.samples = picked
        self.noise_type, self.level, self.rng = noise_type, level, rng
        self.resize = A.Resize(IMAGE_SIZE, IMAGE_SIZE)
        self.normalize = A.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        self.to_tensor = ToTensorV2()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = np.array(Image.open(path).convert("RGB"))
            img = self.resize(image=img)["image"]
            img = apply_noise(img, self.noise_type, self.level, self.rng)
            img = self.normalize(image=img)["image"]
            img = self.to_tensor(image=img)["image"]
            return img, label
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for inputs, targets in loader:
        inputs = inputs.to(DEVICE)
        preds = model(inputs).argmax(1).cpu()
        correct += int((preds == targets).sum())
        total += int(targets.shape[0])
    return 100.0 * correct / total if total else 0.0


def condition_seed(noise_type, level):
    return int(abs(hash(f"{noise_type}:{level}"))) % (2 ** 31)


def find_6650_checkpoints(models):
    """返回 {model_name: ckpt_path}，仅保留头层 out_features==6650 的。"""
    found = {}
    for m in models:
        fn = f"best_{safe_name(m)}.pth"
        for d in CKPT_CANDIDATE_DIRS:
            p = os.path.join(d, fn)
            if not os.path.exists(p):
                continue
            try:
                sd = torch.load(p, map_location="cpu")
            except Exception:
                continue
            # 找最大 out_features 的 2D weight，判断是否为 6650
            max_out = 0
            for k, v in sd.items():
                if k.endswith("weight") and v.dim() == 2:
                    max_out = max(max_out, v.shape[0])
            if max_out == NUM_CLASSES:
                found[m] = p
                break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(
        PROJECT_ROOT, "output", "results", "noise_robustness.json"))
    args = ap.parse_args()

    with open(CLASS_MAP, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)["class_to_idx"]

    ckpts = find_6650_checkpoints(MODEL_CANDIDATES)
    chosen = list(ckpts.keys())
    if not chosen:
        print("ERROR: no 6650-class checkpoints found.")
        return
    print(f"6650-class checkpoints found ({len(chosen)}):")
    for m in chosen:
        print(f"  {m:18s} <- {os.path.relpath(ckpts[m], PROJECT_ROOT)}")
    print(f"test split: {TEST_ROOT}/test | limit={args.limit}\n")

    results = {m: {} for m in chosen}
    conditions = [("clean", None)] + [
        (nt, lv) for nt, lvs in NOISE_LEVELS.items() for lv in lvs
    ]

    # 每个模型只加载一次，跨所有噪声条件复用，避免重复 I/O
    for m in chosen:
        cfg = MODELS_CONFIG[m]
        model = cfg["class"](NUM_CLASSES, config=cfg.get("config", {}))
        state = torch.load(ckpts[m], map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        print(f"\n### model {m}  (ckpt {os.path.relpath(ckpts[m], PROJECT_ROOT)}) ###")
        for noise_type, level in conditions:
            key = "clean" if noise_type == "clean" else f"{noise_type}_{level}"
            rng = np.random.default_rng(condition_seed(noise_type, level))
            ds = NoiseTestDataset(TEST_ROOT, class_to_idx, noise_type, level, rng,
                                  limit=args.limit)
            loader = torch.utils.data.DataLoader(
                ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            t0 = time.perf_counter()
            acc = evaluate(model, loader)
            dt = time.perf_counter() - t0
            results[m][key] = round(acc, 2)
            print(f"  {key:22s} n={len(ds):5d}  {acc:6.2f}%   ({dt:.1f}s)")
        del model

    out = {
        "meta": {
            "test_split": "obwx_processed/test",
            "num_classes_space": NUM_CLASSES,
            "limit": args.limit,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "models": chosen,
            "checkpoints": {m: os.path.relpath(ckpts[m], PROJECT_ROOT) for m in chosen},
            "noise_levels": NOISE_LEVELS,
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {args.out}")

    base = {m: results[m].get("clean") for m in chosen}
    print("\n--- accuracy drop vs clean (%) ---")
    cols = [f"g_{v}" for v in NOISE_LEVELS['gaussian']] + \
           [f"s_{v}" for v in NOISE_LEVELS['salt_pepper']] + \
           [f"e_{v}" for v in NOISE_LEVELS['stroke_erosion']]
    print("model".ljust(20) + "clean".rjust(8) + "".join(c.rjust(12) for c in cols))
    for m in chosen:
        row = f"{m:18s}{base[m]:8.2f}"
        for nt, lvs in NOISE_LEVELS.items():
            for lv in lvs:
                v = results[m].get(f"{nt}_{lv}")
                drop = (base[m] - v) if v is not None else float('nan')
                row += f"{drop:11.2f}"
        print(row)


if __name__ == "__main__":
    main()
