# -*- coding: utf-8 -*-
"""
导出测试集逐样本预测（preds / trues），用于重算表 7（频段准确率）、表 8（Top-10 易混淆字头对）、
图 7 / 图 8（长尾误差结构）。

结果 JSON 里只有 9 个标量/曲线字段，不含逐类数据，所以重算这些必须落到逐样本层面。
本脚本在【训练那台机器 + 本项目根目录】运行即可，产物是纯文本 JSON，拷回本机后我来做全部推导。

用法（单模型）：
    HF_ENDPOINT=https://hf-mirror.com python tools_dump_predictions.py ^
        --model EfficientNetV2-S ^
        --ckpt "C:/Users/geyan/Desktop/OBWX实验结果/10.8结果/best_efficientnetv2_s.pth" ^
        --out-dir "C:/Users/geyan/Desktop/OBWX_predictions"

用法（整批：自动扫描一个目录里所有 best_*.pth）：
    HF_ENDPOINT=https://hf-mirror.com python tools_dump_predictions.py ^
        --ckpt-dir "C:/Users/geyan/Desktop/OBWX实验结果/10.1结果/models" ^
        --out-dir "C:/Users/geyan/Desktop/OBWX_predictions"

产出：每个模型一个 JSON，含 model_name / preds / trues / test_accuracy(复现自校验) / num_classes。
附带 SPLIT.txt 记录本次所用的划分指纹。

注意：
  * 必须从项目根目录运行（config.yaml 由 utils/config.py 自动查找）。
  * preds / trues 存的是【类别索引 int】，配套需同时拷贝 obwx_processed/class_mapping.json，
    里面必须有 index -> class 的映射；我会用它把索引还原为字符（区分字头需要）。
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

from data.dataset import OBWXDataset, get_transform
from utils.config import get_config
from training.benchmark import MODELS_CONFIG

# 论文 14 个模型：文件名片段 -> MODELS_CONFIG 中的正式名
CKPT2MODEL = {
    "efficientnetv2_s": "EfficientNetV2-S",
    "efficientnet_b3": "EfficientNet-B3",
    "maxvit_t": "MaxViT-T",
    "swin_transformer": "Swin Transformer",
    "pvtv2_b2": "PVTv2-B2",
    "efficientformerv2_s": "EfficientFormerV2-S",
    "vit_b_16": "ViT-B/16",
    "inceptionnext_small": "InceptionNeXt-Small",
    "convnext_v2_tiny": "ConvNeXt-V2-Tiny",
    "mobilenetv3_large": "MobileNetV3-Large",
    "mobilevit_v2_s": "MobileViT-V2-S",
    "repvit_m1_1": "RepViT-M1.1",
    "wideresnet_50": "WideResNet-50",
    "resnext_50": "ResNeXt-50",
}


def resolve_model_name(ckpt_path, explicit=None):
    if explicit:
        return explicit
    base = os.path.basename(ckpt_path).lower()
    if not base.endswith(".pth"):
        base += ".pth"
    stem = base[len("best_"):-len(".pth")] if base.startswith("best_") else base[:-len(".pth")]
    if stem in CKPT2MODEL:
        return CKPT2MODEL[stem]
    # 退化为最长前缀匹配，容忍 best_efficientnetv2_s_xxx.pth 之类的变体
    best, best_len = None, 0
    for key, name in CKPT2MODEL.items():
        if stem.startswith(key) and len(key) > best_len:
            best, best_len = name, len(key)
    return best


def build_loader(config, split="test", batch_size=64, num_workers=8):
    root = config["data"]["processed_root"]
    c2i = json.load(open(os.path.join(root, "class_mapping.json"), encoding="utf-8"))["class_to_idx"]
    idx_to_class = {v: k for k, v in c2i.items()}
    ds = OBWXDataset(root, split, c2i, get_transform(False, 224), None, 224)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=False)
    return dl, idx_to_class, len(ds), ds


def run_one(model_name, ckpt_path, config, args):
    cls = MODELS_CONFIG[model_name]["class"]
    num_classes = len(json.load(
        open(os.path.join(config["data"]["processed_root"], "class_mapping.json"),
             encoding="utf-8"))["class_to_idx"])
    model = cls(num_classes, config=dict(MODELS_CONFIG[model_name].get("config", {})))
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    dl, idx_to_class, n, ds = build_loader(config, "test", args.batch_size, args.num_workers)
    preds, trues = [], []
    t0 = time.time()
    with torch.no_grad():
        for x, y in dl:
            out = model(x)
            preds.extend(out.argmax(1).tolist())
            trues.extend(y.tolist())
    acc = sum(int(a == b) for a, b in zip(preds, trues)) / len(trues) * 100.0
    print(f"[dump] {model_name}: N={len(trues)}  本地复现准确率={acc:.2f}%  用时 {time.time()-t0:.0f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    safe = model_name.lower().replace("/", "_").replace(" ", "_")
    out_path = os.path.join(args.out_dir, f"preds_{safe}.json")
    json.dump({
        "model_name": model_name,
        "num_classes": num_classes,
        "n_test": len(trues),
        "reproduced_test_accuracy": acc,
        "preds": preds,
        "trues": trues,
        "ckpt": ckpt_path,
    }, open(out_path, "w", encoding="utf-8"))
    print(f"[dump] 已写出 -> {out_path}")


def write_split_fingerprint(args, config):
    root = config["data"]["processed_root"]
    cm_path = os.path.join(root, "class_mapping.json")
    cm = json.load(open(cm_path, encoding="utf-8"))
    test_dir = os.path.join(root, "test")
    files = sorted(os.listdir(test_dir))
    h = hashlib.md5("\n".join(files).encode("utf-8")).hexdigest()
    txt = [
        "划分指纹（用于核对两边测试集是否一致）",
        f"processed_root      : {os.path.abspath(root)}",
        f"test 目录文件数     : {len(files)}",
        f"test 文件清单 md5   : {h}",
        f"class_mapping 键    : {sorted(cm.keys())}",
    ]
    for k in ("total_train", "total_val", "total_test"):
        if k in cm:
            txt.append(f"{k:19s}: {cm[k]}")
    p = os.path.join(args.out_dir, "SPLIT.txt")
    open(p, "w", encoding="utf-8").write("\n".join(txt))
    print("[dump] " + "\n[dump] ".join(txt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="MODELS_CONFIG 中的模型名；配合 --ckpt 使用")
    ap.add_argument("--ckpt", help="单个检查点路径")
    ap.add_argument("--ckpt-dir", help="批量：扫描该目录下所有 best_*.pth")
    ap.add_argument("--out-dir", required=True, help="输出目录")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", help="批量模式下只跑这些模型名")
    args = ap.parse_args()

    config = get_config()
    write_split_fingerprint(args, config)

    if args.ckpt:
        name = resolve_model_name(args.ckpt, args.model)
        if name not in MODELS_CONFIG:
            raise SystemExit(f"无法从 {args.ckpt} 解析出模型名，请显式给 --model；"
                             f"可选：{list(MODELS_CONFIG)}")
        run_one(name, args.ckpt, config, args)
        return

    if not args.ckpt_dir:
        raise SystemExit("需要 --ckpt 或 --ckpt-dir 其中之一")

    seen = set()
    for f in sorted(os.listdir(args.ckpt_dir)):
        if not f.endswith(".pth"):
            continue
        name = resolve_model_name(f)
        if name is None or name not in MODELS_CONFIG:
            print(f"[dump] 跳过（无法映射模型名）: {f}")
            continue
        if args.only and name not in args.only:
            continue
        if name in seen:
            continue
        seen.add(name)
        try:
            run_one(name, os.path.join(args.ckpt_dir, f), config, args)
        except Exception as e:
            print(f"[dump] 失败 {name} <- {f}: {e}")
    print(f"[dump] 全部完成，共 {len(seen)} 个模型 -> {args.out_dir}")


if __name__ == "__main__":
    main()
