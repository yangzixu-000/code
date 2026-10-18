# -*- coding: utf-8 -*-
"""
OBWX 多种子补充实验运行器（完全隔离，不影响原有模型与结果）

目标
----
为论文补充"多种子重复"证据：在 seed=42（已发表结果，直接复用、不重训）之外，
再跑若干新种子，量化"架构间差距 vs 跨种子波动"，确认排序稳定性。

隔离保证（硬约束：不影响原来的模型和结果）
------------------------------------------
1. 只读既有文件：不修改 config.yaml、data/、models/、output/models/、output/results/。
2. 所有新产物写入独立根目录 output/multiseed/（默认），绝不落回原目录。
3. 路径守卫：解析后的输出目录若命中受保护路径，直接拒绝运行。
4. 前后校验：启动前/结束后对受保护路径做 md5 快照比对，任何变动立即报错退出。
5. 不调用 benchmark.generate_report（它会覆盖 benchmark_summary.csv / detailed_results.json）。

关键差异（相对原 pipeline）
--------------------------
- 默认开启 --reseed-per-model：每个 (模型, 种子) 前重置全部 RNG，
  结果不依赖执行顺序，可独立复现（原 benchmark.py 全程只 set_seed 一次）。
- 默认把 processed_root 覆写为 ./data/obwx_processed_clean（论文实际使用的数据集）；
  config.yaml 中的 ./data/obwx_processed 是旧的不完整构建，但本脚本不修改它。

用法
----
  # 列出 14 个基线模型中当前可训练/不可训练的
  python tools_multiseed.py --list

  # 冒烟测试（CPU 可跑，验证隔离性；不改变任何原文件）
  python tools_multiseed.py --seeds 123 --models "MobileNetV3-Large" --epochs 1 --max-classes 30

  # 正式：14 模型 × 2 个新种子（需 GPU 机器）
  python tools_multiseed.py --seeds 123 2024

  # 汇总（不训练，仅把已有各 seed 结果与 seed42 基线合并统计）
  python tools_multiseed.py --aggregate-only
"""
import argparse
import copy
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.config import get_config
from utils.seed import set_seed
from data.dataset import get_dataloaders
from training.benchmark import MODELS_CONFIG
from training.train import train_model, get_safe_name

# 论文 Table：14 个模型的 seed=42 基线顺序（按基线准确率降序固定，保证可复现）
BASELINE14 = [
    "EfficientNetV2-S",
    "MaxViT-T",
    "Swin Transformer",
    "EfficientFormerV2-S",
    "PVTv2-B2",
    "InceptionNeXt-Small",
    "ViT-B/16",
    "ConvNeXt-V2-Tiny",
    "MobileViT-V2-S",
    "MobileNetV3-Large",
    "RepViT-M1.1",
    "CoAtNet-Tiny",
    "WideResNet-50",
    "ResNeXt-50",
]

DEFAULT_OUT_ROOT = os.path.join(_HERE, "output", "multiseed")
DEFAULT_DATA_ROOT = os.path.join(_HERE, "data", "obwx_processed")
BASELINE_JSON = os.path.join(_HERE, "output", "results", "model_metrics_14.json")

# 受保护：这些路径下的文件在全程前后必须逐字节不变
PROTECTED_FILES = [
    os.path.join(_HERE, "config.yaml"),
    os.path.join(_HERE, "output", "results", "benchmark_summary.csv"),
    os.path.join(_HERE, "output", "results", "detailed_results.json"),
    os.path.join(_HERE, "output", "results", "model_metrics_14.json"),
]
PROTECTED_DIRS = [
    os.path.join(_HERE, "output", "models"),
    os.path.join(_HERE, "output", "results"),
]


# --------------------------------------------------------------------------
# 隔离校验工具
# --------------------------------------------------------------------------
def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def snapshot_protected():
    snap = {}
    for p in PROTECTED_FILES:
        if os.path.isfile(p):
            snap[p] = _md5(p)
    for d in PROTECTED_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            # 跳过本实验自己的输出根目录（若被放在 results 下则不应发生）
            for fn in files:
                fp = os.path.join(root, fn)
                if os.path.abspath(fp).startswith(os.path.abspath(DEFAULT_OUT_ROOT)):
                    continue
                try:
                    snap[fp] = _md5(fp)
                except OSError:
                    pass
    return snap


def diff_snapshot(before, after):
    changed = []
    for k, v in before.items():
        if k not in after:
            changed.append(("deleted", k))
        elif after[k] != v:
            changed.append(("modified", k))
    for k in after:
        if k not in before:
            changed.append(("created", k))
    return changed


def assert_isolated(out_root):
    """路径守卫：输出根目录不得解析到受保护路径内部或之上。"""
    out_root = os.path.abspath(out_root)
    for d in PROTECTED_DIRS:
        d = os.path.abspath(d)
        if out_root == d or out_root.startswith(d + os.sep) or d.startswith(out_root + os.sep):
            raise SystemExit(
                f"[隔离守卫] 输出根目录与受保护路径冲突，拒绝运行:\n"
                f"  out_root = {out_root}\n  protected= {d}\n"
                f"请改用 output/multiseed 之类的独立目录。"
            )


# --------------------------------------------------------------------------
# 配置构造（只在内存中改动，绝不写回 config.yaml）
# --------------------------------------------------------------------------
def build_seed_config(base_config, seed, out_root, data_root, epochs, max_classes, num_workers):
    cfg = copy.deepcopy(base_config)
    seed_root = os.path.join(os.path.abspath(out_root), f"seed{seed}")
    cfg["output"]["model_save_path"] = os.path.join(seed_root, "models")
    cfg["output"]["results_path"] = os.path.join(seed_root, "results")
    cfg["output"]["logs_path"] = os.path.join(seed_root, "logs")
    cfg["data"]["processed_root"] = data_root
    cfg["random_seed"] = seed
    if epochs is not None:
        cfg["training"]["epochs"] = int(epochs)
    if num_workers is not None:
        cfg["training"]["num_workers"] = int(num_workers)
    if max_classes:
        cfg["data"]["max_classes"] = int(max_classes)
    else:
        cfg["data"].pop("max_classes", None)
    for k in ("model_save_path", "results_path", "logs_path"):
        os.makedirs(cfg["output"][k], exist_ok=True)
    return cfg, seed_root


def resolve_models(names=None, verbose=True):
    """返回 (可训练模型列表, 不可训练模型列表)"""
    targets = names if names else list(BASELINE14)
    ok, missing = [], []
    for m in targets:
        if m in MODELS_CONFIG:
            ok.append(m)
        else:
            missing.append(m)
    if verbose and missing:
        print(f"  ⚠️ 下列模型不在 MODELS_CONFIG 中，无法训练（将跳过）: {missing}")
    return ok, missing


# --------------------------------------------------------------------------
# 单个种子的一整轮（arm）
# --------------------------------------------------------------------------
def run_arm(seed, models, base_config, out_root, data_root, epochs,
            max_classes, num_workers, reseed_per_model, skip_existing,
            keep_checkpoints):
    cfg, seed_root = build_seed_config(
        base_config, seed, out_root, data_root, epochs, max_classes, num_workers
    )

    manifest_path = os.path.join(seed_root, "run_manifest.json")
    print(f"\n{'=' * 66}\n🌱 Seed arm: {seed}  ->  {seed_root}\n{'=' * 66}")

    # 忠实复现原协议：整轮只 set_seed 一次（与 benchmark.py 一致）
    set_seed(seed)
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(cfg)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "created": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "model_order": models,
            "reseed_per_model": reseed_per_model,
            "num_classes": num_classes,
            "effective_config": {
                "data": cfg["data"],
                "training": cfg["training"],
                "output": cfg["output"],
                "random_seed": cfg["random_seed"],
            },
        }, f, ensure_ascii=False, indent=2)

    results = {}
    for model_name in models:
        cfg_m = MODELS_CONFIG[model_name]
        safe = get_safe_name(model_name)
        res_path = os.path.join(cfg["output"]["results_path"], f"result_{safe}.json")
        ckpt_path = os.path.join(cfg["output"]["model_save_path"], f"best_{safe}.pth")

        if skip_existing and os.path.isfile(res_path):
            print(f"  ⏭️  {model_name}: 已有结果，跳过")
            results[model_name] = json.load(open(res_path, encoding="utf-8"))
            continue

        # 每个模型独立重置 RNG（默认开启）→ 结果与执行顺序无关
        if reseed_per_model:
            set_seed(seed)

        print(f"\n  🏃 {model_name} (seed={seed}, classes={num_classes})")
        model = cfg_m["class"](num_classes, config=cfg_m.get("config", {}))
        result = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=cfg,
            optimizer_config=cfg_m["optimizer"],
            scheduler_config=cfg_m.get("scheduler"),
        )
        result["seed"] = seed
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        results[model_name] = result
        print(f"  ✅ {model_name}: test={result['test_accuracy']:.2f}%")

        if not keep_checkpoints and os.path.isfile(ckpt_path):
            os.remove(ckpt_path)

    return results, seed_root


# --------------------------------------------------------------------------
# 汇总统计
# --------------------------------------------------------------------------
def load_baseline42(path=BASELINE_JSON):
    if not os.path.isfile(path):
        return {}
    data = json.load(open(path, encoding="utf-8"))
    return {r["model"]: r.get("test_acc") for r in data}


def collect_arms(out_root, seeds):
    """读取各 seed arm 已产出的结果：{seed: {model: test_acc}}"""
    arms = {}
    for s in seeds:
        rd = os.path.join(os.path.abspath(out_root), f"seed{s}", "results")
        if not os.path.isdir(rd):
            continue
        per = {}
        for fn in os.listdir(rd):
            if fn.startswith("result_") and fn.endswith(".json"):
                r = json.load(open(os.path.join(rd, fn), encoding="utf-8"))
                per[r["model_name"]] = r["test_accuracy"]
        if per:
            arms[s] = per
    return arms


def aggregate(out_root, seeds, baseline=None):
    arms = collect_arms(out_root, seeds)
    all_seeds = list(arms.keys())
    if baseline:
        all_seeds = [42] + [s for s in all_seeds if s != 42]

    rows = []
    for m in BASELINE14:
        vals, ranks = {}, {}
        for s in all_seeds:
            src = baseline if (s == 42 and baseline) else arms.get(s, {})
            v = src.get(m)
            if v is not None:
                vals[s] = float(v)
        if not vals:
            continue
        # 排名：在每个种子内部按 test_acc 降序
        for s in all_seeds:
            src = baseline if (s == 42 and baseline) else arms.get(s, {})
            ordered = sorted(src.items(), key=lambda kv: -kv[1])
            for i, (k, _v) in enumerate(ordered, 1):
                if k == m:
                    ranks[s] = i
        nums = list(vals.values())
        mean = sum(nums) / len(nums)
        var = sum((x - mean) ** 2 for x in nums) / (len(nums) - 1) if len(nums) > 1 else 0.0
        std = var ** 0.5
        rows.append({
            "model": m,
            **{f"seed{s}": (round(vals[s], 2) if s in vals else None) for s in all_seeds},
            "mean": round(mean, 2),
            "std": round(std, 2),
            "min": round(min(nums), 2),
            "max": round(max(nums), 2),
            "spread_pp": round(max(nums) - min(nums), 2),
            "n_seeds": len(nums),
            **{f"rank_seed{s}": ranks.get(s) for s in all_seeds},
        })

    rows.sort(key=lambda r: -r["mean"])
    return rows, all_seeds, arms


def write_summary(out_root, rows, all_seeds):
    os.makedirs(out_root, exist_ok=True)
    p_json = os.path.join(out_root, "multiseed_summary.json")
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump({"seeds": all_seeds, "rows": rows}, f, ensure_ascii=False, indent=2)

    # 轻量 CSV（不依赖 pandas，避免与项目其他依赖耦合）
    cols = ["model"] + [f"seed{s}" for s in all_seeds] + \
           ["mean", "std", "min", "max", "spread_pp", "n_seeds"] + \
           [f"rank_seed{s}" for s in all_seeds]
    p_csv = os.path.join(out_root, "multiseed_summary.csv")
    with open(p_csv, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
    return p_json, p_csv


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="OBWX 多种子补充实验（隔离运行）")
    ap.add_argument("--seeds", nargs="+", type=int, default=[123, 2024],
                    help="要补跑的种子（默认 123 2024，与 pilot 一致；42 为已发表基线，不重训）")
    ap.add_argument("--models", nargs="*", default=None, help="只跑指定模型（默认全部 14 个基线模型）")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 epoch 数（默认沿用 config.yaml=80）")
    ap.add_argument("--max-classes", type=int, default=None, help="只取前 N 类（冒烟测试用，正式运行不要设）")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="隔离输出根目录")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="数据集根目录（默认 obwx_processed）")
    ap.add_argument("--num-workers", type=int, default=None, help="覆盖 num_workers（默认沿用 config）")
    ap.add_argument("--reseed-per-model", dest="reseed_per_model", action="store_true", default=True)
    ap.add_argument("--no-reseed-per-model", dest="reseed_per_model", action="store_false",
                    help="关闭每模型重播种，改为整轮只播种一次（与 benchmark.py 完全一致）")
    ap.add_argument("--skip-existing", action="store_true", help="已有 result json 则跳过该 (seed, model)")
    ap.add_argument("--keep-checkpoints", dest="keep_checkpoints", action="store_true", default=True)
    ap.add_argument("--delete-checkpoints", dest="keep_checkpoints", action="store_false",
                    help="跑完删除该 seed 的检查点以省磁盘（结果 json 保留）")
    ap.add_argument("--aggregate-only", action="store_true", help="不训练，只汇总已有结果")
    ap.add_argument("--list", action="store_true", help="列出 14 个基线模型的可训练状态")
    ap.add_argument("--no-isolation-check", action="store_true", help="关闭前后 md5 校验（不建议）")
    args = ap.parse_args()

    if args.list:
        ok, missing = resolve_models(None, verbose=False)
        print(f"可训练 ({len(ok)}):")
        for m in ok:
            print(f"  ✅ {m}")
        print(f"不可训练 ({len(missing)}):")
        for m in missing:
            print(f"  ❌ {m}  (不在 training/benchmark.py 的 MODELS_CONFIG 中)")
        return

    out_root = os.path.abspath(args.out_root)
    assert_isolated(out_root)

    baseline = load_baseline42()

    if args.aggregate_only:
        rows, all_seeds, arms = aggregate(out_root, args.seeds, baseline)
        pj, pc = write_summary(out_root, rows, all_seeds)
        print(f"汇总完成: seeds={all_seeds}\n  {pj}\n  {pc}")
        return

    models, missing = resolve_models(args.models)
    if not models:
        raise SystemExit("没有可训练的模型，退出。")

    print(f"输出根目录(隔离): {out_root}")
    print(f"数据集根目录: {args.data_root}")
    print(f"种子: {args.seeds} | 模型数: {len(models)} | epochs: {args.epochs or 'config(80)'}")
    print(f"基线 seed=42 复用（不重训）: {'是' if baseline else '未找到 model_metrics_14.json'}")

    before = None if args.no_isolation_check else snapshot_protected()

    base_config = get_config()

    for seed in args.seeds:
        if seed == 42:
            print("\n⚠️  seed=42 是已发表基线，默认不重训（会覆盖语义）。跳过。")
            continue
        run_arm(
            seed=seed, models=models, base_config=base_config,
            out_root=out_root, data_root=args.data_root, epochs=args.epochs,
            max_classes=args.max_classes, num_workers=args.num_workers,
            reseed_per_model=args.reseed_per_model, skip_existing=args.skip_existing,
            keep_checkpoints=args.keep_checkpoints,
        )

    rows, all_seeds, arms = aggregate(out_root, args.seeds, baseline)
    pj, pc = write_summary(out_root, rows, all_seeds)
    print(f"\n汇总: {pj}\n       {pc}")

    if before is not None:
        after = snapshot_protected()
        changed = diff_snapshot(before, after)
        if changed:
            print("\n❌ [隔离校验失败] 受保护文件发生变化:")
            for kind, p in changed[:20]:
                print(f"   {kind}: {p}")
            raise SystemExit(1)
        print(f"\n✅ [隔离校验通过] {len(before)} 个受保护文件前后 md5 完全一致，原模型/结果未被改动。")


if __name__ == "__main__":
    main()
