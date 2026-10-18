"""
OBWX 基准 — 多种子重复训练与显著性检验
================================================

产出论文 4.4 节所需的「均值 ± 标准差」证据：

    对指定的核心模型，用 N 个不同随机种子重复**完整**训练流程（相同
    数据划分、相同增强、相同超参数、相同轮数），报告测试准确率的
    均值 ± 标准差，并在模型间做配对 t 检验。

设计要点：
  1) 每个 (模型, 种子) 组合都在ibenchmark 逻辑下重新构造模型实例，
     避免上一轮的权重/优化器状态泄漏到下一轮；
  2) 每个种子前调用 set_seed(seed)，覆盖 Python / NumPy / Torch / CUDA
     与 cuDNN 确定性模式；
  3) 检查点与结果按 seed 分目录保存，避免覆盖已训练好的 best_*.pth；
  4) 全部实验共用同一份 DataLoader 配置——但 DataLoader 本身的 shuffle
     由全局随机源控制，因此每个 (模型,种子) 组合重新创建 DataLoader，
     保证不同种子的数据顺序确实不同。

用法：
    # 默认：4 个核心模型 × 种子 42/123/2024
    python training/multi_seed.py

    # 仅跑 1 个模型做冒烟测试（3 个种子、2 个 epoch）
    python training/multi_seed.py --models "EfficientNetV2-S" --epochs 2

    # 自定义种子
    python training/multi_seed.py --seeds 42 123 2024 7 2026
"""
import os
import sys
import json
import math
import statistics
import argparse
import collections
from datetime import datetime

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.config import get_config                 # noqa: E402
from utils.seed import set_seed                     # noqa: E402
from data.dataset import get_dataloaders            # noqa: E402
from training.benchmark import MODELS_CONFIG        # noqa: E402
from training.train import train_model, get_safe_name  # noqa: E402


# ============================================================
# 默认配置：论文指定的前 4 名核心模型 + 3 个随机种子
# ============================================================
CORE_MODELS = ['EfficientNetV2-S', 'MaxViT-T', 'ConvNeXt-Tiny', 'Swin Transformer']
DEFAULT_SEEDS = [42, 123, 2024]


def _abs_config(config):
    for section in ('data', 'output'):
        for key, val in list(config.get(section, {}).items()):
            if isinstance(val, str) and not os.path.isabs(val):
                config[section][key] = os.path.normpath(os.path.join(_PROJECT_ROOT, val))
    return config


def _try_paired_ttest(results_by_model):
    """
    以「最优模型」为基准，对其他模型做配对 t 检验（种子数 >= 2 时可用）。

    注意：3 个种子的配对 t 检验功效很低，论文中应如实说明——它只能给出
    趋势性证据，不能单独作为显著性结论的充分依据。
    """
    try:
        from scipy import stats
    except Exception:
        return [], 'scipy 未安装，跳过显著性检验'

    means = {}
    for name, runs in results_by_model.items():
        means[name] = statistics.mean([r['test_accuracy'] for r in runs])
    best = max(means, key=means.get)

    out = []
    for name, runs in results_by_model.items():
        if name == best or len(runs) < 2:
            continue
        a = np.array([r['test_accuracy'] for r in results_by_model[best]])
        b = np.array([r['test_accuracy'] for r in runs])
        n = min(len(a), len(b))
        if n < 2 or np.allclose(a[:n], b[:n]):
            continue
        t_stat, p_val = stats.ttest_rel(a[:n], b[:n])
        out.append({
            'pair': f'{best} vs {name}',
            'mean_diff': round(float(a[:n].mean() - b[:n].mean()), 4),
            't': round(float(t_stat), 4),
            'p': round(float(p_val), 6),
            'n_seeds': int(n),
            'significant_at_0.05': bool(p_val < 0.05),
        })
    return out, f'基准模型 = {best}'


# ============================================================
# 主流程
# ============================================================
def run(models, seeds, epochs=None, batch_size=None, device=None, out_dir=None,
        max_classes=None, resume=True, skip_existing=True):
    config = _abs_config(get_config())
    if epochs is not None:
        config['training']['epochs'] = int(epochs)
    if batch_size is not None:
        config['training']['batch_size'] = int(batch_size)
    if device is not None:
        config['device'] = device
    if max_classes is not None:
        config['data']['max_classes'] = int(max_classes)

    base_results = config['output']['results_path']
    out_dir = out_dir or os.path.join(base_results, 'multi_seed')
    os.makedirs(out_dir, exist_ok=True)

    results_by_model = collections.defaultdict(list)

    print('=' * 88)
    print('OBWX 多种子重复训练')
    print(f"模型：{', '.join(models)}")
    print(f"种子：{seeds} | 轮数：{config['training']['epochs']} | "
          f"batch：{config['training']['batch_size']} | "
          f"类别数：{config['data'].get('max_classes', '全量(6650)')} | "
          f"device：{config.get('device', 'cuda')}")
    print('=' * 88)

    for seed in seeds:
        for model_name in models:
            tag = f'{get_safe_name(model_name)}_seed{seed}'
            cache_path = os.path.join(out_dir, f'result_{tag}.json')

            if skip_existing and os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    rec = json.load(f)
                results_by_model[model_name].append(rec)
                print(f"  ⏭  复用已完成结果：{tag} "
                      f"(Test Acc={rec['test_accuracy']:.2f}%)")
                continue

            print(f"\n{'─' * 88}")
            print(f"  🚀 {model_name} | seed={seed} "
                  f"| epochs={config['training']['epochs']}")
            print(f"{'─' * 88}")

            # ---- 每个 (模型,种子) 组合：重设种子 + 重建 DataLoader ----
            set_seed(seed)
            run_config = json.loads(json.dumps(config))   # 深拷贝，避免污染全局配置
            run_config['output']['model_save_path'] = \
                os.path.join(config['output']['model_save_path'], f'seed{seed}')
            run_config['output']['results_path'] = \
                os.path.join(out_dir, 'per_run')

            train_loader, val_loader, test_loader, num_classes = get_dataloaders(run_config)

            cfg = MODELS_CONFIG[model_name]
            model = cfg['class'](num_classes=num_classes, config=cfg.get('config', {}))

            result = train_model(
                model=model,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=run_config,
                optimizer_config=cfg['optimizer'],
                scheduler_config=cfg.get('scheduler'),
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if result:
                rec = {
                    'model_name': model_name,
                    'seed': seed,
                    'test_accuracy': result['test_accuracy'],
                    'best_val_accuracy': result['best_val_accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1_score': result['f1_score'],
                }
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(rec, f, indent=2, ensure_ascii=False)
                results_by_model[model_name].append(rec)
                print(f"  ✅ {tag}: Test Acc={rec['test_accuracy']:.2f}%")

    return results_by_model, out_dir


def summarize(results_by_model, out_dir, seeds):
    summary_rows, detail_rows = [], []
    for name, runs in results_by_model.items():
        accs = [r['test_accuracy'] for r in runs]
        f1s = [r['f1_score'] for r in runs]
        summary_rows.append({
            'Model': name,
            'n_seeds': len(accs),
            'TestAcc_mean': round(statistics.mean(accs), 4),
            'TestAcc_std': round(statistics.stdev(accs) if len(accs) > 1 else 0.0, 4),
            'F1_mean': round(statistics.mean(f1s), 4),
            'F1_std': round(statistics.stdev(f1s) if len(f1s) > 1 else 0.0, 4),
            'All_seeds': [round(a, 4) for a in accs],
        })
        for r in runs:
            detail_rows.append({
                'Model': name, 'Seed': r['seed'],
                'TestAcc': round(r['test_accuracy'], 4),
                'BestValAcc': round(r['best_val_accuracy'], 4),
                'Precision': round(r['precision'], 4),
                'Recall': round(r['recall'], 4),
                'F1': round(r['f1_score'], 4),
            })

    summary_rows.sort(key=lambda d: -d['TestAcc_mean'])

    tests, test_note = _try_paired_ttest(results_by_model)

    # ---- 落盘 ----
    summary_csv = os.path.join(out_dir, 'multi_seed_summary.csv')
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding='utf-8-sig')
    detail_csv = os.path.join(out_dir, 'multi_seed_detail.csv')
    pd.DataFrame(detail_rows).to_csv(detail_csv, index=False, encoding='utf-8-sig')

    json_path = os.path.join(out_dir, 'multi_seed_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'seeds': list(seeds),
            'summary': summary_rows,
            'per_run': detail_rows,
            'significance_tests': tests,
            'significance_note': test_note,
        }, f, indent=2, ensure_ascii=False)

    # Markdown（论文可直接粘贴）
    md_path = os.path.join(out_dir, 'multi_seed_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('**表9　核心模型多种子重复实验结果（测试准确率，%）**\n\n')
        f.write('| 模型 | Seed 42 | Seed 123 | Seed 2024 | 均值±标准差 |\n')
        f.write('|---|---|---|---|---|\n')
        for r in summary_rows:
            cells = []
            for s in seeds:
                hit = [d for d in detail_rows if d['Model'] == r['Model'] and d['Seed'] == s]
                cells.append(f"{hit[0]['TestAcc']:.2f}" if hit else '—')
            f.write(f"| {r['Model']} | {' | '.join(cells)} | "
                    f"{r['TestAcc_mean']:.2f} ± {r['TestAcc_std']:.2f} |\n")
        if tests:
            f.write('\n**配对 t 检验（以最优模型为基准）**\n\n')
            f.write('| 对比 | 均值差/百分点 | t | p | 是否显著(α=0.05) |\n')
            f.write('|---|---|---|---|---|\n')
            for t in tests:
                f.write(f"| {t['pair']} | {t['mean_diff']:.2f} | {t['t']:.3f} | "
                        f"{t['p']:.4f} | {'是' if t['significant_at_0.05'] else '否'} |\n")
            f.write(f"\n> {test_note}；3 个种子的配对 t 检验统计功效较低，"
                    "结果仅作趋势性佐证，不作为显著性结论的充分依据。\n")

    return summary_rows, detail_rows, tests, test_note, [summary_csv, detail_csv, json_path, md_path]


def main():
    ap = argparse.ArgumentParser(description='OBWX 多种子重复训练')
    ap.add_argument('--models', nargs='+', default=CORE_MODELS)
    ap.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    ap.add_argument('--epochs', type=int, default=None, help='覆盖配置文件中的训练轮数')
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--max-classes', type=int, default=None,
                    help='仅使用前 N 个类别（按类别名排序）做多种子探针；'
                         '用于纯CPU环境下快速验证训练流水线稳定性，不反映全文6650类的绝对精度')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--no-skip', action='store_true', help='忽略已缓存结果，强制重跑')
    args = ap.parse_args()

    unknown = [m for m in args.models if m not in MODELS_CONFIG]
    if unknown:
        raise SystemExit(f'未知模型：{unknown}。可用模型：{list(MODELS_CONFIG)}')

    results_by_model, out_dir = run(
        args.models, args.seeds, epochs=args.epochs, batch_size=args.batch_size,
        device=args.device, out_dir=args.out_dir, max_classes=args.max_classes,
        skip_existing=not args.no_skip,
    )

    if not results_by_model:
        print('没有可用的实验结果。')
        return

    summary_rows, detail_rows, tests, test_note, paths = summarize(
        results_by_model, out_dir, args.seeds)

    print('\n' + '=' * 88)
    print('多种子重复实验结果（测试准确率 %）')
    print('=' * 88)
    hdrmodels = f"{'模型':<26}" + ''.join(f'{("seed" + str(s)):>12}' for s in args.seeds) + f"{'均值±标准差':>18}"
    print(hdrmodels)
    print('-' * 88)
    for r in summary_rows:
        per_seed = {d['Seed']: d['TestAcc'] for d in detail_rows if d['Model'] == r['Model']}
        cells = ''.join(f"{per_seed.get(s, float('nan')):>12.2f}" for s in args.seeds)
        print(f"{r['Model']:<26}{cells}{r['TestAcc_mean']:>12.2f} ± {r['TestAcc_std']:.2f}")
    print('=' * 88)
    if tests:
        print(f'\n配对 t 检验（{test_note}）：')
        for t in tests:
            print(f"  {t['pair']}: Δ={t['mean_diff']:+.2f}, t={t['t']}, p={t['p']}, "
                  f"{'显著' if t['significant_at_0.05'] else '不显著'}")
    print('\n结果已保存：')
    for p in paths:
        print(f'  {p}')


if __name__ == '__main__':
    main()
