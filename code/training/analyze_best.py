"""
OBWX 基准 — 最优模型误差结构分析
================================================

产出论文「表7 不同频段类别的性能分析」与「Top-10 易混淆字头对」的全部数据：

  1) 载入最优模型检查点，在测试集上做一次完整推理，保存逐样本预测；
  2) 按每类**原始**训练样本数将 6,650 个类别划分为
       头部 (>=50) / 中部 (15~49) / 尾部 (<15)
     分别统计类别数、测试样本数、准确率、精确率、召回率；
  3) 由混淆矩阵提取误判样本数最多的 10 组字头对，标注
       是否属同一字头（字头编号一致，即跨时期/跨异体的同源混淆）；
  4) 输出 CSV / JSON / Markdown 三份结果 + 三张配图。

为什么用「原始样本数」而不是「增强后训练集样本数」分档？
    数据预处理阶段对不足 15 张的类别做了离线增强，补至 15 张，
    因此划分后的训练集里所有类别都 >=15，无法区分长尾强弱。
    只有「增强前的原始样本数」才真实刻画了古文字存世量的长尾差异，
    也才能真正检验「尾部类别训练信号匮乏 → 判别力下降」这一假设。

用法：
    python training/analyze_best.py                          # 默认 EfficientNetV2-S
    python training/analyze_best.py --model "MaxViT-T"
    python training/analyze_best.py --batch-size 32 --no-figures
"""
import os
import sys
import csv
import json
import time
import argparse
import collections
from datetime import datetime

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.config import get_config                # noqa: E402
from utils.seed import set_seed                    # noqa: E402
from data.dataset import get_test_dataloader       # noqa: E402
from training.benchmark import MODELS_CONFIG       # noqa: E402
from training.train import get_safe_name           # noqa: E402

from sklearn.metrics import (                      # noqa: E402
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)


# ============================================================
# 工具函数
# ============================================================
def _abs_config(config):
    for section in ('data', 'output'):
        for key, val in list(config.get(section, {}).items()):
            if isinstance(val, str) and not os.path.isabs(val):
                config[section][key] = os.path.normpath(os.path.join(_PROJECT_ROOT, val))
    return config


def head_of(cls_name: str) -> str:
    """字头编号：类别名 '0040-2' 的前缀 '0040'，代表同一个字头"""
    return cls_name.split('-')[0]


def tier_of(n: int) -> str:
    if n >= 50:
        return 'head'
    if n >= 15:
        return 'mid'
    return 'tail'


TIER_LABEL = {
    'head': '头部（样本数≥50）',
    'mid': '中部（15≤样本数<50）',
    'tail': '尾部（样本数<15）',
}
TIER_ORDER = ['head', 'mid', 'tail']


# ============================================================
# 测试集推理
# ============================================================
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels, scores = [], [], []
    t0 = time.perf_counter()
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        preds.append(logits.argmax(1).cpu())
        labels.append(targets.cpu())
        scores.append(torch.softmax(logits.float(), dim=1).max(1)[0].cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    y_conf = torch.cat(scores).numpy()
    return y_true, y_pred, y_conf, time.perf_counter() - t0


# ============================================================
# 频段性能分析
# ============================================================
def long_tail_analysis(y_true, y_pred, idx_to_class, class_stats, num_classes):
    """
    按每类原始样本数分组，统计组内准确率 / 精确率 / 召回率。

    指标口径（与论文 3.3 节一致）：
      - 组内准确率：把真实标签属于该频段的样本挑出来，在完整 6,650 类
        标签空间下计算 Top-1 命中率（等价于该频段的 micro recall）；
      - 组内加权精确率/召回率：在上述样本子集上，以各类支持度为权重
        加权平均（weighted），口径与全文主表一致；
      - 宏平均召回率：先在类内算 recall 再在类间取算术平均，用于观察
        长尾类别是否被"票仓"掩盖（作为补充指标）。
    """
    original = {c: int(v.get('original', 0)) for c, v in class_stats.items()}
    tier_of_class = {c: tier_of(original.get(c, 0)) for c in original}

    counter_true = collections.Counter(int(t) for t in y_true)
    counter_hit = collections.Counter(int(t) for t, p in zip(y_true, y_pred) if int(t) == int(p))

    rows = []
    for tier in TIER_ORDER:
        cls_list = [c for c, t in tier_of_class.items() if t == tier]
        idx_list = [int(idx) for idx, c in idx_to_class.items() if c in tier_of_class and tier_of_class[c] == tier]
        idx_set = set(idx_list)

        mask = np.array([int(t) in idx_set for t in y_true])
        n_samples = int(mask.sum())
        if n_samples == 0:
            continue
        sub_true = y_true[mask].astype(int)
        sub_pred = y_pred[mask].astype(int)

        acc = accuracy_score(sub_true, sub_pred) * 100.0
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(
            sub_true, sub_pred, average='weighted', zero_division=0)
        _, r_m, _, _ = precision_recall_fscore_support(
            sub_true, sub_pred, average='macro', zero_division=0)

        rows.append({
            'tier': tier,
            'tier_label': TIER_LABEL[tier],
            'num_classes_total': len(cls_list),
            'num_classes_in_test': sum(1 for i in idx_list if counter_true.get(i, 0) > 0),
            'num_test_samples': n_samples,
            'sample_ratio_pct': round(100.0 * n_samples / len(y_true), 2),
            'accuracy': round(acc, 2),
            'precision_weighted': round(p_w, 4),
            'recall_weighted': round(r_w, 4),
            'f1_weighted': round(f1_w, 4),
            'recall_macro': round(r_m, 4),
        })

    # 逐类明细（供附录/复核使用）
    per_class = []
    for idx_str, cls in idx_to_class.items():
        i = int(idx_str)
        if counter_true.get(i, 0) == 0:
            continue
        hits = counter_hit.get(i, 0)
        total = counter_true[i]
        per_class.append({
            'class': cls,
            'head_code': head_of(cls),
            'original_samples': original.get(cls, 0),
            'tier': tier_of_class.get(cls, 'NA'),
            'test_samples': total,
            'correct': hits,
            'recall': round(100.0 * hits / total, 2),
        })
    per_class.sort(key=lambda d: (-d['test_samples'], d['class']))

    return rows, per_class, tier_of_class, original


# ============================================================
# Top-N 易混淆字头对
# ============================================================
def top_confusions(y_true, y_pred, idx_to_class, class_stats, top_n=10):
    pairs = collections.Counter()
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if t != p:
            pairs[(t, p)] += 1

    original = {c: int(v.get('original', 0)) for c, v in class_stats.items()}
    counter_true = collections.Counter(int(t) for t in y_true)

    rows = []
    for (t, p), cnt in pairs.most_common(top_n):
        tc = idx_to_class[str(t)]
        pc = idx_to_class[str(p)]
        total = counter_true.get(t, 0)
        rows.append({
            'rank': len(rows) + 1,
            'true_class': tc,
            'pred_class': pc,
            'true_head': head_of(tc),
            'pred_head': head_of(pc),
            'same_head': head_of(tc) == head_of(pc),
            'error_count': cnt,
            'true_class_test_samples': total,
            'error_rate_within_class_pct': round(100.0 * cnt / total, 1) if total else None,
            'true_class_original': original.get(tc, 0),
            'pred_class_original': original.get(pc, 0),
            'true_tier': tier_of(original.get(tc, 0)),
        })
    return rows, pairs


# ============================================================
# 配图
# ============================================================
def make_figures(out_dir, tier_rows, conf_rows, err_stats):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 1) 频段性能
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        labels = [r['tier_label'] for r in tier_rows]
        accs = [r['accuracy'] for r in tier_rows]
        prec = [r['precision_weighted'] * 100 for r in tier_rows]
        rec = [r['recall_weighted'] * 100 for r in tier_rows]
        x = np.arange(len(labels))
        w = 0.26
        ax.bar(x - w, accs, w, label='准确率', color='#4C72B0')
        ax.bar(x, prec, w, label='精确率', color='#DD8452')
        ax.bar(x + w, rec, w, label='召回率', color='#55A868')
        for i, r in enumerate(tier_rows):
            ax.text(i, -6.5, f"{r['num_classes_total']}类\n{r['num_test_samples']}张",
                    ha='center', va='top', fontsize=9, color='#555555')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(max(0, min(accs) - 18), max(accs) + 6)
        ax.set_ylabel('%', fontsize=11)
        ax.legend(fontsize=9, ncol=3, loc='upper center', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title('不同频段类别的性能分解', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'fig_longtail_tiers.png'), dpi=300)
        plt.close(fig)

        # 2) Top-10 易混淆字头对
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        names = [f"{r['true_class']} → {r['pred_class']}" for r in conf_rows][::-1]
        vals = [r['error_count'] for r in conf_rows][::-1]
        colors = ['#C44E52' if r['same_head'] else '#4C72B0' for r in conf_rows][::-1]
        ax.barh(np.arange(len(vals)), vals, color=colors)
        ax.set_yticks(np.arange(len(vals)))
        ax.set_yticklabels(names, fontsize=9)
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.012, i, str(v), va='center', fontsize=9)
        ax.set_xlabel('误判样本数', fontsize=11)
        ax.set_title('Top-10 易混淆字头对（红=同一字头的跨时期/异体混淆）', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'fig_top_confusions.png'), dpi=300)
        plt.close(fig)
        return True
    except Exception as e:                                   # 配图失败不影响数据产出
        print(f'  ⚠ 配图生成失败（不影响数据结果）：{e}')
        return False


def make_glyph_grid(out_dir, conf_rows, processed_root, per_side=3):
    """Top-10 混淆对的代表性字形对照图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image

        rows = conf_rows[:10]
        fig_h = 1.25 * len(rows)
        fig, axes = plt.subplots(len(rows), 2, figsize=(4.6, fig_h))
        for i, r in enumerate(rows):
            for j, cls in enumerate([r['true_class'], r['pred_class']]):
                ax = axes[i, j] if len(rows) > 1 else axes[j]
                ax.set_xticks([])
                ax.set_yticks([])
                d = os.path.join(processed_root, 'train', cls)
                shown = False
                if os.path.isdir(d):
                    fs = sorted([f for f in os.listdir(d)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    if fs:
                        img = Image.open(os.path.join(d, fs[0])).convert('L')
                        ax.imshow(np.array(img), cmap='gray', vmin=0, vmax=255)
                        shown = True
                if not shown:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                for side in ('top', 'bottom', 'left', 'right'):
                    ax.spines[side].set_color('#999999')
        axes[0][0].set_title('真实字头', fontsize=10)
        axes[0][1].set_title('被误判为', fontsize=10)
        if len(rows) > 1:
            for i, r in enumerate(rows):
                axes[i, 0].set_ylabel(f"{r['true_class']}", fontsize=8, rotation=0,
                                      labelpad=28, va='center')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'fig_confusion_glyphs.png'), dpi=300)
        plt.close(fig)
        return True
    except Exception as e:
        print(f'  ⚠ 字形对照图生成失败（不影响数据结果）：{e}')
        return False


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser(description='OBWX 最优模型误差结构分析')
    ap.add_argument('--model', default='EfficientNetV2-S', help='要分析的模型名')
    ap.add_argument('--checkpoint', default=None, help='检查点路径（默认为 best_<model>.pth）')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--device', default=None)
    ap.add_argument('--top-n', type=int, default=10)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--predictions', default=None, help='已保存的预测 npz（跳过重新推理）')
    ap.add_argument('--no-figures', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    config = _abs_config(get_config())
    set_seed(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    processed_root = config['data']['processed_root']
    with open(os.path.join(processed_root, 'class_mapping.json'), 'r', encoding='utf-8') as f:
        cm = json.load(f)
    class_to_idx, idx_to_class = cm['class_to_idx'], cm['idx_to_class']
    num_classes = len(class_to_idx)
    with open(os.path.join(processed_root, 'preprocess_stats.json'), 'r', encoding='utf-8') as f:
        class_stats = json.load(f)['class_stats']

    out_dir = args.out_dir or os.path.join(config['output']['results_path'], 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    model_tag = get_safe_name(args.model)

    print('=' * 72)
    print(f'OBWX 误差结构分析 | 模型：{args.model} | 类别数：{num_classes} | 设备：{device}')
    print('=' * 72)

    # ---------- 1. 推理 ----------
    npz_path = args.predictions or os.path.join(out_dir, f'predictions_{model_tag}.npz')
    if args.predictions and os.path.exists(args.predictions):
        z = np.load(args.predictions)
        y_true, y_pred, y_conf = z['y_true'], z['y_pred'], z['y_conf']
        print(f'  载入已有预测：{args.predictions}（{len(y_true)} 个样本）')
    else:
        ckpt = args.checkpoint or os.path.join(
            config['output']['model_save_path'], f'best_{model_tag}.pth')
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f'检查点不存在：{ckpt}')

        cfg = MODELS_CONFIG[args.model]
        model = cfg['class'](num_classes=num_classes, config=cfg.get('config', {}))
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        model.to(device).eval()
        print(f'  已载入检查点：{ckpt}')

        loader, _ = get_test_dataloader(config, batch_size=args.batch_size)
        y_true, y_pred, y_conf, dt = predict(model, loader, device)
        np.savez_compressed(npz_path, y_true=y_true, y_pred=y_pred, y_conf=y_conf)
        print(f'  测试集推理完成：{len(y_true)} 个样本，用时 {dt:.1f}s -> {npz_path}')

    # ---------- 2. 总体指标（与既有 result_*.json 交叉校验） ----------
    acc = accuracy_score(y_true, y_pred) * 100.0
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    overall = {
        'model': args.model, 'num_test_samples': int(len(y_true)),
        'test_accuracy': round(acc, 4), 'precision_weighted': round(p_w, 6),
        'recall_weighted': round(r_w, 6), 'f1_weighted': round(f1_w, 6),
    }
    print(f"  总体：Accuracy={acc:.2f}%  Precision={p_w:.4f}  Recall={r_w:.4f}  F1={f1_w:.4f}")

    # ---------- 3. 长尾分档 ----------
    tier_rows, per_class, tier_of_class, original = long_tail_analysis(
        y_true, y_pred, idx_to_class, class_stats, num_classes)
    print('\n  频段性能：')
    print(f"    {'频段':<22}{'类别数':>8}{'测试样本':>10}{'准确率%':>10}{'精确率':>10}{'召回率':>10}")
    for r in tier_rows:
        print(f"    {r['tier_label']:<22}{r['num_classes_total']:>8}{r['num_test_samples']:>10}"
              f"{r['accuracy']:>10.2f}{r['precision_weighted']:>10.4f}{r['recall_weighted']:>10.4f}")

    # ---------- 4. Top-N 混淆 ----------
    conf_rows, _ = top_confusions(y_true, y_pred, idx_to_class, class_stats, top_n=args.top_n)
    n_err = int((y_true != y_pred).sum())
    same_head_err = sum(r['error_count'] for r in conf_rows if r['same_head'])
    err_stats = {
        'total_errors': n_err,
        'topN_error_sum': sum(r['error_count'] for r in conf_rows),
        'topN_same_head_count': sum(1 for r in conf_rows if r['same_head']),
    }
    print(f'\n  Top-{args.top_n} 易混淆字头对（全部误判 {n_err} 例）：')
    print(f"    {'#':<4}{'真实字头':<12}{'被判为':<12}{'误判数':>8}{'同字头':>8}")
    for r in conf_rows:
        print(f"    {r['rank']:<4}{r['true_class']:<12}{r['pred_class']:<12}"
              f"{r['error_count']:>8}{'是' if r['same_head'] else '否':>8}")
    print(f"    -> 其中 {err_stats['topN_same_head_count']}/{len(conf_rows)} 组属同一字头编号")

    # ---------- 5. 落盘 ----------
    tier_csv = os.path.join(out_dir, f'longtail_tiers_{model_tag}.csv')
    with open(tier_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(tier_rows[0].keys()))
        w.writeheader()
        w.writerows(tier_rows)

    conf_csv = os.path.join(out_dir, f'top_confusions_{model_tag}.csv')
    with open(conf_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(conf_rows[0].keys()))
        w.writeheader()
        w.writerows(conf_rows)

    percls_csv = os.path.join(out_dir, f'per_class_detail_{model_tag}.csv')
    with open(percls_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(per_class[0].keys()))
        w.writeheader()
        w.writerows(per_class)

    summary = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': args.model,
        'device': str(device),
        'overall': overall,
        'tiers': tier_rows,
        'top_confusions': conf_rows,
        'error_stats': err_stats,
        'tier_definition': {
            'basis': '每个类别的原始（离线增强前）样本数',
            'head': 'original >= 50', 'mid': '15 <= original < 50', 'tail': 'original < 15',
        },
    }
    json_path = os.path.join(out_dir, f'error_analysis_{model_tag}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Markdown（论文可直接粘贴）
    md_path = os.path.join(out_dir, f'tables_{model_tag}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('**表7　不同频段类别的性能分析**\n\n')
        f.write('| 类别频段 | 类别数 | 测试样本数/张 | 准确率/% | 精确率 | 召回率 |\n')
        f.write('|---|---|---|---|---|---|\n')
        for r in tier_rows:
            f.write(f"| {r['tier_label']} | {r['num_classes_total']} | {r['num_test_samples']} | "
                    f"{r['accuracy']:.2f} | {r['precision_weighted']:.4f} | "
                    f"{r['recall_weighted']:.4f} |\n")
        f.write('\n**表8　Top-10 易混淆字头对**\n\n')
        f.write('| 序号 | 真实字头 | 被误判为 | 误判样本数 | 该字头测试样本数 | 组内错误率/% | 同一字头 |\n')
        f.write('|---|---|---|---|---|---|---|\n')
        for r in conf_rows:
            f.write(f"| {r['rank']} | {r['true_class']} | {r['pred_class']} | {r['error_count']} | "
                    f"{r['true_class_test_samples']} | {r['error_rate_within_class_pct']} | "
                    f"{'是' if r['same_head'] else '否'} |\n")
        f.write(f"\n> 全部误判样本 {n_err} 例；Top-10 合计 {err_stats['topN_error_sum']} 例，"
                f"其中 {err_stats['topN_same_head_count']} 组属同一字头编号。\n")

    figures = []
    if not args.no_figures:
        if make_figures(out_dir, tier_rows, conf_rows, err_stats):
            figures = ['fig_longtail_tiers.png', 'fig_top_confusions.png']
        if make_glyph_grid(out_dir, conf_rows, processed_root):
            figures.append('fig_confusion_glyphs.png')

    print('\n' + '=' * 72)
    print('输出文件：')
    for p in [tier_csv, conf_csv, percls_csv, json_path, md_path] + \
             [os.path.join(out_dir, x) for x in figures]:
        print(f'  {p}')
    print('=' * 72)


if __name__ == '__main__':
    main()
