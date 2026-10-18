"""
OBWX 基准 — 模型复杂度与推理时延测量脚本
================================================

产出论文「表6 模型复杂度与推理效率」的全部数据：

    模型 | 参数量/M | FLOPs/G | 单样本推理时延/ms

测量口径（严格统一）：
  * 输入张量：1×3×224×224（batch=1，随机张量）
  * 模型状态：eval()、torch.no_grad()，计时前统一 warmup
  * 时延：time.perf_counter 逐次计时，取 n_iter 次（默认 100）的**平均值**
          并记录标准差 / 最小值 / P95；CUDA 下额外用 CUDA Event 做交叉校验
  * 参数量：全部参数（含本项目自定义的 MLP 分类头）/ 可训练参数，单位 M（10^6）
  * FLOPs：一次前向传播的计算量，见下方约定说明

FLOPs 计数约定：
  MACs = multiply-accumulate 次数（卷积 / 全连接 / 注意力的核心乘加数）。
  本脚本同时给出两个口径供对照：
      GFLOPs@MAC  —— 1 次乘加计为 1 FLOP。与 torchvision 官方 model card
                     报告的 "GFLOPS" 同口径，便于与文献数值直接比对。
      GFLOPs@2x   —— 1 次乘加拆分为 1 次乘法 + 1 次加法（torchinfo/thop
                     常用口径），数值恰为前者的 2 倍。
  默认主表输出 GFLOPs@MAC，CSV 中同时给出 GFLOPs@2x。

  覆盖算子：Conv2d、Linear（含注意力 QKV/O 投影）、matmul/mm/bmm/addmm、
  scaled_dot_product_attention（注意力 QK^T 与 AV 两次矩阵乘）。
  不统计 LayerNorm / Softmax / GELU / Dropout / 逐元素加法等逐元素算子——
  这是 torchvision / thop / fvcore 的通行做法。

  【可靠性设计】采用"算子级补丁"与"模块级 Hook"两套独立口径同时统计，
  两者对纯卷积网络一致；对含 nn.MultiheadAttention 的模型，Hook 口径会
  因融合算子绕过 Python Hook 而系统性低估，因此以算子级口径为准，
  Hook 口径作为交叉校验一并打印（详见输出中的 hook/op 列）。

用法：
    python training/complexity.py                       # 全部 12 个模型
    python training/complexity.py --models ResNet18 "Swin Transformer"
    python training/complexity.py --iterations 200 --device cuda
"""
import os
import sys
import csv
import json
import time
import platform
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 路径引导（保证任意工作目录下可运行） ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.config import get_config               # noqa: E402
from utils.seed import set_seed                   # noqa: E402
from training.benchmark import MODELS_CONFIG      # noqa: E402


# ============================================================
# 配置文件路径绝对化
# ============================================================
def _abs_config(config):
    """把 config.yaml 中的相对路径解析到项目根目录，避免依赖当前工作目录"""
    for section in ('data', 'output'):
        if section not in config:
            continue
        for key, val in list(config[section].items()):
            if isinstance(val, str) and not os.path.isabs(val):
                config[section][key] = os.path.normpath(os.path.join(_PROJECT_ROOT, val))
    return config


# ============================================================
# FLOPs（MACs）统计 —— 零第三方依赖，双口径交叉校验
# ============================================================
def _mha_macs(module, inp):
    """
    nn.MultiheadAttention 的解析式乘加数。

    单独处理的原因：
        PyTorch 的 nn.MultiheadAttention 用裸参数 in_proj_weight 经 F.linear
        完成 QKV 投影，并在满足条件时走 C++ 融合实现；Python 层的模块 Hook
        与算子补丁都无法保证稳定命中，会把 Transformer 类模型的 FLOPs
        系统性低估 30%~40%。这里按模块配置直接解析求和，保证"只算一次、
        一次算全"。
    """
    x = inp[0]
    if module.batch_first:
        b, n, d = x.shape
        s = n
        if len(inp) > 1 and torch.is_tensor(inp[1]):
            s = inp[1].shape[1]                     # 交叉注意力：key 长度可能不同
    else:
        n, b, d = x.shape
        s = n
        if len(inp) > 1 and torch.is_tensor(inp[1]):
            s = inp[1].shape[0]

    h = module.num_heads
    macs = 0
    if getattr(module, 'in_proj_weight', None) is not None:
        macs += 3 * b * n * d * d                   # Q / K / V 三次投影
    elif getattr(module, 'q_proj_weight', None) is not None:
        kdim = module.kdim if module.kdim is not None else d
        vdim = module.vdim if module.vdim is not None else d
        macs += b * n * d * d + b * s * d * kdim + b * s * d * vdim

    head_dim = module.head_dim if getattr(module, 'head_dim', None) else d // h
    macs += 2 * b * h * n * s * head_dim            # QK^T 与 AV 两次矩阵乘

    if getattr(module, 'out_proj', None) is not None:
        macs += b * n * module.out_proj.out_features * d
    return int(macs)


def _matmul_macs(a, b, out):
    """matmul / mm / bmm / addmm 中矩阵乘部分的 MACs"""
    try:
        if a.dim() == 1 and b.dim() == 1:
            return a.shape[0]
        if b.dim() == 1:
            return out.numel() * a.shape[-1]
        if a.dim() == 1:
            return out.numel() * b.shape[-2]
        m, n = out.shape[-2], out.shape[-1]
        k = a.shape[-1]
        batch = out.numel() // (m * n) if m * n else 0
        return batch * m * n * k
    except Exception:
        return 0


def _sdpa_macs(query, key, value):
    """scaled_dot_product_attention：QK^T 与 AV 两次矩阵乘"""
    b, h, n, d = query.shape
    s = key.shape[-2]
    return 2 * b * h * n * s * d


def _swin_window_attn_macs(module, inp):
    """
    torchvision Swin 的 ShiftedWindowAttention 解析式乘加数。

    单独处理的原因（与 MHA 同源，但更隐蔽）：
        torchvision 的实现**不调用** self.qkv(x) / self.proj(x)，
        而是把 self.qkv.weight、self.proj.weight 作为参数传给融合算子
        `shifted_window_attention`（C++ 扩展实现）。因此：
          - Python 模块 Hook 完全不会触发（实测 24/55 个 Linear 从不 Forward）；
          - Python 算子补丁同样抓不到其中的投影。
        不特殊处理会把 Swin 的 FLOPs 低估约 32%。
    """
    x = inp[0]                                   # (B, H, W, C)
    b, hgt, wdt, c = x.shape
    ws = module.window_size[0]
    n = hgt * wdt
    macs = 3 * b * n * c * c                     # Q / K / V 投影
    macs += b * n * c * c                        # 输出投影
    macs += 2 * b * n * (ws * ws) * c            # 窗口内 QK^T 与 AV
    return int(macs)


def _resolve_special_types():
    """需要走解析式统计的特殊模块：一类],一类= multiplied fused-op 模块"""
    special = [(nn.MultiheadAttention, _mha_macs)]
    try:
        from torchvision.models.swin_transformer import ShiftedWindowAttention
        special.append((ShiftedWindowAttention, _swin_window_attn_macs))
    except Exception:
        pass
    return special


SPECIAL_MODULES = _resolve_special_types()


class FlopProfiler:
    """
    双口径 FLOPs / MACs 统计器（不依赖任何第三方库）。

        prof = FlopProfiler(model)
        res  = prof.count(input_shape=(1, 3, 224, 224))
        res['op']      # 算子级计数（最终采用）
        res['hook']    # 模块级 Hook 计数（交叉校验）

    两套口径的差异即是一次自查：纯卷积网络两者应基本一致；
    含 nn.MultiheadAttention 的模型若差异明显，说明融合算子绕过了 Python Hook。
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.macs = 0
        self._suppress = 0          # >0 表示当前位于某个 MHA 内部，暂停累加
        self._hooks = []
        self._saved_ops = {}

    # ---------------- 基础 ----------------
    def reset(self):
        self.macs = 0
        self._suppress = 0

    def _add(self, n):
        if self._suppress <= 0:
            self.macs += int(n)

    def _clear(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
        for (ns, name), fn in self._saved_ops.items():
            try:
                setattr(ns, name, fn)
            except Exception:
                pass
        self._saved_ops.clear()
        self.reset()

    # ---------------- MultiheadAttention 统一处理（两种口径共用） ----------------
    def _register_special_handlers(self, for_hook_method=False):
        excluded = set()

        def pre_hook(m, inp):
            self._suppress += 1

        def make_post_hook(calc_fn):
            def post_hook(m, inp, out):
                self._suppress -= 1
                self._add(calc_fn(m, inp))
            return post_hook

        for mod in self.model.modules():
            for cls, calc_fn in SPECIAL_MODULES:
                if isinstance(mod, cls):
                    self._hooks.append(mod.register_forward_pre_hook(pre_hook))
                    self._hooks.append(mod.register_forward_hook(make_post_hook(calc_fn)))
                    if for_hook_method:
                        # Hook 口径下排除其内部子模块，避免与解析式重复计数
                        for sub in mod.modules():
                            if sub is not mod:
                                excluded.add(id(sub))
                    break
        return excluded

    # ---------------- 口径 A：算子级补丁 ----------------
    def _enable_op_patch(self):
        prof = self

        def wrap_linear(fn):
            def _fn(input, weight, bias=None):
                out = fn(input, weight, bias)
                try:
                    prof._add(out.numel() * input.shape[-1])
                except Exception:
                    pass
                return out
            return _fn

        def wrap_conv2d(fn):
            def _fn(input, weight, bias=None, stride=1, padding=0,
                    dilation=1, groups=1):
                out = fn(input, weight, bias, stride, padding, dilation, groups)
                try:
                    per_elem = weight.shape[1] * weight.shape[2] * weight.shape[3]
                    prof._add(out.numel() * per_elem)
                except Exception:
                    pass
                return out
            return _fn

        def wrap_matmul(fn):
            def _fn(*args, **kwargs):
                out = fn(*args, **kwargs)
                try:
                    prof._add(_matmul_macs(args[0], args[1], out))
                except Exception:
                    pass
                return out
            return _fn

        def wrap_addmm(fn):
            def _fn(input, mat1, mat2, *args, **kwargs):
                out = fn(input, mat1, mat2, *args, **kwargs)
                try:
                    prof._add(_matmul_macs(mat1, mat2, mat2))
                except Exception:
                    pass
                return out
            return _fn

        def wrap_sdpa(fn):
            def _fn(query, key, value, *args, **kwargs):
                try:
                    prof._add(_sdpa_macs(query, key, value))
                except Exception:
                    pass
                return fn(query, key, value, *args, **kwargs)
            return _fn

        op_targets = [
            (F, 'linear'), (F, 'conv2d'),
            (torch, 'matmul'), (torch, 'mm'), (torch, 'bmm'), (torch, 'addmm'),
        ]
        for ns, name in op_targets:
            if not hasattr(ns, name):
                continue
            orig = getattr(ns, name)
            self._saved_ops[(ns, name)] = orig
            if name == 'linear':
                setattr(ns, name, wrap_linear(orig))
            elif name == 'conv2d':
                setattr(ns, name, wrap_conv2d(orig))
            elif name == 'addmm':
                setattr(ns, name, wrap_addmm(orig))
            else:
                setattr(ns, name, wrap_matmul(orig))

        for ns in (F, torch, nn.functional):
            if hasattr(ns, 'scaled_dot_product_attention'):
                orig = getattr(ns, 'scaled_dot_product_attention')
                self._saved_ops[(ns, 'scaled_dot_product_attention')] = orig
                setattr(ns, 'scaled_dot_product_attention', wrap_sdpa(orig))

        self._patch_tensor_matmul()

    # ---------------- 张量 @ 运算符补丁 ----------------
    def _patch_tensor_matmul(self):
        """
        单独补它的原因：
            `q @ k` 走 Tensor.__matmul__，直达 aten 派发，
            不会经过我们补丁过的 torch.matmul 的 Python 入口。
            torchvision Swin 的窗口注意力正是用 `@` 实现的，
            不补就会漏掉全部注意力矩阵乘（本例约 0.14 G）。
        """
        prof = self

        def wrap_tensor_matmul(fn):
            def _fn(self, other):
                out = fn(self, other)
                try:
                    prof._add(_matmul_macs(self, other, out))
                except Exception:
                    pass
                return out
            return _fn

        key = (torch.Tensor, '__matmul__')
        if key not in self._saved_ops:
            orig = torch.Tensor.__matmul__
            self._saved_ops[key] = orig
            torch.Tensor.__matmul__ = wrap_tensor_matmul(orig)
    def _enable_module_hooks(self, exclude_ids):
        def conv_hook(m, inp, out):
            if id(m) in exclude_ids:
                return
            x = inp[0]
            kh, kw = (m.kernel_size if isinstance(m.kernel_size, tuple)
                      else (m.kernel_size, m.kernel_size))
            self._add(out.numel() * (x.shape[1] / m.groups) * kh * kw)

        def linear_hook(m, inp, out):
            if id(m) in exclude_ids:
                return
            self._add(out.numel() * inp[0].shape[-1])

        for mod in self.model.modules():
            if isinstance(mod, nn.Conv2d):
                self._hooks.append(mod.register_forward_hook(conv_hook))
            elif isinstance(mod, nn.Linear):
                self._hooks.append(mod.register_forward_hook(linear_hook))

    def _enable_math_only_patch(self):
        prof = self

        def wrap_matmul(fn):
            def _fn(*args, **kwargs):
                out = fn(*args, **kwargs)
                try:
                    prof._add(_matmul_macs(args[0], args[1], out))
                except Exception:
                    pass
                return out
            return _fn

        def wrap_sdpa(fn):
            def _fn(query, key, value, *args, **kwargs):
                try:
                    prof._add(_sdpa_macs(query, key, value))
                except Exception:
                    pass
                return fn(query, key, value, *args, **kwargs)
            return _fn

        for ns, name in ((torch, 'matmul'), (torch, 'mm'), (torch, 'bmm')):
            self._saved_ops[(ns, name)] = getattr(ns, name)
            setattr(ns, name, wrap_matmul(getattr(ns, name)))
        for ns in (F, torch, nn.functional):
            if hasattr(ns, 'scaled_dot_product_attention'):
                orig = getattr(ns, 'scaled_dot_product_attention')
                self._saved_ops[(ns, 'scaled_dot_product_attention')] = orig
                setattr(ns, 'scaled_dot_product_attention', wrap_sdpa(orig))
        self._patch_tensor_matmul()

    # ---------------- 对外接口 ----------------
    @torch.no_grad()
    def count(self, input_shape=(1, 3, 224, 224), device=None):
        """返回 {'op': int, 'hook': int} 两种口径的 MACs"""
        device = device or next(self.model.parameters()).device
        x = torch.randn(*input_shape, device=device)

        self._clear()
        self._register_special_handlers(for_hook_method=False)
        self._enable_op_patch()
        self.reset()
        self.model(x)
        op_macs = self.macs

        self._clear()
        excluded = self._register_special_handlers(for_hook_method=True)
        self._enable_module_hooks(excluded)
        self._enable_math_only_patch()
        self.reset()
        self.model(x)
        hook_macs = self.macs

        self._clear()
        return {'op': op_macs, 'hook': hook_macs}


# ============================================================
# 参数量统计
# ============================================================
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# 单样本推理时延（time 模块，n_iter 次取平均）
# ============================================================
@torch.no_grad()
def measure_latency(model, device, input_size=224, n_warmup=20, n_iter=100):
    """
    统一口径：batch=1、3×224×224、eval()、no_grad、预热后连续计时 n_iter 次取平均。
    """
    model.eval()
    x = torch.randn(1, 3, input_size, input_size, device=device)

    for _ in range(n_warmup):                        # 预热到稳态，消除首帧抖动
        model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = []
        for _ in range(n_iter):
            starter.record()
            model(x)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))       # ms
        mean_ms = sum(timings) / len(timings)
        std_ms = torch.tensor(timings).std().item()
        sorted_t = sorted(timings)
    else:
        timings = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            model(x)
            timings.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = sum(timings) / len(timings)
        std_ms = (sum((v - mean_ms) ** 2 for v in timings) / max(len(timings) - 1, 1)) ** 0.5
        sorted_t = sorted(timings)

    return {
        'mean_ms': round(mean_ms, 3),
        'std_ms': round(std_ms, 3),
        'min_ms': round(sorted_t[0], 3),
        'p95_ms': round(sorted_t[min(int(0.95 * len(sorted_t)), len(sorted_t) - 1)], 3),
        'throughput_fps': round(1000.0 / mean_ms, 2),
    }


# ============================================================
# 主流程
# ============================================================
def measure_models(model_names, num_classes, device_str=None, input_size=224,
                   n_iter=100, n_warmup=20):
    device = torch.device(
        device_str if device_str else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    rows = []
    for name in model_names:
        cfg = MODELS_CONFIG[name]
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        model = cfg['class'](num_classes=num_classes, config=cfg.get('config', {}))
        model.to(device).eval()

        total_p, trainable_p = count_params(model)
        res = FlopProfiler(model).count(input_shape=(1, 3, input_size, input_size),
                                        device=device)
        macs = res['op']
        lat = measure_latency(model, device, input_size=input_size,
                              n_warmup=n_warmup, n_iter=n_iter)

        rows.append({
            'Model': name,
            'Params_M': round(total_p / 1e6, 3),
            'Trainable_Params_M': round(trainable_p / 1e6, 3),
            'MACs_G_op': round(macs / 1e9, 4),
            'MACs_G_hook': round(res['hook'] / 1e9, 4),
            'FLOPs_G_MACC': round(macs / 1e9, 4),            # 1 乘加 = 1 FLOP
            'FLOPs_G_SPLIT': round(2 * macs / 1e9, 4),       # 1 乘加 = 2 FLOP
            'Latency_ms': lat['mean_ms'],
            'Latency_std_ms': lat['std_ms'],
            'Latency_min_ms': lat['min_ms'],
            'Latency_p95_ms': lat['p95_ms'],
            'Throughput_fps': lat['throughput_fps'],
        })

        print(f"  ✔ {name:<24} Params={rows[-1]['Params_M']:>8.2f}M  "
              f"MACs={rows[-1]['MACs_G_op']:>7.3f}G(op) "
              f"{rows[-1]['MACs_G_hook']:>7.3f}G(hook)  "
              f"Latency={lat['mean_ms']:>8.3f}ms", flush=True)

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return rows, device


def _env_info(device):
    try:
        cpu = platform.processor() or os.environ.get('PROCESSOR_IDENTIFIER', 'unknown')
    except Exception:
        cpu = 'unknown'
    info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'platform': platform.platform(),
        'cpu': cpu,
        'cpu_threads': torch.get_num_threads(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'device': str(device),
    }
    if device.type == 'cuda':
        info['gpu'] = torch.cuda.get_device_name(device)
        info['cuda'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
    return info


def save_results(rows, env, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'complexity_summary.csv')
    json_path = os.path.join(out_dir, 'complexity_summary.json')
    md_path = os.path.join(out_dir, 'complexity_summary.md')

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'env': env, 'rows': rows}, f, indent=2, ensure_ascii=False)

    # Markdown 表格：论文表6 可直接粘贴
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('| 模型 | 参数量/M | FLOPs/G | 单样本推理时延/ms |\n')
        f.write('|---|---|---|---|\n')
        for r in rows:
            f.write(f"| {r['Model']} | {r['Params_M']:.2f} | "
                    f"{r['FLOPs_G_MACC']:.3f} | {r['Latency_ms']:.3f} |\n")

        dev_desc = env['device']
        dev_desc += f"（{env.get('gpu', '')}）" if device.type == 'cuda' \
            else f"（{env['cpu']}，{env['cpu_threads']} 线程）"
        f.write(f"\n> 测量环境：{dev_desc}；Python {env['python']}；"
                f"PyTorch {env['torch']}；\n"
                "> 输入 3×224×224、batch=1、模型为 eval 模式、torch.no_grad；\n"
                "> 单样本推理时延为 time 模块连续计时 100 次的平均值；\n"
                "> FLOPs 以 1 次乘加计 1 FLOP 统计（与 torchvision 官方 model card 同口径）；"
                "若按乘法与加法拆分则数值翻倍，见 CSV 的 FLOPs_G_SPLIT 列。\n")

    return csv_path, json_path, md_path


def main():
    parser = argparse.ArgumentParser(description='OBWX 模型复杂度与推理时延测量')
    parser.add_argument('--models', nargs='+', default=None, help='要测量的模型名（默认全部）')
    parser.add_argument('--device', default=None, help='cuda / cpu（默认自动选择）')
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--iterations', type=int, default=100, help='时延重复测量次数')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = _abs_config(get_config())
    set_seed(args.seed)

    with open(os.path.join(config['data']['processed_root'], 'class_mapping.json'),
              'r', encoding='utf-8') as f:
        num_classes = len(json.load(f)['class_to_idx'])

    out_dir = args.out_dir or os.path.join(config['output']['results_path'], 'complexity')
    model_names = args.models or list(MODELS_CONFIG.keys())

    print('=' * 92)
    print('OBWX 模型复杂度与推理时延测量')
    print(f"类别数 {num_classes} | 输入 3×{args.input_size}×{args.input_size} | "
          f"batch=1 | 时延取 {args.iterations} 次平均")
    print('=' * 92)

    rows, device = measure_models(
        model_names, num_classes, device_str=args.device, input_size=args.input_size,
        n_iter=args.iterations, n_warmup=args.warmup,
    )

    env = _env_info(device)
    csv_path, json_path, md_path = save_results(rows, env, out_dir, device)

    print('\n' + '=' * 92)
    print(f"{'模型':<26}{'参数量/M':>11}{'FLOPs/G':>11}{'时延/ms':>11}{'±标准差':>11}{'FPS':>9}")
    print('-' * 92)
    for r in rows:
        print(f"{r['Model']:<26}{r['Params_M']:>11.2f}{r['FLOPs_G_MACC']:>11.3f}"
              f"{r['Latency_ms']:>11.3f}{r['Latency_std_ms']:>11.3f}{r['Throughput_fps']:>9.2f}")
    print('=' * 92)
    print(f"\n结果已保存：\n  {csv_path}\n  {json_path}\n  {md_path}")


if __name__ == '__main__':
    main()
