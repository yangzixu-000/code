#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_period_images.py — OBWX 数据集「各分期图像数量」独立统计工具

设计约束
--------
1. 完全独立：只用 Python 标准库，不 import 项目内任何模块（utils / data / training 等），
   不读 config.yaml，不写任何数据目录，对原有程序零影响。
2. 只读：只遍历与读取，不创建 / 移动 / 删除 / 重命名任何文件。
3. 可移植：单文件，拷到服务器直接 `python count_period_images.py --root <数据根>` 即可。

为什么需要它
------------
论文表 2 的分期图像数为 6,204 + 22,389 + 41,040 + 4,415 = 74,048，
而 Total 行与全文口径为 74,047，差 1。原始分发包（OBWX.zip）与解压目录里
**不含任何时期标签文件**——目录按字头组织，时期信息只可能来自：
  (a) 文件名数字编码（前 3 位 = 来源纂集编号）
  (b) 服务器上的另一套目录布局（按时期分目录）
  (c) 数据集作者手上的时期标注表（CSV / JSON）
本工具三种来源都支持，并在无法判定时输出「指纹」供人工确认判别字段。

默认判别依据（已在本地 74,047 张图上实测）
------------------------------------------
文件名 stem 为 14 位纯数字，结构为：

    <来源3><定位5><序号2><字头4>      例：005 00033 01 0011.jpg

前 3 位（来源纂集编号）把全集干净地四等分，且互斥无重叠：

    001 → 11,488 张（全部 .PNG）
    004 → 13,832 张
    005 → 35,967 张（覆盖全部 6,650 个类别 = 通贯性汇编）
    006 → 12,760 张
    合计 = 74,047 ✓

注意：该字段是「来源纂集」而非「历史时期」——005 覆盖全部 6,650 类，
说明它不是单一时期。若要得到论文表 2 的时期口径，请用 --mode dir / --mode meta
提供真实时期标签，或用 --map 指定来源→时期的映射。

用法示例
--------
  # 1) 先做指纹，确认判别字段（推荐第一次运行）
  python count_period_images.py --root ./data/obwx_raw --fingerprint

  # 2) 按文件名来源编号统计（默认）
  python count_period_images.py --root ./data/obwx_raw

  # 3) 直接读压缩包，无需解压
  python count_period_images.py --zip ./data/OBWX.zip

  # 4) 服务器按时期分目录：./OBWX/<时期>/<字头>/*.jpg
  python count_period_images.py --root ./OBWX --mode dir

  # 5) 有时期标注表（CSV 两列 路径,时期 或 JSON {"路径":"时期"}）
  python count_period_images.py --root ./data/obwx_raw --mode meta --meta periods.csv

  # 6) 自定义来源→时期映射
  python count_period_images.py --root ./data/obwx_raw --map '{"001":"Oracle bone"}'

  # 7) 导出结果 + 与论文表 2 比对
  python count_period_images.py --root ./data/obwx_raw --json out.json --csv out.csv --compare-paper
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 来源纂集编号 → 默认显示名（未知语义前不臆断为时期，用中性名）
DEFAULT_SOURCE_NAMES = {
    "001": "Source-001 (PNG)",
    "004": "Source-004",
    "005": "Source-005",
    "006": "Source-006",
}

# 论文表 2 现值，仅用于 --compare-paper 比对，不参与统计
PAPER_TABLE2 = {
    "Oracle bone": (1848, 6204),
    "Bronze": (3339, 22389),
    "Warring States": (5003, 41040),
    "Small seal": (4031, 4415),
}
PAPER_TOTAL_IMAGES = 74047
PAPER_TOTAL_CLASSES = 6650

# 已实测的来源编号（用于校验：若出现新编号说明数据版本不同，需人工确认）
KNOWN_SOURCE_CODES = {"001", "004", "005", "006"}


# --------------------------------------------------------------------------- #
# 扫描：目录 / 压缩包
# --------------------------------------------------------------------------- #

def iter_images_from_dir(root):
    """遍历目录，产出 (相对路径, 类别目录名, 文件名)。只读。

    类别目录 = 图像文件的直接父目录，这是一条通用规则：
      obwx_raw            : <类别>/<文件>            -> 类别
      obwx_processed      : <划分>/<类别>/<文件>      -> 类别（不是划分）
      按时期组织的服务器布局: <时期>/<类别>/<文件>      -> 类别（不是时期）
    """
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        cls = "." if os.path.abspath(dirpath) == root else os.path.basename(dirpath)
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
                continue
            yield os.path.relpath(os.path.join(dirpath, fn), root), cls, fn


def iter_images_from_zip(zip_path):
    """遍历 zip，产出 (包内路径, 类别目录名, 文件名)。只读，不解压。"""
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                continue
            parts = [p for p in name.replace("\\", "/").split("/") if p]
            # 类别目录 = 直接父目录（与目录模式保持同一规则）
            cls = parts[-2] if len(parts) >= 2 else "."
            yield name, cls, parts[-1]


# --------------------------------------------------------------------------- #
# 判别：把每张图映射到一个分组标签
# --------------------------------------------------------------------------- #

def parse_stem_fields(stem):
    """解析 14 位数字文件名，返回 dict；非纯数字返回 None。"""
    if not stem.isdigit():
        return None
    n = len(stem)
    if n < 10:
        return None
    return {
        "src3": stem[0:3] if n >= 3 else "",
        "mid5": stem[3:8] if n >= 8 else "",
        "seq2": stem[8:10] if n >= 10 else "",
        "tail4": stem[10:14] if n >= 14 else "",
        "raw": stem,
    }


def make_resolver(mode, root, meta_path, mapping, pattern, level):
    """返回一个函数 (relpath, cls, filename) -> 分组标签 或 None（无法判定）。"""
    if mode == "filename":
        names = dict(DEFAULT_SOURCE_NAMES)
        if mapping:
            names.update(mapping)
        unknown_seen = set()

        def resolve(relpath, cls, filename):
            f = parse_stem_fields(os.path.splitext(filename)[0])
            if f is None:
                return None
            code = f["src3"]
            if code not in names:
                unknown_seen.add(code)
                return "UNKNOWN-src%s" % code
            return names[code]

        resolve.unknown_seen = unknown_seen
        return resolve

    if mode == "dir":
        def resolve(relpath, cls, filename):
            parts = relpath.replace("\\", "/").split("/")
            # level=1 → 顶层目录；level=2 → 第二层
            idx = max(0, min(level - 1, len(parts) - 2))
            return parts[idx] or "."

        resolve.unknown_seen = set()
        return resolve

    if mode == "meta":
        meta = load_meta(meta_path)
        miss = set()

        def resolve(relpath, cls, filename):
            key = relpath.replace("\\", "/")
            if key in meta:
                return meta[key]
            if filename in meta:
                return meta[filename]
            stem = os.path.splitext(filename)[0]
            if stem in meta:
                return meta[stem]
            miss.add(key)
            return None

        resolve.unknown_seen = miss
        return resolve

    if mode == "regex":
        rx = re.compile(pattern)

        def resolve(relpath, cls, filename):
            m = rx.search(relpath.replace("\\", "/"))
            if not m:
                return None
            return m.group(1) if m.groups() else m.group(0)

        resolve.unknown_seen = set()
        return resolve

    raise ValueError("未知 mode: %s" % mode)


def load_meta(path):
    """载入时期标注表：CSV（两列：路径/文件名, 时期）或 JSON（dict）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON 标注表应为 {\"路径\": \"时期\"} 形式的字典")
        return {k.replace("\\", "/"): v for k, v in data.items()}

    meta = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError("标注表为空: %s" % path)
    # 自动跳过表头
    start = 0
    head0 = rows[0][0].strip().lower()
    if head0 in ("path", "file", "filename", "image", "relpath", "路径", "文件名"):
        start = 1
    for r in rows[start:]:
        if len(r) < 2:
            continue
        meta[r[0].strip().replace("\\", "/")] = r[1].strip()
    return meta


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #

def collect(root=None, zip_path=None, resolver=None, by_split=False):
    """返回统计结构。只读。"""
    src = iter_images_from_zip(zip_path) if zip_path else iter_images_from_dir(root)

    total = 0
    by_group = Counter()
    classes_by_group = defaultdict(set)
    split_by_group = Counter()
    unresolved = []
    exts = Counter()
    all_classes = set()

    for relpath, cls, filename in src:
        total += 1
        all_classes.add(cls)
        exts[os.path.splitext(filename)[1].lower()] += 1
        g = resolver(relpath, cls, filename)
        if g is None:
            if len(unresolved) < 20:
                unresolved.append(relpath)
            g = "__UNRESOLVED__"
        by_group[g] += 1
        classes_by_group[g].add(cls)

        if by_split:
            first = relpath.replace("\\", "/").split("/")[0]
            if first in ("train", "val", "test", "训练", "验证", "测试"):
                split_by_group[(first, g)] += 1

    return {
        "total": total,
        "total_classes": len(all_classes),
        "by_group": by_group,
        "classes_by_group": {k: len(v) for k, v in classes_by_group.items()},
        "split_by_group": dict(split_by_group),
        "unresolved": unresolved,
        "exts": dict(exts),
        "unknown_seen": sorted(getattr(resolver, "unknown_seen", set())),
    }


def fingerprint(root=None, zip_path=None, top=12):
    """输出各候选判别字段的经验分布，用于人工确认哪个字段是时期。只读。"""
    src = iter_images_from_zip(zip_path) if zip_path else iter_images_from_dir(root)

    total = 0
    ext_c = Counter()
    src3_c = Counter()
    mid5_c = Counter()
    seq2_c = Counter()
    tail4_c = Counter()
    nondigit = []
    cls_of_src3 = defaultdict(set)
    topdirs = Counter()

    for relpath, cls, filename in src:
        total += 1
        ext = os.path.splitext(filename)[1].lower()
        ext_c[ext] += 1
        parts = relpath.replace("\\", "/").split("/")
        if len(parts) > 1:
            topdirs[parts[0]] += 1
        stem = os.path.splitext(filename)[0]
        f = parse_stem_fields(stem)
        if f is None:
            if len(nondigit) < 10:
                nondigit.append(relpath)
            continue
        src3_c[f["src3"]] += 1
        mid5_c[f["mid5"]] += 1
        seq2_c[f["seq2"]] += 1
        tail4_c[f["tail4"]] += 1
        cls_of_src3[f["src3"]].add(cls)

    return {
        "total": total,
        "non_image_or_odd": nondigit,
        "ext": ext_c,
        "src3": src3_c,
        "src3_class_count": {k: len(v) for k, v in cls_of_src3.items()},
        "mid5": mid5_c,
        "seq2": seq2_c,
        "tail4": tail4_c,
        "topdirs": topdirs,
        "top": top,
    }


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #

def hr(title=""):
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


def print_fingerprint(fp):
    hr("指纹：候选判别字段的经验分布（只读扫描）")
    print("扫描到图像总数 : {:,}".format(fp["total"]))
    print("扩展名分布     : {}".format(dict(fp["ext"])))
    if fp["non_image_or_odd"]:
        print("非数字文件名样例: {}".format(fp["non_image_or_odd"][:5]))

    print("\n[字段 A] stem 前 3 位（来源纂集编号）——四值互斥，最可能是判别字段")
    for k, v in sorted(fp["src3"].items()):
        flag = "" if k in KNOWN_SOURCE_CODES else "   <== 新编号！数据版本可能不同"
        print("   {:<6s} {:>8,d} 张   覆盖类别 {:>5,d}{}".format(
            k, v, fp["src3_class_count"].get(k, 0), flag))

    print("\n[字段 B] stem 第 4–8 位（册/页定位）——取值数 {}".format(len(fp["mid5"])))
    print("   top{}: {}".format(fp["top"], fp["mid5"].most_common(fp["top"])))

    print("\n[字段 C] stem 第 9–10 位（同页序号）——取值数 {}".format(len(fp["seq2"])))
    print("   top{}: {}".format(fp["top"], fp["seq2"].most_common(fp["top"])))

    print("\n[字段 D] stem 末 4 位（字头编号）——取值数 {}".format(len(fp["tail4"])))

    if fp["topdirs"]:
        print("\n[字段 E] 顶层目录——取值数 {}".format(len(fp["topdirs"])))
        print("   top{}: {}".format(fp["top"], fp["topdirs"].most_common(fp["top"])))
        if len(fp["topdirs"]) <= 12:
            print("   （若顶层就是时期目录，请用 --mode dir）")


def print_result(res, mode, by_split, compare_paper, expect_total):
    hr("统计结果（判别方式：%s）" % mode)
    groups = res["by_group"]
    order = sorted(groups.items(), key=lambda kv: -kv[1])

    print("{:<28s} {:>10s} {:>10s} {:>8s}".format("分组/时期", "图像数", "类别数", "占比"))
    print("-" * 62)
    for g, n in order:
        print("{:<28s} {:>10,d} {:>10,d} {:>7.2f}%".format(
            g[:28], n, res["classes_by_group"].get(g, 0),
            100.0 * n / res["total"] if res["total"] else 0.0))
    print("-" * 62)
    print("{:<28s} {:>10,d} {:>10,d}".format(
        "合计（去重类别）", sum(groups.values()), res["total_classes"]))
    print("注：分组类别数按组内去重统计，跨组会重复计入，故各分组之和通常大于合计。")

    # ---- 一致性校验 ----
    hr("一致性校验")
    s = sum(groups.values())
    print("分组求和 = {:,d}   扫描总数 = {:,d}   {} ".format(
        s, res["total"], "OK" if s == res["total"] else "不一致！"))
    if expect_total is not None:
        print("期望总数 = {:,d}   差值 = {:+d}   {}".format(
            expect_total, res["total"] - expect_total,
            "OK" if res["total"] == expect_total else "不一致！"))
    if res["unresolved"]:
        print("无法判期的图像 {} 张，样例: {}".format(
            len(res["unresolved"]), res["unresolved"][:3]))
    if res["unknown_seen"]:
        print("出现未知分组码 {} 个，样例: {}".format(
            len(res["unknown_seen"]), res["unknown_seen"][:5]))

    if by_split and res["split_by_group"]:
        hr("各划分（train/val/test）× 分组")
        print("{:<10s} {:<24s} {:>10s}".format("划分", "分组", "图像数"))
        print("-" * 48)
        for (sp, g), n in sorted(res["split_by_group"].items(), key=lambda kv: (kv[0][0], -kv[1])):
            print("{:<10s} {:<24s} {:>10,d}".format(sp, g[:24], n))

    if compare_paper:
        hr("与论文表 2 现值比对（仅供核对，不用于统计）")
        paper_sum = sum(v[1] for v in PAPER_TABLE2.values())
        print("论文表 2 分期图像数：")
        for k, (c, n) in PAPER_TABLE2.items():
            print("   {:<16s} 类别 {:>5,d}   图像 {:>6,d}".format(k, c, n))
        print("   论文分期求和 = {:,d}   论文 Total = {:,d}   差值 = {:+d}".format(
            paper_sum, PAPER_TOTAL_IMAGES, paper_sum - PAPER_TOTAL_IMAGES))
        print("\n实测扫描总数 = {:,d}".format(res["total"]))
        print("→ 论文 Total 与实测{}一致；分期求和比 Total 多 {} 张，".format(
            "" if res["total"] == PAPER_TOTAL_IMAGES else "不",
            paper_sum - PAPER_TOTAL_IMAGES))
        print("  说明表 2 中某一个分期数多了 1（需以本工具实测分期数为准逐项对照）。")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="OBWX 各分期图像数量统计（独立、只读、不影响原程序）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--root", help="图像根目录（如 ./data/obwx_raw）")
    src.add_argument("--zip", dest="zip_path", help="直接读压缩包（如 ./data/OBWX.zip），无需解压")

    ap.add_argument("--mode", default="filename",
                    choices=["filename", "dir", "meta", "regex"],
                    help="判别方式：filename=文件名数字编码(默认) / dir=目录名 / meta=标注表 / regex=正则")
    ap.add_argument("--meta", help="--mode meta 时的标注表路径（CSV 或 JSON）")
    ap.add_argument("--map", dest="mapping",
                    help='覆盖分组名，JSON 字符串，如 \'{"001":"Oracle bone"}\'')
    ap.add_argument("--pattern", help="--mode regex 时的正则（含 1 个捕获组）")
    ap.add_argument("--level", type=int, default=1,
                    help="--mode dir 时取第几层目录名作为分组（默认 1）")
    ap.add_argument("--fingerprint", action="store_true",
                    help="只输出各候选字段的经验分布，用于确认判别字段")
    ap.add_argument("--by-split", action="store_true",
                    help="若根目录下为 train/val/test，额外按划分交叉统计")
    ap.add_argument("--compare-paper", action="store_true",
                    help="附带输出与论文表 2 现值的比对")
    ap.add_argument("--expect-total", type=int, default=None,
                    help="期望图像总数（用于差值校验，如 74047）")
    ap.add_argument("--json", dest="json_out", help="结果导出 JSON 路径")
    ap.add_argument("--csv", dest="csv_out", help="结果导出 CSV 路径")
    args = ap.parse_args(argv)

    if not args.root and not args.zip_path:
        # 自动探测常见位置
        for cand in ("./data/obwx_raw", "./obwx_raw", "./OBWX", "."):
            if os.path.isdir(cand):
                args.root = cand
                print("[自动探测] 使用根目录: %s" % os.path.abspath(cand))
                break
        if not args.root:
            ap.error("请用 --root 或 --zip 指定数据位置")

    if args.zip_path and not os.path.isfile(args.zip_path):
        ap.error("压缩包不存在: %s" % args.zip_path)
    if args.root and not os.path.isdir(args.root):
        ap.error("目录不存在: %s" % args.root)

    location = ("zip: " + os.path.abspath(args.zip_path)) if args.zip_path \
        else ("dir: " + os.path.abspath(args.root))
    print("OBWX 分期图像统计工具（只读）")
    print("数据位置 : %s" % location)

    if args.fingerprint:
        fp = fingerprint(root=args.root, zip_path=args.zip_path)
        print_fingerprint(fp)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump({k: (dict(v) if isinstance(v, Counter) else v)
                           for k, v in fp.items()}, f, ensure_ascii=False, indent=2)
            print("\n[导出] %s" % args.json_out)
        return 0

    mapping = None
    if args.mapping:
        try:
            mapping = json.loads(args.mapping)
        except json.JSONDecodeError as e:
            ap.error("--map 不是合法 JSON: %s" % e)
    if args.mode == "meta" and not args.meta:
        ap.error("--mode meta 需要 --meta 指定标注表")
    if args.mode == "regex" and not args.pattern:
        ap.error("--mode regex 需要 --pattern")

    resolver = make_resolver(args.mode, args.root, args.meta,
                             mapping, args.pattern, args.level)
    res = collect(root=args.root, zip_path=args.zip_path,
                  resolver=resolver, by_split=args.by_split)
    print_result(res, args.mode, args.by_split, args.compare_paper, args.expect_total)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({
                "location": location,
                "mode": args.mode,
                "total": res["total"],
                "by_group": dict(res["by_group"]),
                "classes_by_group": res["classes_by_group"],
                "split_by_group": {f"{k[0]}|{k[1]}": v
                                   for k, v in res["split_by_group"].items()},
                "exts": res["exts"],
                "unresolved_sample": res["unresolved"],
            }, f, ensure_ascii=False, indent=2)
        print("\n[导出] %s" % args.json_out)

    if args.csv_out:
        with open(args.csv_out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "images", "classes", "pct"])
            for g, n in sorted(res["by_group"].items(), key=lambda kv: -kv[1]):
                w.writerow([g, n, res["classes_by_group"].get(g, 0),
                            round(100.0 * n / res["total"], 4) if res["total"] else 0])
            w.writerow(["TOTAL", sum(res["by_group"].values()), "", 100.0])
        print("[导出] %s" % args.csv_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
