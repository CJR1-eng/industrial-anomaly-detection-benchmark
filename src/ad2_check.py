# -*- coding: utf-8 -*-
"""MVTec AD 2 数据核查：解压、结构、划分、掩码、光照信息探查。

MVTec AD 2 官方划分（据官方下载页说明）
--------------------------------------
train                无缺陷训练图
validation           无缺陷验证图
test_public          各光照条件测试图 + 像素级 GT（本地可评估）
test_private         常规光照测试图（GT 不公开，只能走评测服务器）
test_private_mixed   与 test_private 同场景、不同光照（GT 不公开）

用法:
    python src/ad2_check.py --archive datasets/raw/ad2/sheet_metal.tar.gz
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paths import ad2_root  # noqa: E402
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def extract(arc: Path, out: Path) -> Path:
    tag = out / (".extracted_" + arc.name.replace(".", "_"))
    if tag.exists():
        print(f"[SKIP] 已解压: {out}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    print(f"解压 {arc.name} -> {out} ...")
    with tarfile.open(arc, "r:*") as tf:
        tf.extractall(out, filter="data")
    tag.write_text("ok", encoding="utf-8")
    print("[OK] 解压完成")
    return out


def image_files(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)


def analyse(cat_dir: Path) -> dict:
    from PIL import Image

    r: dict = {"category_dir": str(cat_dir), "exists": cat_dir.is_dir(), "errors": [], "warnings": []}
    if not cat_dir.is_dir():
        r["errors"].append(f"目录不存在: {cat_dir}")
        return r

    top = sorted(p.name for p in cat_dir.iterdir() if p.is_dir())
    r["top_level_dirs"] = top

    # ---- 顶层划分计数 ----
    split_counts: dict[str, dict[str, int]] = {}
    for split in top:
        subs = sorted(p.name for p in (cat_dir / split).iterdir() if p.is_dir())
        cnt = {s: len(image_files(cat_dir / split / s)) for s in subs}
        n_direct = len([p for p in (cat_dir / split).iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if n_direct:
            cnt["<direct>"] = n_direct
        split_counts[split] = cnt
    r["split_counts"] = split_counts

    # ---- test_public 细节（唯一有 GT 的测试集）----
    tp = cat_dir / "test_public"
    if tp.is_dir():
        subs = sorted(p.name for p in tp.iterdir() if p.is_dir())
        r["test_public_subsets"] = subs
        gt = cat_dir / "test_public" / "ground_truth"
        r["test_public_has_gt_dir"] = gt.is_dir()
        if gt.is_dir():
            r["test_public_gt_counts"] = {
                s: len(image_files(gt / s)) for s in sorted(p.name for p in gt.iterdir() if p.is_dir())
            }

    # ---- 分辨率分布（抽样，最多 120 张）----
    sizes: Counter = Counter()
    modes: Counter = Counter()
    all_imgs = image_files(cat_dir)
    for p in all_imgs[:120]:
        try:
            with Image.open(p) as im:
                sizes[f"{im.size[0]}x{im.size[1]}"] += 1
                modes[im.mode] += 1
        except Exception:  # noqa: BLE001
            continue
    r["image_size_distribution_sample"] = dict(sizes.most_common(10))
    r["image_mode_distribution_sample"] = dict(modes)
    r["n_images_total"] = len(all_imgs)

    # ---- 文件名模式：探查是否编码光照条件 ----
    # 取 test_public 下某子集的若干文件名
    name_samples = {}
    if tp.is_dir():
        for s in sorted(p.name for p in tp.iterdir() if p.is_dir())[:4]:
            fs = image_files(tp / s)[:6]
            name_samples[s] = [f.name for f in fs]
    r["filename_samples_test_public"] = name_samples

    # 关键词扫描（光照相关）
    kw = ["light", "lighting", "illum", "bright", "dark", "shadow", "expos", "glare", "backlight", "mixed"]
    hits: dict[str, int] = defaultdict(int)
    for p in all_imgs:
        low = str(p).lower()
        for k in kw:
            if k in low:
                hits[k] += 1
    r["lighting_keyword_hits"] = dict(hits)

    # 文件名里是否有可解析的数值模式（常用来编码场景/光照编号）
    pat = re.compile(r"(\d{3,})")
    nums = []
    if tp.is_dir():
        for s in sorted(p.name for p in tp.iterdir() if p.is_dir())[:3]:
            for f in image_files(tp / s)[:20]:
                m = pat.search(f.stem)
                if m:
                    nums.append(m.group(1))
    r["numeric_tokens_sample"] = nums[:20]

    if not r.get("test_public_has_gt_dir"):
        r["warnings"].append("未在 test_public 下找到 ground_truth 目录，需人工确认掩码位置")
    r["ok"] = len(r["errors"]) == 0
    return r


def to_md(r: dict, cat: str) -> str:
    L = [f"# MVTec AD 2 数据核查：{cat}\n"]
    L.append(f"- 目录: `{r['category_dir']}`")
    L.append(f"- 图像总数: {r.get('n_images_total')}")
    L.append(f"- 顶层目录: {r.get('top_level_dirs')}\n")

    if r["errors"]:
        L.append("## 错误\n")
        L += [f"- {e}" for e in r["errors"]]
        L.append("")

    L.append("## 各划分样本数\n")
    L.append("| 划分 | 子集 | 数量 |")
    L.append("| --- | --- | --- |")
    for split, cnt in r.get("split_counts", {}).items():
        for s, n in cnt.items():
            L.append(f"| {split} | {s} | {n} |")
    L.append("")

    L.append("## 分辨率与通道（抽样 ≤120 张）\n")
    L.append(f"- 尺寸分布: {r.get('image_size_distribution_sample')}")
    L.append(f"- 通道模式: {r.get('image_mode_distribution_sample')}\n")

    L.append("## test_public（唯一可本地评估的测试集）\n")
    L.append(f"- 子集: {r.get('test_public_subsets')}")
    L.append(f"- 含 ground_truth 目录: {r.get('test_public_has_gt_dir')}")
    L.append(f"- GT 计数: {r.get('test_public_gt_counts')}\n")

    L.append("## 光照条件线索探查\n")
    L.append(f"- 文件名关键词命中: {r.get('lighting_keyword_hits')}")
    L.append(f"- 文件名示例: {json.dumps(r.get('filename_samples_test_public'), ensure_ascii=False)}\n")

    if r.get("warnings"):
        L.append("## 警告\n")
        L += [f"- {w}" for w in r["warnings"]]
        L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(ad2_root() / "sheet_metal.tar.gz"))
    ap.add_argument("--out", default=str(ad2_root()))
    args = ap.parse_args()

    arc = Path(args.archive)
    out = Path(args.out)
    if not arc.exists():
        print(f"[ERR] 找不到压缩包: {arc}")
        return 1
    print(f"压缩包: {arc} ({arc.stat().st_size/1024**2:.1f} MB)")
    extract(arc, out)

    # 定位类别目录
    cand = [p for p in out.iterdir() if p.is_dir() and (p / "train").is_dir()]
    if not cand:
        # 可能多一层
        cand = [p for p in out.rglob("*") if p.is_dir() and (p / "train").is_dir()]
    if not cand:
        print("[ERR] 未找到含 train/ 的类别目录")
        print("顶层内容:", [p.name for p in out.iterdir()])
        return 2
    cat_dir = cand[0]
    cat = cat_dir.name
    print(f"类别目录: {cat_dir}")

    r = analyse(cat_dir)
    r["category"] = cat
    mdir = PROJECT_ROOT / "results" / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"ad2_check_{cat}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    ddir = PROJECT_ROOT / "results" / "metrics"
    ddir.mkdir(parents=True, exist_ok=True)
    md = to_md(r, cat)
    (ddir / f"ad2_check_{cat}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0 if r.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())
