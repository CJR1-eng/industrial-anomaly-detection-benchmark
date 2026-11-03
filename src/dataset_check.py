# -*- coding: utf-8 -*-
"""MVTec AD `tile` 类别数据集完整性校验。

校验项：
1. 目录结构是否符合官方 MVTec AD 约定（train/good, test/*, ground_truth/*）。
2. 各类别图像数量。
3. 图像分辨率分布、通道模式。
4. 每张异常测试图是否都有同名 _mask.png 掩码；掩码尺寸是否与图像一致。
5. 掩码是否非空（全 0 掩码说明标注缺失）。
6. 训练集纯度：train 下只应有 good，不得出现缺陷子目录。
7. train / test 文件名是否重叠（防止数据泄漏）。
8. 汇总像素级异常占比（缺陷区域平均面积占比）。

用法:
    python src/dataset_check.py --data-root datasets/raw/mvtec_ad --category tile
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paths import MVTEC_AD_URL, mvtec_ad_root, require_category  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def check(category_dir: Path) -> dict:
    from PIL import Image

    result: dict = {
        "category_dir": str(category_dir),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "exists": category_dir.is_dir(),
        "errors": [],
        "warnings": [],
    }
    if not category_dir.is_dir():
        result["errors"].append(f"类别目录不存在: {category_dir}")
        return result

    train_dir = category_dir / "train"
    test_dir = category_dir / "test"
    gt_dir = category_dir / "ground_truth"

    for name, d in [("train", train_dir), ("test", test_dir), ("ground_truth", gt_dir)]:
        if not d.is_dir():
            result["errors"].append(f"缺少目录: {d}")

    # ---------- 1. train ----------
    train_subsets = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    result["train"] = {"subsets": train_subsets, "counts": {}, "total": 0}
    for s in train_subsets:
        n = len(list_images(train_dir / s))
        result["train"]["counts"][s] = n
        result["train"]["total"] += n
    if train_subsets != ["good"]:
        result["errors"].append(f"train 下应只有 'good' 子目录，实际为 {train_subsets}（测试集泄漏风险）")

    # ---------- 2. test ----------
    test_subsets = sorted(p.name for p in test_dir.iterdir() if p.is_dir())
    result["test"] = {"subsets": test_subsets, "counts": {}, "total": 0}
    for s in test_subsets:
        n = len(list_images(test_dir / s))
        result["test"]["counts"][s] = n
        result["test"]["total"] += n

    good_n = result["test"]["counts"].get("good", 0)
    defect_n = result["test"]["total"] - good_n
    result["test"]["good_count"] = good_n
    result["test"]["defect_count"] = defect_n

    # ---------- 3. ground_truth ----------
    gt_subsets = sorted(p.name for p in gt_dir.iterdir() if p.is_dir()) if gt_dir.is_dir() else []
    result["ground_truth"] = {"subsets": gt_subsets, "counts": {}, "total": 0}
    for s in gt_subsets:
        n = len(list_images(gt_dir / s))
        result["ground_truth"]["counts"][s] = n
        result["ground_truth"]["total"] += n

    defect_subsets = [s for s in test_subsets if s != "good"]
    if sorted(defect_subsets) != sorted(gt_subsets):
        result["errors"].append(
            f"缺陷类别与 ground_truth 类别不一致: test={defect_subsets} gt={gt_subsets}"
        )

    # ---------- 4. 掩码对应关系 + 尺寸 ----------
    sizes: Counter = Counter()
    mask_sizes: Counter = Counter()
    modes: Counter = Counter()
    missing_masks: list[str] = []
    size_mismatch: list[str] = []
    empty_masks: list[str] = []
    area_ratios: list[float] = []

    for s in defect_subsets:
        imgs = list_images(test_dir / s)
        for img_path in imgs:
            with Image.open(img_path) as im:
                iw, ih = im.size
                modes[im.mode] += 1
            sizes[(iw, ih)] += 1

            mask_path = gt_dir / s / f"{img_path.stem}_mask.png"
            if not mask_path.exists():
                missing_masks.append(str(mask_path.relative_to(category_dir)))
                continue
            with Image.open(mask_path) as m:
                mw, mh = m.size
                mask_sizes[(mw, mh)] += 1
                if (mw, mh) != (iw, ih):
                    size_mismatch.append(
                        f"{img_path.name}: img={iw}x{ih} mask={mw}x{mh}"
                    )
                arr = np.array(m)
            # 掩码二值化（官方掩码为 0/255 单通道，个别为多通道）
            if arr.ndim == 3:
                arr = arr[..., 0]
            fg = int((arr > 0).sum())
            total_px = arr.shape[0] * arr.shape[1]
            if fg == 0:
                empty_masks.append(str(mask_path.relative_to(category_dir)))
            else:
                area_ratios.append(fg / total_px)

    # 正常测试图不应有掩码
    good_masks = list((gt_dir / "good").glob("*")) if (gt_dir / "good").is_dir() else []

    result["image_sizes"] = {f"{w}x{h}": c for (w, h), c in sorted(sizes.items())}
    result["mask_sizes"] = {f"{w}x{h}": c for (w, h), c in sorted(mask_sizes.items())}
    result["image_modes"] = dict(modes)
    result["missing_masks"] = missing_masks
    result["size_mismatch"] = size_mismatch
    result["empty_masks"] = empty_masks
    result["good_masks_found"] = len(good_masks)
    if missing_masks:
        result["errors"].append(f"{len(missing_masks)} 张缺陷图缺少掩码")
    if size_mismatch:
        result["errors"].append(f"{len(size_mismatch)} 张掩码尺寸与图像不一致")
    if empty_masks:
        result["warnings"].append(f"{len(empty_masks)} 张掩码全黑（可能是无缺陷或标注缺失）")
    if good_masks:
        result["errors"].append(f"ground_truth/good 下存在 {len(good_masks)} 个文件，不合规范")

    result["defect_area_ratio"] = {
        "mean": float(np.mean(area_ratios)) if area_ratios else None,
        "median": float(np.median(area_ratios)) if area_ratios else None,
        "min": float(np.min(area_ratios)) if area_ratios else None,
        "max": float(np.max(area_ratios)) if area_ratios else None,
    }

    # ---------- 5. train / test 是否重复使用同一张图（内容哈希比对） ----------
    # 注意：MVTec 的 train/good 与 test/good 都以 000.png 起编号，
    # 文件名相同不代表图像相同，必须比内容哈希。
    import hashlib

    def md5(p: Path) -> str:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()

    train_files = [p for p in train_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    test_files = [p for p in test_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]

    train_hashes: dict[str, str] = {}
    for p in train_files:
        train_hashes.setdefault(md5(p), str(p.relative_to(category_dir)))
    dup_content: list[str] = []
    for p in test_files:
        h = md5(p)
        if h in train_hashes:
            dup_content.append(f"{p.relative_to(category_dir)} == {train_hashes[h]}")

    name_collisions = sorted({p.name for p in train_files} & {p.name for p in test_files})
    result["train_test_name_collisions"] = len(name_collisions)  # 正常现象，仅记录
    result["train_test_duplicate_content"] = dup_content
    if dup_content:
        result["errors"].append(
            f"train 与 test 存在 {len(dup_content)} 组内容完全相同的图像（真实数据泄漏）"
        )

    # ---------- 6. 期望值对照（官方 tile.tar.xz 实际计数） ----------
    # train/good=230, test/good=33, test 缺陷=84（crack17 glue_strip18 gray_stroke16 oil18 rough15）
    EXPECTED = {"train_good": 230, "test_good": 33, "test_defect": 84}
    result["expected"] = EXPECTED
    result["matches_expected"] = {
        "train_good": result["train"]["counts"].get("good") == EXPECTED["train_good"],
        "test_good": good_n == EXPECTED["test_good"],
        "test_defect": defect_n == EXPECTED["test_defect"],
    }
    for k, v in result["matches_expected"].items():
        if not v:
            result["warnings"].append(f"计数与官方论文不一致: {k}")

    result["ok"] = len(result["errors"]) == 0
    return result


def to_markdown(r: dict) -> str:
    L = []
    L.append(f"# 数据集校验报告：{Path(r['category_dir']).name}\n")
    L.append(f"- 生成时间: {r['timestamp']}")
    L.append(f"- 目录: `{r['category_dir']}`")
    L.append(f"- 结论: **{'通过' if r.get('ok') else '存在问题'}**\n")

    if r["errors"]:
        L.append("## 错误\n")
        for e in r["errors"]:
            L.append(f"- {e}")
        L.append("")
    if r["warnings"]:
        L.append("## 警告\n")
        for w in r["warnings"]:
            L.append(f"- {w}")
        L.append("")

    L.append("## 样本数量\n")
    L.append("| 划分 | 子类 | 数量 |")
    L.append("| --- | --- | --- |")
    for s, n in r["train"]["counts"].items():
        L.append(f"| train | {s} | {n} |")
    for s, n in r["test"]["counts"].items():
        L.append(f"| test | {s} | {n} |")
    for s, n in r["ground_truth"]["counts"].items():
        L.append(f"| ground_truth | {s} | {n} |")
    L.append(f"| **合计** | train={r['train']['total']} / test={r['test']['total']} / gt={r['ground_truth']['total']} | |")
    L.append("")

    L.append("## 分辨率与通道\n")
    L.append(f"- 图像尺寸分布: {r['image_sizes']}")
    L.append(f"- 掩码尺寸分布: {r['mask_sizes']}")
    L.append(f"- 通道模式: {r['image_modes']}\n")

    dar = r["defect_area_ratio"]
    if dar["mean"] is not None:
        L.append("## 缺陷面积占比（像素级）\n")
        L.append(
            f"- mean={dar['mean']:.4f} / median={dar['median']:.4f} / "
            f"min={dar['min']:.4f} / max={dar['max']:.4f}"
        )
        L.append("")

    L.append("## 掩码对应性\n")
    L.append(f"- 缺失掩码: {len(r['missing_masks'])}")
    L.append(f"- 尺寸不一致: {len(r['size_mismatch'])}")
    L.append(f"- 全黑掩码: {len(r['empty_masks'])}")
    L.append(f"- train/test 同名文件（正常编号现象）: {r.get('train_test_name_collisions', 0)}")
    L.append(f"- train/test 内容重复（真实泄漏）: {len(r.get('train_test_duplicate_content', []))}\n")

    L.append("## 与官方计数对照\n")
    for k, v in r["matches_expected"].items():
        L.append(f"- {k}: {'一致' if v else '不一致'} (期望 {r['expected'][k]})")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(mvtec_ad_root()))
    ap.add_argument("--category", default="tile")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    try:
        cat_dir = require_category(data_root, args.category, "MVTec AD", MVTEC_AD_URL)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    r = check(cat_dir)

    out_json = PROJECT_ROOT / "results" / "metrics" / f"dataset_check_{args.category}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")

    md = to_markdown(r)
    out_md = PROJECT_ROOT / "results" / "metrics" / f"dataset_check_{args.category}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")

    print(md)
    print(f"[OK] JSON -> {out_json}")
    print(f"[OK] MD   -> {out_md}")
    return 0 if r.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
