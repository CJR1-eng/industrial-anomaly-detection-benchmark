# -*- coding: utf-8 -*-
"""任务 2/3：冻结 PaDiM 的光照扰动压力测试。

协议（严格遵循，任何一条变动都会让结果不可比）
----------------------------------------------
1. 模型**只拟合一次**（184 张 train/good），6 种条件全部复用同一份高斯参数。
2. 117 张测试图固定不变，标签固定不变。
3. 阈值只来自 46 张验证集正常图（`mean+3σ`），**绝不用测试集结果调阈值**。
4. 扰动只作用于 840×840 的 8-bit 原图，空间位置与 GT 完全不动。
5. 每种条件独立作用于原图，不累积。
6. 评估分辨率两套：256（模型原生）与 840（原图），掩码分别用 nearest 缩放 / 原始掩码。

用法
----
    python src/run_illumination.py                 # 正式运行
    python src/run_illumination.py --tag run2      # 复现性检查用，输出加后缀
    python src/run_illumination.py --conditions original,brightness_low
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_core as ec  # noqa: E402
import perturbations as pt  # noqa: E402


def compute_thresholds(model, dm, device: str) -> dict:
    """阈值只从验证集（46 张正常图）来。"""
    val = dm.val_data.samples
    tensors = []
    for i in range(len(val)):
        u8 = ec.load_image_u8(val.iloc[i]["image_path"])
        tensors.append(ec.to_model_input(u8))
    scores, maps = ec.infer_tensors(model, tensors, device=device)
    thr_img = float(scores.mean() + 3.0 * scores.std())
    all_pix = np.concatenate([m.reshape(-1) for m in maps])
    thr_pix = float(np.quantile(all_pix, 0.995))
    return {
        "n_val": int(len(scores)),
        "val_score_mean": float(scores.mean()),
        "val_score_std": float(scores.std()),
        "image_threshold_mean_plus_3std": thr_img,
        "pixel_threshold_p99.5": thr_pix,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="输出后缀，复现性检查用")
    ap.add_argument("--conditions", default=",".join(pt.CONDITION_NAMES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    tag = f"_{args.tag}" if args.tag else ""
    device = args.device if torch.cuda.is_available() else "cpu"

    det = ec.setup_determinism(args.seed)
    t_start = time.time()

    print("[1/5] 拟合 PaDiM（全场只拟合一次）...")
    model, dm, _engine = ec.fit_padim(seed=args.seed, device=device)
    manifest = ec.save_split_manifest(dm, args.seed, PROJECT_ROOT / "results" / "metrics" / "day2_split_manifest.csv")
    print(f"      划分: {manifest['counts']}")

    print("[2/5] 从验证集计算阈值...")
    thr = compute_thresholds(model, dm, device)
    print(f"      image thr={thr['image_threshold_mean_plus_3std']:.3f}, "
          f"pixel thr={thr['pixel_threshold_p99.5']:.4f}")

    samples = dm.test_data.samples
    n = len(samples)
    labels = np.array([1 if int(samples.iloc[i]["label_index"]) == 1 else 0 for i in range(n)])
    paths = [str(samples.iloc[i]["image_path"]) for i in range(n)]

    # 预读原图与原始掩码（840），6 种条件共享，避免重复 IO
    print(f"[3/5] 预读 {n} 张原图与原始 GT 掩码...")
    originals, masks840, has_mask = [], [], []
    for i in range(n):
        originals.append(ec.load_image_u8(paths[i]))
        mp = samples.iloc[i]["mask_path"]
        if labels[i] == 1 and isinstance(mp, str) and Path(mp).exists():
            masks840.append(ec.load_mask_bool(mp))
            has_mask.append(True)
        else:
            masks840.append(np.zeros((ec.IMG_SIZE, ec.IMG_SIZE), dtype=bool))
            has_mask.append(False)
    print(f"      含缺陷掩码: {sum(has_mask)} / {n}")

    masks256 = [ec.mask_to_size(m, ec.MODEL_SIZE, ec.InterpolationMode.NEAREST) for m in masks840]

    print("[4/5] 逐条件推理与评估...")
    rows, per_image_rows = [], []
    all_maps, all_scores = {}, {}
    for cond in conds:
        t0 = time.time()
        tensors, clip_stats = [], []
        for i in range(n):
            pert, st = pt.apply_condition(originals[i], cond)
            tensors.append(ec.to_model_input(pert))
            clip_stats.append(st)

        scores, maps = ec.infer_tensors(model, tensors, device=device)

        # --- 256 尺度 ---
        m256 = ec.pixel_metrics(list(maps), masks256)
        aupro256 = ec.aupro(list(maps), masks256)
        pi_auroc = ec.pixel_auroc_per_image(list(maps), masks256)
        img_m = ec.image_metrics(scores, labels)
        thm = ec.threshold_metrics(scores, labels, thr["image_threshold_mean_plus_3std"])

        # --- 840 尺度：分数图双线性上采样，掩码保持原始二值 ---
        maps840 = [ec.upsample_map(m, ec.IMG_SIZE) for m in maps]
        m840 = ec.pixel_metrics(maps840, masks840)
        del maps840

        rows.append(
            {
                "condition": cond,
                "family": next(c.family for c in pt.CONDITIONS if c.name == cond),
                "n_images": n,
                "n_normal": int((labels == 0).sum()),
                "n_defect": int((labels == 1).sum()),
                **img_m,
                **{f"{k}": v for k, v in m256.items()},
                **{f"{k}_840": v for k, v in m840.items()},
                "pixel_auroc_per_image_256": pi_auroc,
                "pixel_aupro_256": aupro256,
                **thm,
                "mean_clip_low_ratio": float(np.mean([c["clip_low_ratio"] for c in clip_stats])),
                "mean_clip_high_ratio": float(np.mean([c["clip_high_ratio"] for c in clip_stats])),
                "mean_intensity_delta": float(np.mean([c["mean_intensity_delta"] for c in clip_stats])),
                "elapsed_s": round(time.time() - t0, 1),
            }
        )

        all_maps[cond] = np.asarray(maps, dtype=np.float16)
        all_scores[cond] = scores

        for i in range(n):
            per_image_rows.append(
                {
                    "condition": cond,
                    "image_path": paths[i],
                    "gt_label": int(labels[i]),
                    "score_raw": float(scores[i]),
                    "pred_at_val_thr": int(scores[i] >= thr["image_threshold_mean_plus_3std"]),
                    "anomaly_map_max": float(maps[i].max()),
                    "anomaly_map_mean": float(maps[i].mean()),
                    "anomaly_map_mean_in_gt": float(maps[i][masks256[i]].mean()) if masks256[i].any() else float("nan"),
                    "gt_area_ratio_256": float(masks256[i].mean()),
                    "clip_low_ratio": clip_stats[i]["clip_low_ratio"],
                    "clip_high_ratio": clip_stats[i]["clip_high_ratio"],
                    "mean_intensity_delta": clip_stats[i]["mean_intensity_delta"],
                }
            )
        print(f"      {cond:16s} imageAUROC={img_m['image_auroc']:.4f} pxAUROC256={m256['pixel_auroc']:.4f} "
              f"pxAUROC840={m840['pixel_auroc']:.4f} ({rows[-1]['elapsed_s']}s)")

    print("[5/5] 落盘...")
    import pandas as pd

    met_dir = PROJECT_ROOT / "results" / "metrics"
    met_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    # 与原条件做差
    base = df[df["condition"] == "original"]
    if not base.empty:
        b = base.iloc[0]
        for col in ["image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "pixel_auroc_840",
                    "pixel_ap_840", "pixel_aupro_256", "fpr_normal", "recall_defect"]:
            df[f"delta_{col}"] = df[col] - b[col]
    df.to_csv(met_dir / f"day2_illumination_results{tag}.csv", index=False, float_format="%.6f")

    per_df = pd.DataFrame(per_image_rows)
    per_df.to_csv(met_dir / f"day2_per_image{tag}.csv", index=False, float_format="%.6f")

    np.savez_compressed(
        met_dir / f"day2_maps{tag}.npz",
        conditions=np.array(conds),
        maps=np.stack([all_maps[c] for c in conds]),
        scores=np.stack([all_scores[c] for c in conds]),
        labels=labels.astype(np.uint8),
        paths=np.array(paths),
    )

    summary = {
        "tag": args.tag,
        "determinism": det,
        "thresholds": thr,
        "conditions": pt.condition_metadata(),
        "split": manifest,
        "elapsed_total_s": round(time.time() - t_start, 1),
        "device": device,
    }
    (PROJECT_ROOT / "results" / "logs" / f"day2_run{tag}.json").parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "results" / "logs" / f"day2_run{tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n| 条件 | Image AUROC | Δ | Pixel AUROC@256 | Δ | Pixel AUROC@840 | Δ | 误报率 | 召回率 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        d_img = r.get("delta_image_auroc", 0.0)
        d_p = r.get("delta_pixel_auroc", 0.0)
        d_p8 = r.get("delta_pixel_auroc_840", 0.0)
        print(f"| {r['condition']} | {r['image_auroc']:.4f} | {d_img:+.4f} | {r['pixel_auroc']:.4f} | "
              f"{d_p:+.4f} | {r['pixel_auroc_840']:.4f} | {d_p8:+.4f} | {r['fpr_normal']:.3f} | {r['recall_defect']:.3f} |")
    print(f"\n[OK] 总耗时 {summary['elapsed_total_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
