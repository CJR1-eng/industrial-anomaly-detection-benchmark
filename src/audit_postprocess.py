# -*- coding: utf-8 -*-
"""任务 1.1：审计 anomalib 2.7.0 的后处理与评估口径。

要回答的四个问题
----------------
Q1 模型原始 anomaly_map 是什么？（单位：马氏距离，非归一化）
Q2 PostProcessor 用的是哪套统计量？对分数做了什么变换？（不是逐图 Min-Max！）
Q3 图像级分数原始 vs 后处理为何不同？(0.8445 -> 0.8849)
Q4 像素级 AUROC 为何大幅下降？(0.8387 -> 0.5889)

额外必须查清的一件事故
----------------------
Day 1 用 `infer_raw(model, dm.test_dataloader())` 直接取 batch —— 但 anomalib 的
[Resize + Normalize] 中，**Resize 被注入数据集、Normalize 由 PreProcessor 回调施加**。
若直接遍历 dataloader 而不经过回调，模型可能吃到"已 resize 但未归一化"的 [0,1] 张量。
本脚本给出判定证据（数字），不靠推测。

用法:
    python src/audit_postprocess.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_core as ec  # noqa: E402

OUT_JSON = PROJECT_ROOT / "results" / "metrics" / "day2_audit.json"


def describe(name: str, t: torch.Tensor) -> dict:
    x = t.detach().float()
    return {
        "name": name,
        "shape": list(x.shape),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
    }


def main() -> int:
    det = ec.setup_determinism(42)
    report: dict = {"determinism": det, "steps": {}}

    print("=" * 78)
    print("A. 数据管线对齐核查")
    print("=" * 78)

    # ---- A1: fit 之前的数据集样本 ----
    dm = ec.build_datamodule(seed=42)
    dm.setup()
    item0 = dm.test_data[0]
    a1 = describe("pre-fit dataset item", item0.image)
    print(f"[A1] fit 前 dataset[0].image : shape={a1['shape']} min={a1['min']:.4f} max={a1['max']:.4f}")
    print(f"     dataset.augmentations = {dm.test_data.augmentations}")

    # ---- A2: fit 之后取一个 batch ----
    model, dm, engine = ec.fit_padim(seed=42)
    print(f"[A2] fit 后 model.pre_processor.transform = {model.pre_processor.transform}")

    dm.setup()
    batch = next(iter(dm.test_dataloader()))
    a2 = describe("post-fit dataloader batch", batch.image)
    print(f"     dataloader batch       : shape={a2['shape']} min={a2['min']:.4f} max={a2['max']:.4f}")
    print(f"     dataset.augmentations  = {dm.test_data.augmentations}")

    # ---- A3: 我显式重建的管线 ----
    path0 = dm.test_data.samples.iloc[0]["image_path"]
    u8 = ec.load_image_u8(path0)
    x_mine = ec.to_model_input(u8)
    a3 = describe("explicit pipeline", x_mine)
    print(f"[A3] 显式管线 TRANSFORM(u8)  : shape={a3['shape']} min={a3['min']:.4f} max={a3['max']:.4f}")

    # 与 batch 中的第 0 张比较
    b0 = batch.image[0]
    diff_no_norm = float((b0 - x_mine).abs().max())
    # 与"再补一次 Normalize"比较
    from torchvision.transforms import v2 as T

    x_norm_again = T.Normalize(mean=ec.IMAGENET_MEAN, std=ec.IMAGENET_STD)(b0.float() / 1.0)
    print(f"[A3] |batch[0] - 显式管线| max = {diff_no_norm:.6f}")
    print(f"     batch[0] 是否已归一化？ min={float(b0.min()):.4f}（归一化后应出现负值）")

    pipeline_verdict = {
        "batch_shape": a2["shape"],
        "batch_min": a2["min"],
        "batch_max": a2["max"],
        "explicit_min": a3["min"],
        "explicit_max": a3["max"],
        "max_abs_diff_batch_vs_explicit": diff_no_norm,
        "batch_looks_normalized": bool(a2["min"] < -0.1),
    }
    report["steps"]["A_pipeline"] = {
        "pre_fit_dataset_item": a1,
        "post_fit_batch": a2,
        "explicit_pipeline": a3,
        "dataset_augmentations_after_fit": str(dm.test_data.augmentations),
        "verdict": pipeline_verdict,
    }

    # ---- A4: 两种输入喂同一模型，比较输出差异 ----
    model.eval()
    with torch.no_grad():
        out_batch = model.model(b0.unsqueeze(0).cuda())
        out_mine = model.model(x_mine.unsqueeze(0).cuda())
    s_batch = float(out_batch.pred_score.reshape(-1)[0])
    s_mine = float(out_mine.pred_score.reshape(-1)[0])
    map_diff = float((out_batch.anomaly_map - out_mine.anomaly_map).abs().max())
    print(f"[A4] 同一张图，输入不同 -> 图像分数: batch口径={s_batch:.3f} / 显式管线={s_mine:.3f}")
    print(f"     热力图最大绝对差 = {map_diff:.3f}")
    report["steps"]["A4_model_input_effect"] = {
        "score_with_dataloader_batch": s_batch,
        "score_with_explicit_pipeline": s_mine,
        "anomaly_map_max_abs_diff": map_diff,
    }

    print()
    print("=" * 78)
    print("B. PostProcessor 审计")
    print("=" * 78)
    pp = model.post_processor
    pp_state = {
        "enable_normalization": bool(pp.enable_normalization),
        "enable_thresholding": bool(pp.enable_thresholding),
        "image_sensitivity": float(pp.image_sensitivity),
        "pixel_sensitivity": float(pp.pixel_sensitivity),
        "image_min": float(pp.image_min),
        "image_max": float(pp.image_max),
        "image_threshold": float(pp.image_threshold),
        "pixel_min": float(pp.pixel_min),
        "pixel_max": float(pp.pixel_max),
        "pixel_threshold": float(pp.pixel_threshold),
        "normalized_image_threshold": float(pp.normalized_image_threshold),
        "normalized_pixel_threshold": float(pp.normalized_pixel_threshold),
    }
    print(json.dumps(pp_state, indent=2, ensure_ascii=False))
    report["steps"]["B_postprocessor_state"] = pp_state

    # ---- 用同一批测试样本，取原始输出 ----
    samples = dm.test_data.samples
    test_tensors, test_labels, test_paths = [], [], []
    masks256_nearest = []
    for i in range(len(samples)):
        row = samples.iloc[i]
        u8 = ec.load_image_u8(row["image_path"])
        test_tensors.append(ec.to_model_input(u8))
        test_labels.append(1 if int(row["label_index"]) == 1 else 0)
        test_paths.append(str(row["image_path"]))
        if int(row["label_index"]) == 1 and isinstance(row["mask_path"], str):
            masks256_nearest.append(ec.mask_to_size(ec.load_mask_bool(row["mask_path"]), 256, ec.InterpolationMode.NEAREST))
        else:
            masks256_nearest.append(np.zeros((256, 256), dtype=bool))

    raw_scores, raw_maps = ec.infer_tensors(model, test_tensors)
    labels = np.asarray(test_labels)
    print(f"[B1] 测试集原始输出: scores [{raw_scores.min():.2f}, {raw_scores.max():.2f}], "
          f"maps [{raw_maps.min():.2f}, {raw_maps.max():.2f}]")

    # ---- 用 PostProcessor 的公开接口做归一化 ----
    from anomalib.data import InferenceBatch

    t_scores = torch.from_numpy(raw_scores).float().unsqueeze(1)
    t_maps = torch.from_numpy(raw_maps).float().unsqueeze(1)
    pp_dev = next(pp.buffers()).device
    with torch.no_grad():
        pp_out = pp(InferenceBatch(pred_score=t_scores.to(pp_dev), anomaly_map=t_maps.to(pp_dev)))
    norm_scores = pp_out.pred_score.reshape(-1).cpu().numpy().astype(np.float64)
    norm_maps = pp_out.anomaly_map[:, 0].cpu().numpy().astype(np.float32)

    # ---- 验证我对公式的理解：clamp((x - thr)/(max-min) + 0.5, 0, 1) ----
    def manual_norm(x, lo, hi, thr):
        rng = hi - lo
        safe = rng if rng != 0 else 1.0
        return np.clip((x - thr) / safe + 0.5, 0.0, 1.0)

    man_scores = manual_norm(raw_scores, pp_state["image_min"], pp_state["image_max"], pp_state["image_threshold"])
    man_maps = manual_norm(raw_maps, pp_state["pixel_min"], pp_state["pixel_max"], pp_state["pixel_threshold"])
    score_formula_diff = float(np.abs(man_scores - norm_scores).max())
    map_formula_diff = float(np.abs(man_maps - norm_maps).max())
    print(f"[B2] 公式复现误差: image={score_formula_diff:.3e}, pixel={map_formula_diff:.3e} "
          f"(≈0 说明 PostProcessor = 全局仿射 + clamp)")

    # ---- 截断/并列分析 ----
    tie = {
        "raw_scores_unique": int(np.unique(raw_scores).size),
        "norm_scores_unique": int(np.unique(norm_scores).size),
        "norm_scores_at_0": float((norm_scores == 0).mean()),
        "norm_scores_at_1": float((norm_scores == 1).mean()),
        "raw_pixels_unique": int(np.unique(raw_maps[0]).size),
        "norm_pixels_at_0": float((norm_maps == 0).mean()),
        "norm_pixels_at_1": float((norm_maps == 1).mean()),
    }
    print(f"[B3] 归一化后图像分数: {tie['raw_scores_unique']} 个唯一值 -> {tie['norm_scores_unique']} 个")
    print(f"     {tie['norm_scores_at_0']*100:.1f}% 的图像分数被压到 0，{tie['norm_scores_at_1']*100:.1f}% 被顶到 1")
    print(f"[B4] 归一化后像素值: {tie['norm_pixels_at_0']*100:.1f}% 被压到 0，{tie['norm_pixels_at_1']*100:.1f}% 被顶到 1")

    # ---- 指标对比 ----
    raw_img = ec.image_metrics(raw_scores, labels)
    norm_img = ec.image_metrics(norm_scores, labels)
    raw_pix = ec.pixel_metrics(list(raw_maps), masks256_nearest)
    norm_pix = ec.pixel_metrics(list(norm_maps), masks256_nearest)

    print("\n[指标对比] 同一模型 / 同一 117 张测试图 / 同一掩码（256 nearest）")
    print(f"  Image AUROC : raw={raw_img['image_auroc']:.4f}  postprocessed={norm_img['image_auroc']:.4f}")
    print(f"  Image AP    : raw={raw_img['image_ap']:.4f}  postprocessed={norm_img['image_ap']:.4f}")
    print(f"  Pixel AUROC : raw={raw_pix['pixel_auroc']:.4f}  postprocessed={norm_pix['pixel_auroc']:.4f}")
    print(f"  Pixel AP    : raw={raw_pix['pixel_ap']:.4f}  postprocessed={norm_pix['pixel_ap']:.4f}")

    # ---- 单调性检验：归一化是否保持顺序 ----
    order_raw = np.argsort(raw_scores)
    order_norm = np.argsort(norm_scores)
    same_order = float((order_raw == order_norm).mean())

    report["steps"]["B_normalization_effect"] = {
        "formula": "normalized = clip((x - threshold) / (max - min) + 0.5, 0, 1)",
        "formula_reproduction_error_image": score_formula_diff,
        "formula_reproduction_error_pixel": map_formula_diff,
        "tie_stats": tie,
        "frac_same_sort_order": same_order,
        "metrics_raw": {**raw_img, **raw_pix},
        "metrics_postprocessed": {**norm_img, **norm_pix},
        "delta_image_auroc": norm_img["image_auroc"] - raw_img["image_auroc"],
        "delta_pixel_auroc": norm_pix["pixel_auroc"] - raw_pix["pixel_auroc"],
    }
    print(f"     归一化后排序与原顺序一致的比例: {same_order*100:.1f}%")

    # =================================================================
    print()
    print("=" * 78)
    print("C. 缺失归一化对指标的实际影响（Day 1 事故量化）")
    print("=" * 78)
    from torchvision.transforms import v2 as T2

    resize_only = T2.Resize([ec.MODEL_SIZE, ec.MODEL_SIZE], interpolation=ec.InterpolationMode.BILINEAR, antialias=True)
    tensors_unnorm = []
    for row_i in range(len(samples)):
        u8 = ec.load_image_u8(samples.iloc[row_i]["image_path"])
        tensors_unnorm.append(resize_only(u8.to(torch.float32).div(255.0)))

    scores_unnorm, maps_unnorm = ec.infer_tensors(model, tensors_unnorm)
    img_unnorm = ec.image_metrics(scores_unnorm, labels)
    pix_unnorm = ec.pixel_metrics(list(maps_unnorm), masks256_nearest)

    print("同一模型、同一 117 张图、同一掩码（256 nearest），只改模型输入是否做 ImageNet 归一化：")
    print(f"  {'输入方式':26s} {'Image AUROC':>12s} {'Pixel AUROC':>12s} {'Pixel AP':>10s}")
    print(f"  {'未归一化 [0,1] (Day 1 实际)':26s} {img_unnorm['image_auroc']:12.4f} {pix_unnorm['pixel_auroc']:12.4f} {pix_unnorm['pixel_ap']:10.4f}")
    print(f"  {'已归一化 (正确做法)':26s} {raw_img['image_auroc']:12.4f} {raw_pix['pixel_auroc']:12.4f} {raw_pix['pixel_ap']:10.4f}")
    print(f"  → Image AUROC 提升 {raw_img['image_auroc']-img_unnorm['image_auroc']:+.4f}, "
          f"Pixel AUROC 提升 {raw_pix['pixel_auroc']-pix_unnorm['pixel_auroc']:+.4f}")

    report["steps"]["C_missing_normalization"] = {
        "metrics_unnormalized_input": {**img_unnorm, **pix_unnorm},
        "metrics_normalized_input": {**raw_img, **raw_pix},
        "delta_image_auroc": raw_img["image_auroc"] - img_unnorm["image_auroc"],
        "delta_pixel_auroc": raw_pix["pixel_auroc"] - pix_unnorm["pixel_auroc"],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] 审计结果 -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
