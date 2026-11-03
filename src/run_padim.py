# -*- coding: utf-8 -*-
"""Day 1 PaDiM pipeline for MVTec AD / tile  --  SUPERSEDED, READ THIS FIRST.

The evaluation path in this file produced the Day 1 numbers that were later shown
to be WRONG: the inference loop iterates over the dataloader directly, which
bypasses anomalib's PreProcessor callback, so the backbone received [0,1] inputs
without ImageNet normalization. On the same 117 test images this costs about
0.10 Image AUROC and 0.07 Pixel AUROC.

The numbers in ``docs/reports/Day1_Report.md`` are kept only as a record of that
error. Do not use them. Use ``src/run_illumination.py`` (which goes through
``eval_core.py``) for correct scored evaluation, and read
``docs/reports/Day2_Errata.md`` and ``docs/evaluation_pitfalls.md`` first.

This script is still useful for: reproducing Day 1, and as a worked example of
the 184/46/117 split (train / validation / test) that avoids fitting a threshold
on the test set.

---
PaDiM @ MVTec AD (tile) —— Day1 链路验证基线。

设计原则
--------
1. 只使用官方 anomalib 实现，不改动 PaDiM 算法本身。
2. 训练（建立正常特征分布）只使用 train/good；测试只用 test/*。
3. 阈值不从测试集选：把 train/good 再切一部分做验证集（val_split_mode=FROM_TRAIN），
   阈值由验证集正常样本的得分分布决定。测试集仅用于最终评估。
4. 指标用 sklearn 直接算（AUROC / AP），避免依赖框架内部聚合口径，
   同时保留 anomalib 自身 test 输出做交叉核对。
5. 记录耗时、峰值显存、实际配置，落到 results/ 下供复现。

用法
----
    python src/run_padim.py --config configs/padim_tile.yaml
    python src/run_padim.py --config configs/padim_tile.yaml --smoke   # 只跑 20 张，验证链路
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- utilities
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass


def load_config(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 相对路径 -> 绝对路径
    for sec in ("data", "output"):
        for k, v in cfg.get(sec, {}).items():
            if isinstance(v, str) and ("/" in v or "\\" in v) and not Path(v).is_absolute():
                cfg[sec][k] = str(PROJECT_ROOT / v)
    return cfg


def gpu_mem_mb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def field(rec, *names, default=None):
    """从 Batch / dict 中按候选名取字段（兼容 anomalib 不同版本的字段命名）。"""
    if isinstance(rec, dict):
        for n in names:
            if n in rec and rec[n] is not None:
                return rec[n]
        return default
    for n in names:
        v = getattr(rec, n, None)
        if v is not None:
            return v
    return default


def to_numpy(x) -> np.ndarray:
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------- metrics
def image_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = {
        "n": int(len(labels)),
        "n_normal": int((labels == 0).sum()),
        "n_abnormal": int((labels == 1).sum()),
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_ap": float(average_precision_score(labels, scores)),
    }
    return out


def pixel_metrics(maps: np.ndarray, masks: np.ndarray) -> dict:
    """maps: (N,H,W) float, masks: (N,H,W) {0,1}."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = masks.reshape(-1).astype(np.uint8)
    s = maps.reshape(-1).astype(np.float64)
    return {
        "pixel_auroc": float(roc_auc_score(y, s)),
        "pixel_ap": float(average_precision_score(y, s)),
        "n_pixels": int(y.size),
        "n_anomalous_pixels": int(y.sum()),
    }


def f1_at(scores, labels, thr) -> dict:
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": float(thr), "f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def aupro(maps: np.ndarray, masks: np.ndarray, fpr_limit: float = 0.3, n_thresholds: int = 200) -> float:
    """AUPRO（per-region overlap），在 FPR ∈ [0, fpr_limit] 上积分后归一化。

    参考 MVTec 官方 evaluate 脚本的 PRO 定义：
    对每个连通域单独算重叠率，再对连通域取平均。
    """
    from scipy import ndimage

    flat_maps = maps.reshape(len(maps), -1)
    flat_masks = masks.reshape(len(masks), -1) > 0
    total_normal = int((~flat_masks).sum())
    if total_normal == 0:
        return float("nan")

    # 逐图取连通域，记录 (图像索引, 该连通域在展平图中的像素索引)
    per_region: list[tuple[int, np.ndarray]] = []
    for img_i, mask in enumerate(masks):
        mask_b = mask > 0
        if mask_b.sum() == 0:
            continue
        lab, n = ndimage.label(mask_b)
        for i in range(1, n + 1):
            idx = np.where((lab == i).reshape(-1))[0]
            if idx.size:
                per_region.append((img_i, idx))
    if not per_region:
        return float("nan")

    qs = np.unique(np.quantile(flat_maps.reshape(-1), np.linspace(0.5, 1.0, n_thresholds)))

    pro_vals, fpr_vals = [], []
    for thr in qs:
        pred = flat_maps >= thr
        overlaps = []
        for img_i, idx in per_region:
            sub = pred[img_i, idx]
            overlaps.append(sub.mean())
        pro_vals.append(float(np.mean(overlaps)))
        fp = int((pred & ~flat_masks).sum())
        fpr_vals.append(fp / total_normal if total_normal else 0.0)

    pro = np.array(pro_vals)
    fpr = np.array(fpr_vals)
    order = np.argsort(fpr)
    fpr, pro = fpr[order], pro[order]

    # 只保留 FPR <= limit 的部分，并在末端线性插值到 limit
    keep = fpr <= fpr_limit
    if keep.sum() < 2:
        return float("nan")
    f, p = fpr[keep], pro[keep]
    # 起点补 (0, pro[0])
    f = np.concatenate([[0.0], f])
    p = np.concatenate([[p[0]], p])
    # 末端补插值
    if f[-1] < fpr_limit:
        p_end = np.interp(fpr_limit, fpr, pro)
        f = np.concatenate([f, [fpr_limit]])
        p = np.concatenate([p, [p_end]])
    return float(np.trapezoid(p, f) / fpr_limit)


# ---------------------------------------------------------------- visualization
def _mpl_setup():
    """统一 matplotlib 后端与中文字体（否则中文标题渲染成方块）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def save_case_figure(
    img: np.ndarray,
    gt_mask: np.ndarray | None,
    amap: np.ndarray,
    score: float,
    thr: float,
    title: str,
    out_path: Path,
    pred_mask: np.ndarray | None = None,
) -> None:
    plt = _mpl_setup()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(img)
    axes[0].set_title("Input image")
    axes[1].imshow(img)
    im = axes[1].imshow(amap, cmap="jet", alpha=0.45, vmin=0, vmax=max(amap.max(), 1e-6))
    axes[1].set_title(f"Anomaly heatmap\nscore={score:.2f} (thr={thr:.2f})")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(amap, cmap="jet")
    axes[2].set_title("Anomaly map (raw Mahalanobis)")
    if gt_mask is not None and gt_mask.sum() > 0:
        axes[3].imshow(gt_mask, cmap="gray")
        axes[3].set_title("GT mask")
    elif pred_mask is not None:
        axes[3].imshow(pred_mask, cmap="gray")
        axes[3].set_title("Pred mask (no GT: normal)")
    for a in axes:
        a.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main flow
def build(cfg: dict):
    import torch
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models.image.padim import Padim

    d = cfg["data"]
    m = cfg["model"]
    e = cfg["engine"]

    # val_split_mode 用字符串传入，anomalib 内部会做枚举转换
    dm_kwargs = dict(
        root=d["root"],
        category=d["category"],
        train_batch_size=d["train_batch_size"],
        eval_batch_size=d["eval_batch_size"],
        num_workers=d["num_workers"],
        val_split_mode=d["val_split_mode"],
        val_split_ratio=d["val_split_ratio"],
        test_split_mode=d["test_split_mode"],
        seed=cfg["experiment"]["seed"],
    )
    datamodule = MVTecAD(**dm_kwargs)

    model = Padim(
        backbone=m["backbone"],
        layers=list(m["layers"]),
        pre_trained=m["pre_trained"],
        n_features=m["n_features"],
    )

    runs_dir = Path(cfg["output"]["runs_dir"]) / cfg["experiment"]["name"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    engine = Engine(
        default_root_dir=str(runs_dir),
        max_epochs=int(e["max_epochs"]),
        accelerator=e["accelerator"] if torch.cuda.is_available() else "cpu",
        devices=int(e["devices"]),
    )
    return datamodule, model, engine, dm_kwargs


def run(cfg: dict, smoke: bool = False) -> dict:
    import torch

    # 环境变量必须在导入 anomalib / timm 之前生效（骨干权重下载端点等）
    for k, v in (cfg.get("env") or {}).items():
        os.environ.setdefault(str(k), str(v))

    set_seed(cfg["experiment"]["seed"])
    report: dict = {
        "experiment": cfg["experiment"]["name"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "smoke_test": smoke,
        "status": "unknown",
        "errors": [],
        "warnings": [],
        "timings_s": {},
        "memory_MB": {},
    }

    if torch.cuda.is_available():
        report["device"] = {
            "type": "cuda",
            "name": torch.cuda.get_device_name(0),
            "capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        torch.cuda.reset_peak_memory_stats()
    else:
        report["device"] = {"type": "cpu", "torch": torch.__version__}
        report["warnings"].append("CUDA 不可用，本次在 CPU 上运行")

    datamodule, model, engine, dm_kwargs = build(cfg)
    report["data_config"] = {k: str(v) for k, v in dm_kwargs.items()}
    report["model_config"] = {k: str(v) for k, v in cfg["model"].items()}

    # ---- 数据划分核查 ----
    datamodule.setup()
    counts = {
        "train": len(datamodule.train_data) if hasattr(datamodule, "train_data") else None,
        "val": len(datamodule.val_data) if hasattr(datamodule, "val_data") else None,
        "test": len(datamodule.test_data) if hasattr(datamodule, "test_data") else None,
    }
    report["split_counts"] = counts

    # ---- 训练（PaDiM 拟合高斯）----
    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    report["timings_s"]["fit"] = round(time.time() - t0, 2)
    report["memory_MB"]["after_fit_peak"] = round(gpu_mem_mb(), 1)

    # ---- 验证集推理（用于定阈值，不参与训练）----
    # 不走 engine.predict：anomalib 2.x 在 predict 之后会施加 PostProcessor 的
    # 归一化（anomaly_map 被压到 [0,1]，且逐图重标定），会破坏跨图可比性。
    # PaDiM 的论文口径是直接用马氏距离，故这里取模型原始输出。
    t0 = time.time()
    val_scores, val_maps = infer_raw(model, datamodule.val_dataloader())
    report["timings_s"]["predict_val"] = round(time.time() - t0, 2)
    report["val"] = {"n": int(len(val_scores))}

    # ---- 测试集推理 ----
    t0 = time.time()
    test_paths = datamodule.test_data.samples["image_path"].tolist()
    imgs, masks, scores, maps = infer_raw(model, datamodule.test_dataloader(), return_images=True)
    report["timings_s"]["predict_test"] = round(time.time() - t0, 2)
    report["memory_MB"]["peak_test"] = round(gpu_mem_mb(), 1)

    labels = np.array([1 if (m.sum() > 0) else 0 for m in masks])
    paths = [str(p) for p in test_paths]
    report["test"] = {
        "n": int(len(scores)),
        "n_normal": int((labels == 0).sum()),
        "n_abnormal": int((labels == 1).sum()),
    }

    if smoke:
        report["status"] = "smoke_ok"
        return report

    # ---- 阈值：来自验证集正常样本 ----
    thr_img = float(np.mean(val_scores) + 3 * np.std(val_scores)) if len(val_scores) else float("nan")
    all_val_pix = np.concatenate([m.reshape(-1) for m in val_maps]) if val_maps else np.array([0.0])
    thr_pix = float(np.quantile(all_val_pix, 0.995))
    report["thresholds"] = {
        "image_mean_plus_3std_from_val": thr_img,
        "pixel_p99.5_from_val": thr_pix,
        "val_scores_mean": float(np.mean(val_scores)) if len(val_scores) else None,
        "val_scores_std": float(np.std(val_scores)) if len(val_scores) else None,
    }

    # ---- 指标 ----
    report["metrics"] = {}
    report["metrics"].update(image_metrics(scores, labels))
    H, W = maps[0].shape
    report["metrics"].update(pixel_metrics(maps, masks))
    report["metrics"]["metrics_resolution"] = f"{H}x{W}"
    report["metrics"]["pixel_auroc_per_image_mean"] = per_image_pixel_auroc(maps, masks)
    report["metrics"]["f1_at_val_threshold"] = f1_at(scores, labels, thr_img)

    # 空间对齐诊断：区分"定位错位"与"特征无区分力"
    report["alignment"] = alignment_check(maps, masks)

    # 仅作参考：测试集最优 F1（乐观口径，不可作为论文结论）
    best = None
    for t in np.quantile(scores, np.linspace(0.01, 0.99, 200)):
        r = f1_at(scores, labels, float(t))
        if best is None or r["f1"] > best["f1"]:
            best = r
    report["metrics"]["f1_at_test_optimal_threshold_OPTIMISTIC"] = best

    if len(maps) <= 400:  # AUPRO 计算较贵，样本很多时按需关闭
        try:
            report["metrics"]["pixel_aupro"] = aupro(maps, masks)
        except Exception as exc:  # noqa: BLE001
            report["warnings"].append(f"AUPRO 计算失败: {exc}")

    # ---- anomalib 自带 evaluator 结果做交叉核对 ----
    try:
        t0 = time.time()
        alb_metrics = engine.test(model=model, datamodule=datamodule)
        report["timings_s"]["anomalib_test"] = round(time.time() - t0, 2)
        report["anomalib_official_pipeline_metrics"] = alb_metrics
    except Exception as exc:  # noqa: BLE001
        report["warnings"].append(f"anomalib engine.test 失败（不影响自有指标）: {exc}")

    # ---- 落盘：逐图得分 ----
    metrics_dir = Path(cfg["output"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / f"{cfg['experiment']['name']}_per_image.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("image_path,gt_label,pred_score,pred_label_at_val_thr,anomaly_map_max,anomaly_map_mean\n")
        for p, l, s, m in zip(paths, labels, scores, maps):
            f.write(f"{p},{int(l)},{s:.6f},{int(s >= thr_img)},{m.max():.6f},{m.mean():.6f}\n")
    report["per_image_csv"] = str(csv_path)

    # ---- 可视化案例 ----
    figs = save_cases(cfg, imgs, masks, maps, scores, labels, paths, thr_img, thr_pix)
    report["figures"] = figs

    # ---- 保存重分布（便于后续做光照实验复现）----
    npz_path = metrics_dir / f"{cfg['experiment']['name']}_test_maps.npz"
    np.savez_compressed(
        npz_path,
        scores=scores.astype(np.float32),
        labels=labels.astype(np.uint8),
        maps=maps.astype(np.float16),
        masks=masks.astype(np.uint8),
        paths=np.array(paths),
    )
    report["maps_npz"] = str(npz_path)

    report["status"] = "success"
    return report


def infer_raw(model, loader, device: str = "cuda", return_images: bool = False):
    """用模型原始输出做推理，绕过 PostProcessor 的归一化。

    anomalib 2.x 的 predict_step 是 `self.model(batch.image)`，之后由回调
    施加 PostProcessor（逐图 min-max 归一化）。归一化会抹掉跨图可比性，
    使全局 pixel AUROC 失去意义，因此这里直接取 PadimModel 的原始
    马氏距离图与得分。

    返回: (scores, maps) 或 (images, masks, scores, maps)
    """
    import torch

    model.eval()
    model = model.to(device)
    scores, maps, masks, imgs = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch.image.to(device)
            out = model.model(x)
            s = out.pred_score.detach().float().cpu().numpy().reshape(-1)
            m = out.anomaly_map.detach().float().cpu().numpy()
            if m.ndim == 4:
                m = m[:, 0]
            gm = batch.gt_mask
            gm = gm.detach().cpu().numpy() if gm is not None else np.zeros_like(m)
            if gm.ndim == 4:
                gm = gm[:, 0]
            for i in range(len(s)):
                scores.append(float(s[i]))
                maps.append(m[i])
                if return_images:
                    im = batch.image[i].detach().float().cpu().numpy()
                    im = np.transpose(im, (1, 2, 0)) if im.ndim == 3 else im
                    imgs.append(np.clip(im, 0, 1))
                    masks.append((gm[i] > 0.5).astype(np.uint8))
    if return_images:
        return np.array(imgs), np.array(masks), np.array(scores), np.array(maps)
    return np.array(scores), maps


def alignment_check(maps, masks, max_shift: int = 32, step: int = 8) -> dict:
    """诊断热力图与 GT 的空间是否对齐。

    做法：把图和掩码降采样到 64x64，在所有整数平移上计算与 GT 的相关系数，
    找最佳平移。若最佳平移显著偏离 (0,0)，说明存在系统性错位，
    pixel 级指标会被严重低估。
    """
    from scipy import ndimage

    def down(x, size=64):
        return ndimage.zoom(x.astype(np.float32), size / x.shape[0], order=1)

    shifts, corrs0, corrs_best = [], [], []
    for m, g in zip(maps, masks):
        if g.sum() == 0:
            continue
        md, gd = down(m), down(g)
        md = (md - md.mean()) / (md.std() + 1e-8)
        gd = gd - gd.mean()
        denom = (gd.std() + 1e-8)
        best = (-2.0, 0, 0)
        c0 = None
        for dy in range(-max_shift // 4, max_shift // 4 + 1, step // 4):
            for dx in range(-max_shift // 4, max_shift // 4 + 1, step // 4):
                shifted = ndimage.shift(md, (dy, dx), order=1, mode="nearest")
                c = float((shifted * gd).mean() / denom)
                if (dy, dx) == (0, 0):
                    c0 = c
                if c > best[0]:
                    best = (c, dy, dx)
        corrs0.append(c0 if c0 is not None else 0.0)
        corrs_best.append(best[0])
        shifts.append((best[1] * 4, best[2] * 4))  # 换算回 256 像素尺度

    if not shifts:
        return {"n": 0}
    arr = np.array(shifts)
    return {
        "n": len(shifts),
        "median_best_shift_px": [float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))],
        "mean_corr_at_zero_shift": float(np.mean(corrs0)),
        "mean_corr_at_best_shift": float(np.mean(corrs_best)),
        "frac_best_shift_is_zero": float(np.mean((arr[:, 0] == 0) & (arr[:, 1] == 0))),
    }


def per_image_pixel_auroc(maps, masks) -> float:
    """逐图 pixel AUROC 的平均值（只统计有缺陷的图）。

    与全局池化口径的区别：逐图口径不受"不同图像距离尺度不同"影响，
    用于判断"图内是否定位正确"。
    """
    from sklearn.metrics import roc_auc_score

    vals = []
    for m, g in zip(maps, masks):
        y = (g.reshape(-1) > 0).astype(np.uint8)
        if y.sum() == 0 or y.sum() == y.size:
            continue
        vals.append(roc_auc_score(y, m.reshape(-1)))
    return float(np.mean(vals)) if vals else float("nan")


def save_cases(cfg, imgs, masks, maps, scores, labels, paths, thr_img, thr_pix) -> list[str]:
    import re

    viz = cfg["visualization"]
    fig_dir = Path(cfg["output"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []

    def norm_name(p: str) -> str:
        s = Path(p).stem
        parent = Path(p).parent.name
        return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{parent}__{s}")

    def iou(pred, gt):
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        return float(inter / union) if union else 0.0

    # 1) 正常样本中得分最高的 -> 误报
    normal_idx = np.where(labels == 0)[0]
    fp_order = normal_idx[np.argsort(-scores[normal_idx])][: viz["num_normal_fp"]]
    for rank, i in enumerate(fp_order, 1):
        name = f"fig_{rank:02d}_normal_false_positive__{norm_name(paths[i])}.png"
        save_case_figure(
            imgs[i], None, maps[i], scores[i], thr_img,
            f"[NORMAL / false positive candidate] score={scores[i]:.2f}",
            fig_dir / name,
            pred_mask=(maps[i] >= thr_pix).astype(np.uint8),
        )
        out.append(str(fig_dir / name))

    # 2) 异常样本中：得分高且 IoU 高 -> 正确定位
    ab_idx = np.where(labels == 1)[0]
    scored = []
    for i in ab_idx:
        pm = maps[i] >= thr_pix
        scored.append((iou(pm, masks[i] > 0), scores[i], i))
    scored.sort(key=lambda x: (-(x[0] > 0.1), -x[1]))
    hits = [i for iou_v, s, i in scored if iou_v > 0.1][: viz["num_defect_hit"]]
    for rank, i in enumerate(hits, 1):
        iou_v = iou(maps[i] >= thr_pix, masks[i] > 0)
        name = f"fig_{rank+3:02d}_defect_localized__{norm_name(paths[i])}.png"
        save_case_figure(
            imgs[i], masks[i], maps[i], scores[i], thr_img,
            f"[DEFECT / correctly localized] IoU={iou_v:.2f}, score={scores[i]:.2f}",
            fig_dir / name,
        )
        out.append(str(fig_dir / name))

    # 3) 异常样本中得分最低 -> 漏检
    miss_order = ab_idx[np.argsort(scores[ab_idx])][: viz["num_defect_miss"]]
    for rank, i in enumerate(miss_order, 1):
        name = f"fig_{rank+6:02d}_defect_missed__{norm_name(paths[i])}.png"
        save_case_figure(
            imgs[i], masks[i], maps[i], scores[i], thr_img,
            f"[DEFECT / missed] score={scores[i]:.2f} < thr={thr_img:.2f}",
            fig_dir / name,
        )
        out.append(str(fig_dir / name))

    # 4) 总览图
    try:
        plt = _mpl_setup()

        n = min(len(out), 9)
        fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6))
        axes = np.atleast_1d(axes)
        chosen = list(fp_order) + list(hits) + list(miss_order[: max(0, n - len(fp_order) - len(hits))])
        for a, i in zip(axes, chosen):
            a.imshow(imgs[i])
            a.imshow(maps[i], cmap="jet", alpha=0.45)
            a.axis("off")
            a.set_title(f"{'N' if labels[i]==0 else 'A'}\n{scores[i]:.2f}", fontsize=8)
        fig.suptitle("PaDiM @ tile: overview (N=normal, A=anomaly)", fontsize=10)
        fig.tight_layout()
        overview = fig_dir / "fig_00_overview_grid.png"
        fig.savefig(overview, dpi=140, bbox_inches="tight")
        plt.close(fig)
        out.insert(0, str(overview))
    except Exception:  # noqa: BLE001
        pass

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "padim_tile.yaml"))
    ap.add_argument("--smoke", action="store_true", help="只跑极小规模验证链路")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.smoke:
        cfg["data"]["train_batch_size"] = 4
        cfg["data"]["eval_batch_size"] = 4

    try:
        report = run(cfg, smoke=args.smoke)
    except Exception as exc:  # noqa: BLE001
        report = {
            "status": "failed",
            "experiment": cfg["experiment"]["name"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        out_dir = PROJECT_ROOT / "results" / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"padim_run_failed_{datetime.now():%Y%m%d_%H%M%S}.json"
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[FAILED] {exc}\n详见 {p}")
        print(report["traceback"])
        return 1

    out_dir = PROJECT_ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{cfg['experiment']['name']}_summary.json"
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k in
                      ("status", "timings_s", "memory_MB", "metrics", "split_counts", "thresholds")},
                     indent=2, ensure_ascii=False, default=str))
    print(f"\n[OK] summary -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
