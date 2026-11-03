# -*- coding: utf-8 -*-
"""AD2 评估工具：高效像素级指标 + AUPRO（大分辨率可用）。

与 Day2 的 eval_core.aupro 相比，这里把"每个阈值重扫全图"改成
**对每个区域/每张图的正常像素分数预先排序 + searchsorted**，
使 AUPRO 在 2000×1000 级别图像上也可行（复杂度从 O(T·N) 降到 O(N log N)+O(T·log N)）。

所有指标口径与 Day 2 一致：
- 分数 = 模型原始马氏距离（未截断）
- 像素指标 = 全局池化
- AUPRO = 逐连通域重叠率对 FPR 的积分（FPR 截断到 0.3）
"""
from __future__ import annotations

import numpy as np


def image_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_ap": float(average_precision_score(labels, scores)),
    }


def pixel_metrics(maps: list[np.ndarray], masks: list[np.ndarray]) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.concatenate([m.reshape(-1) for m in masks]).astype(np.uint8)
    s = np.concatenate([a.reshape(-1) for a in maps]).astype(np.float32)
    return {
        "pixel_auroc": float(roc_auc_score(y, s)),
        "pixel_ap": float(average_precision_score(y, s)),
        "n_pixels": int(y.size),
        "n_anomalous_pixels": int(y.sum()),
    }


def pixel_auroc_per_image(maps: list[np.ndarray], masks: list[np.ndarray]) -> float:
    from sklearn.metrics import roc_auc_score

    vals = []
    for a, m in zip(maps, masks):
        y = m.reshape(-1).astype(np.uint8)
        if y.sum() == 0 or y.sum() == y.size:
            continue
        vals.append(roc_auc_score(y, a.reshape(-1)))
    return float(np.mean(vals)) if vals else float("nan")


def aupro_fast(maps: list[np.ndarray], masks: list[np.ndarray], fpr_limit: float = 0.3,
               n_thresholds: int = 100) -> float:
    """逐连通域 PRO 曲线（searchsorted 加速）。"""
    from scipy import ndimage

    # 1) 预排序：每张图的正常像素分数、每个区域的像素分数
    normal_sorted: list[np.ndarray] = []
    region_sorted: list[np.ndarray] = []
    for a, m in zip(maps, masks):
        flat = a.reshape(-1)
        mb = m.reshape(-1)
        normal_sorted.append(np.sort(flat[~mb]))
        if mb.any():
            lab, n = ndimage.label(m)
            labf = lab.reshape(-1)
            for k in range(1, n + 1):
                idx = labf == k
                if idx.any():
                    region_sorted.append(np.sort(flat[idx]))
    if not region_sorted:
        return float("nan")

    total_normal = int(sum(x.size for x in normal_sorted))
    if total_normal == 0:
        return float("nan")

    all_scores = np.concatenate([a.reshape(-1) for a in maps])
    qs = np.unique(np.quantile(all_scores, np.linspace(0.5, 1.0, n_thresholds)))

    pro, fpr = [], []
    for thr in qs:
        # 每个区域到阈值以上像素的比例
        overlaps = [(r.size - np.searchsorted(r, thr, side="left")) / r.size for r in region_sorted]
        fp = sum((n_.size - np.searchsorted(n_, thr, side="left")) for n_ in normal_sorted)
        pro.append(float(np.mean(overlaps)))
        fpr.append(fp / total_normal)

    pro = np.asarray(pro)
    fpr = np.asarray(fpr)
    order = np.argsort(fpr)
    fpr, pro = fpr[order], pro[order]
    keep = fpr <= fpr_limit
    if keep.sum() < 2:
        return float("nan")
    f, p = fpr[keep], pro[keep]
    f = np.concatenate([[0.0], f])
    p = np.concatenate([[p[0]], p])
    if f[-1] < fpr_limit:
        p = np.concatenate([p, [np.interp(fpr_limit, fpr, pro)]])
        f = np.concatenate([f, [fpr_limit]])
    return float(np.trapezoid(p, f) / fpr_limit)


def threshold_metrics(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    pred = scores >= thr
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    return {
        "threshold": float(thr),
        "fpr_normal": fp / max(fp + tn, 1),
        "recall_defect": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
