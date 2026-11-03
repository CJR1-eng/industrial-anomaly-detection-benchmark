# -*- coding: utf-8 -*-
"""Day 4：受控聚合实验 + 极值诊断 + 分组 Bootstrap。

核心原则
--------
**同一配置内，模型/输入/特征分布/热力图完全不变，只改聚合规则。**
因此脚本先把热力图缓存到 npz，之后所有聚合都在缓存上做，物理上不可能
改变像素级指标（并会显式断言缓存哈希一致）。

聚合规则（全部为已有/常规方法，不宣称原创）
--------------------------------------------
max / top1% mean / top5% mean / p99 / global mean(对照)

阈值：每张热力图的规则分数在**验证集**上取 mean+3σ，测试集只用该阈值评估。

用法:
    python src/day4_aggregation.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ad2_eval as ae  # noqa: E402

MET = PROJECT_ROOT / "results" / "metrics"
CACHE = MET / "day4_maps"
RULES = ["max", "top1", "top5", "p99", "gmean"]


# ------------------------------------------------------------------ aggregation
def aggregate(amap: np.ndarray, rule: str) -> float:
    v = amap.reshape(-1).astype(np.float32)
    if rule == "max":
        return float(v.max())
    if rule == "gmean":
        return float(v.mean())
    if rule == "p99":
        return float(np.quantile(v, 0.99))
    if rule in ("top1", "top5"):
        frac = 0.01 if rule == "top1" else 0.05
        k = max(1, int(round(v.size * frac)))
        # 取最大的 k 个的均值
        if k >= v.size:
            return float(v.mean())
        part = np.partition(v, v.size - k)[v.size - k :]
        return float(part.mean())
    msg = f"未知规则 {rule}"
    raise ValueError(msg)


# ------------------------------------------------------------------ build cache
def build_cache(cfg: str, n_features: int = 100, seed: int = 42) -> Path:
    """拟合一次 PaDiM，缓存 val/test 的原始热力图（float16）。"""
    out = CACHE / f"{cfg}.npz"
    if out.exists():
        print(f"[cache] 复用 {out.name}")
        return out

    from run_ad2 import (  # 延迟导入，避免无 GPU 时报错
        build_index, find_category, fit_padim, infer, AD2_ROOT,
    )

    cat = find_category(AD2_ROOT, "sheet_metal")
    idx = build_index(cat)
    print(f"[{cfg}] 拟合 PaDiM ...")
    model, info = fit_padim(idx["train"], cfg, "cuda", seed, n_features=n_features)
    print(f"  {info}")
    val_scores_max, val_maps = infer(model, idx["val"], cfg, "cuda")
    test_scores_max, test_maps = infer(model, idx["test"], cfg, "cuda")

    # 掩码：缩到热力图网格，用于极值/小块缺陷诊断
    from run_ad2 import load_mask, resize_mask

    val_grid = val_maps[0].shape
    masks_grid = []
    for it in idx["test"]:
        if it["mask"]:
            masks_grid.append(resize_mask(load_mask(it["mask"]), val_grid))
        else:
            masks_grid.append(np.zeros(val_grid, dtype=bool))

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        cfg=cfg,
        val_maps=np.asarray(val_maps, dtype=np.float16),
        test_maps=np.asarray(test_maps, dtype=np.float16),
        masks_grid=np.asarray(masks_grid, dtype=np.uint8),
        labels=np.array([it["label"] for it in idx["test"]], dtype=np.uint8),
        paths=np.array([it["path"] for it in idx["test"]]),
        scenes=np.array([it["scene"] for it in idx["test"]]),
        lighting=np.array([it["lighting"] for it in idx["test"]]),
        gaussian_shape=np.array(info["gaussian_mean_shape"]),
    )
    return out


def hash_maps(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()[:16]


# ------------------------------------------------------------------ main analysis
def analyse(cfg_npz: Path) -> dict:
    z = np.load(cfg_npz, allow_pickle=True)
    cfg = str(z["cfg"])
    val_maps = z["val_maps"].astype(np.float32)
    test_maps = z["test_maps"].astype(np.float32)
    labels = z["labels"].astype(int)
    masks = z["masks_grid"].astype(bool)
    paths = [str(p) for p in z["paths"]]
    scenes = [str(s) for s in z["scenes"]]

    h_before = hash_maps(test_maps)

    # 像素指标只算一次；后面所有聚合都必须复用它（不变性检查）
    pix = ae.pixel_metrics(list(test_maps), list(masks))
    aupro = ae.aupro_fast(list(test_maps), list(masks))
    pix_auroc_pi = ae.pixel_auroc_per_image(list(test_maps), list(masks))

    rows = []
    scores_by_rule = {}
    for rule in RULES:
        vs = np.array([aggregate(m, rule) for m in val_maps])
        ts = np.array([aggregate(m, rule) for m in test_maps])
        scores_by_rule[rule] = ts
        thr = float(vs.mean() + 3 * vs.std())
        im = ae.image_metrics(ts, labels)
        th = ae.threshold_metrics(ts, labels, thr)
        rows.append(
            {
                "config": cfg, "rule": rule, "n_val": len(vs), "n_test": len(ts),
                "image_auroc": im["image_auroc"], "image_ap": im["image_ap"],
                "threshold": thr, "val_mean": float(vs.mean()), "val_std": float(vs.std()),
                "fpr_normal": th["fpr_normal"], "recall_defect": th["recall_defect"],
                "precision": th["precision"], "f1": th["f1"],
                "tp": th["tp"], "fp": th["fp"], "fn": th["fn"], "tn": th["tn"],
                # 像素指标：所有规则共用同一份（受控）
                "pixel_auroc": pix["pixel_auroc"], "pixel_ap": pix["pixel_ap"],
                "pixel_aupro": aupro, "pixel_auroc_per_image": pix_auroc_pi,
                "maps_sha256_16": h_before,
                "n_positions": int(np.prod(test_maps.shape[1:])),
            }
        )

    h_after = hash_maps(test_maps)
    if h_before != h_after:
        msg = "热力图在聚合后被修改，实验无效"
        raise RuntimeError(msg)

    return {"cfg": cfg, "rows": rows, "scores": scores_by_rule,
            "labels": labels, "scenes": scenes, "paths": paths,
            "maps": test_maps, "masks": masks, "val_maps": val_maps,
            "hash": h_before}


# ------------------------------------------------------------------ extremes
def extremes(res: dict) -> dict:
    """Max 聚合的极值位置诊断（不使用缺陷掩码做任何决策）。"""
    maps = res["maps"]
    labels = res["labels"]
    norm_idx = np.where(labels == 0)[0]
    H, W = maps.shape[1:]

    # 1) 正常图 argmax 的位置：到最近边界的距离（归一化）
    pos = []
    for i in norm_idx:
        r, c = np.unravel_index(int(np.argmax(maps[i])), (H, W))
        dn = min(r, H - 1 - r) / (H - 1)
        dw = min(c, W - 1 - c) / (W - 1)
        pos.append({"row": r, "col": c, "dist_border_h": dn, "dist_border_w": dw,
                    "border_band_10pct": bool(dn < 0.1 or dw < 0.1)})

    # 2) 受控位置抽样：同一张图，随机取 N 个位置求 max —— 只改位置数，不改热力图
    rng = np.random.default_rng(42)
    n_total = H * W
    ns = [256, 1024, 4096, n_total]
    ns = sorted({min(n, n_total) for n in ns})
    growth = {}
    flat_all = maps.reshape(maps.shape[0], -1)
    perm = rng.permutation(n_total)  # 所有图共用同一位置子集，保证可比
    for n in ns:
        idx = perm[:n]
        m_norm = flat_all[labels == 0][:, idx].max(axis=1)
        m_def = flat_all[labels == 1][:, idx].max(axis=1)
        growth[str(n)] = {
            "normal_mean_max": float(m_norm.mean()),
            "defect_mean_max": float(m_def.mean()),
            "gap": float(m_def.mean() - m_norm.mean()),
        }

    # 3) max vs top1 对"小缺陷"的影响：按 GT 面积分位分成小/大两组
    areas = res["masks"].reshape(len(maps), -1).mean(axis=1)
    def_idx = np.where(labels == 1)[0]
    def_areas = areas[def_idx]
    med = float(np.median(def_areas)) if len(def_areas) else 0.0
    small = def_idx[def_areas <= med]
    large = def_idx[def_areas > med]

    def rule_scores(rule: str, sel) -> np.ndarray:
        return np.array([aggregate(maps[i], rule) for i in sel])

    smallmax = rule_scores("max", small)
    smalltop = rule_scores("top1", small)
    largemax = rule_scores("max", large)
    largetop = rule_scores("top1", large)
    normmax = rule_scores("max", np.where(labels == 0)[0])
    normtop = rule_scores("top1", np.where(labels == 0)[0])

    return {
        "n_positions": int(n_total), "grid": [H, W],
        "normal_argmax_position": pos,
        "frac_normal_argmax_in_border_band": float(np.mean([p["border_band_10pct"] for p in pos])) if pos else None,
        "median_dist_to_border": [
            float(np.median([p["dist_border_h"] for p in pos])),
            float(np.median([p["dist_border_w"] for p in pos])),
        ],
        "max_vs_positions": growth,
        "small_defect_threshold_area": med,
        "n_small": int(len(small)), "n_large": int(len(large)),
        "max_gap_over_small": float(smallmax.mean() - normmax.mean()),
        "top1_gap_over_small": float(smalltop.mean() - normtop.mean()),
        "max_gap_over_large": float(largemax.mean() - normmax.mean()),
        "top1_gap_over_large": float(largetop.mean() - normtop.mean()),
    }


# ------------------------------------------------------------------ bootstrap
def grouped_bootstrap(res: dict, n_boot: int = 2000, seed: int = 42) -> dict:
    """以**原始场景**为重采样单位的分组 Bootstrap，配对比较各规则。"""
    from sklearn.metrics import roc_auc_score

    scores = res["scores"]
    labels = res["labels"]
    scenes = np.array(res["scenes"])
    uniq = np.unique(scenes)

    # 每个场景对应的样本索引
    idx_by_scene = {s: np.where(scenes == s)[0] for s in uniq}
    rng = np.random.default_rng(seed)

    base_auroc = {r: roc_auc_score(labels, scores[r]) for r in RULES}
    diffs = {r: [] for r in RULES if r != "max"}
    skipped = 0

    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_scene[s] for s in pick])
        y = labels[sel]
        if len(np.unique(y)) < 2:
            skipped += 1
            continue
        a_max = roc_auc_score(y, scores["max"][sel])
        for r in diffs:
            diffs[r].append(roc_auc_score(y, scores[r][sel]) - a_max)

    out = {"n_boot": n_boot, "n_skipped": skipped, "n_scenes": int(len(uniq)),
           "point": {r: float(base_auroc[r]) for r in RULES}, "delta_vs_max": {}}
    for r, v in diffs.items():
        v = np.asarray(v)
        out["delta_vs_max"][r] = {
            "point": float(base_auroc[r] - base_auroc["max"]),
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "frac_positive": float((v > 0).mean()),
        }
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    rows_all = []
    extras = {}
    for cfg in ("squash", "aspect"):
        p = build_cache(cfg)
        res = analyse(p)
        rows_all += res["rows"]
        ex = extremes(res)
        bs = grouped_bootstrap(res)
        # 极值位置信息可能很大，只保留汇总
        ex.pop("normal_argmax_position", None)
        extras[cfg] = {"extremes": ex, "bootstrap": bs, "maps_hash": res["hash"]}
        print(f"\n=== {cfg} ===")
        for r in res["rows"]:
            print(f"  {r['rule']:6s} AUROC={r['image_auroc']:.4f} AP={r['image_ap']:.4f} "
                  f"thr={r['threshold']:.2f} FPR={r['fpr_normal']:.3f} Recall={r['recall_defect']:.3f}")
        print("  分组 bootstrap Δ(AUROC) vs max:")
        for r, v in extras[cfg]["bootstrap"]["delta_vs_max"].items():
            print(f"    {r:6s} {v['point']:+.4f}  95%CI[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")

    df = pd.DataFrame(rows_all)
    df.to_csv(MET / "day4_aggregation.csv", index=False, float_format="%.6f")
    (MET / "day4_extremes_bootstrap.json").write_text(
        json.dumps(extras, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] {MET/'day4_aggregation.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
