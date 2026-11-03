# -*- coding: utf-8 -*-
"""任务 3：光照失效分析 + 可视化。

产出
----
figures/day2_illumination/
    fig01_condition_metrics.png        各条件指标对比（柱状 + 差值）
    fig02_normal_score_shift.png       33 张正常图的分数位移（配对斜率图 + 箱线）
    fig03_defect_score_shift.png       84 张缺陷图的分数位移
    fig04_delta_hist.png               分数变化量分布（正常 vs 缺陷）
    fig05_heatmap_normal_*.png         3 组正常图 × 6 条件热力图（统一色标）
    fig06_heatmap_defect_*.png         3 组缺陷图 × 6 条件热力图（统一色标）
    fig07_flip_summary.png             误报/漏检翻转计数

用法:
    python src/day2_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import perturbations as pt  # noqa: E402

MET_DIR = PROJECT_ROOT / "results" / "metrics"
FIG_DIR = PROJECT_ROOT / "figures" / "day2_illumination"
COND_ORDER = pt.CONDITION_NAMES
SHADOW_LABEL = {
    "original": "Original",
    "brightness_low": "Brightness 0.8",
    "brightness_high": "Brightness 1.2",
    "gamma_low": "Gamma 0.7",
    "gamma_high": "Gamma 1.4",
    "local_shadow": "Local shadow",
}


def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def load(tag: str = ""):
    met = pd.read_csv(MET_DIR / f"day2_illumination_results{tag}.csv")
    per = pd.read_csv(MET_DIR / f"day2_per_image{tag}.csv")
    npz = np.load(MET_DIR / f"day2_maps{tag}.npz", allow_pickle=True)
    return met, per, npz


# --------------------------------------------------------------- fig01
def fig_condition_metrics(met: pd.DataFrame, thr: dict) -> Path:
    plt = mpl()
    conds = [c for c in COND_ORDER if c in set(met["condition"])]
    m = met.set_index("condition").loc[conds]
    x = np.arange(len(conds))
    w = 0.2

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    panels = [
        ("image_auroc", "Image AUROC", 0.55),
        ("pixel_auroc", "Pixel AUROC @256", 0.55),
        ("pixel_auroc_840", "Pixel AUROC @840", 0.55),
    ]
    colors = ["#2b6cb0", "#dd6b20", "#2f855a"]
    for ax, (col, title, ymin) in zip(axes, panels):
        vals = m[col].values
        bars = ax.bar(x, vals, color=colors[list(panels).index((col, title, ymin))], width=0.62)
        base = m.loc["original", col] if "original" in m.index else vals[0]
        ax.axhline(base, color="gray", ls="--", lw=1, label=f"Original={base:.4f}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([SHADOW_LABEL[c] for c in conds], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(ymin, 1.0)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("PaDiM @ MVTec tile — metrics under synthetic illumination perturbations "
                 "(frozen model, same 117 test images)", fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "fig01_condition_metrics.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------- fig02/03
def fig_score_shift(per: pd.DataFrame, thr_val: float, kind: str) -> Path:
    plt = mpl()
    label = 0 if kind == "normal" else 1
    sub = per[per["gt_label"] == label]
    conds = [c for c in COND_ORDER if c in set(sub["condition"])]
    piv = sub.pivot_table(index="image_path", columns="condition", values="score_raw")[conds]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.35, 1]})

    xs = np.arange(len(conds))
    for _, row in piv.iterrows():
        axes[0].plot(xs, row.values, color="#4a5568", alpha=0.35, lw=0.9)
    axes[0].plot(xs, piv.mean(axis=0).values, color="#c53030", lw=2.4, marker="o", label="mean")
    axes[0].plot(xs, piv.median(axis=0).values, color="#2b6cb0", lw=2.0, marker="s", ls="--", label="median")
    axes[0].axhline(thr_val, color="green", ls=":", lw=1.6, label=f"threshold={thr_val:.1f}")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([SHADOW_LABEL[c] for c in conds], rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("raw anomaly score (max of Mahalanobis map)")
    axes[0].set_title(f"{'Normal' if kind=='normal' else 'Defect'} images (n={len(piv)}): paired score trajectories")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    data = [piv[c].values for c in conds]
    bp = axes[1].boxplot(data, tick_labels=[SHADOW_LABEL[c] for c in conds], patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#bee3f8")
    axes[1].axhline(thr_val, color="green", ls=":", lw=1.6)
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)
    axes[1].set_title("Score distribution per condition")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    p = FIG_DIR / f"fig0{2 if kind=='normal' else 3}_{kind}_score_shift.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------- fig04
def fig_delta_hist(per: pd.DataFrame) -> Path:
    plt = mpl()
    base = per[per["condition"] == "original"].set_index("image_path")["score_raw"]
    conds = [c for c in COND_ORDER if c != "original"]
    fig, axes = plt.subplots(1, len(conds), figsize=(3.4 * len(conds), 3.6), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), conds):
        cur = per[per["condition"] == c].set_index("image_path")["score_raw"]
        lab = per[per["condition"] == c].set_index("image_path")["gt_label"]
        d = (cur - base).dropna()
        d_n = d[lab.reindex(d.index) == 0]
        d_a = d[lab.reindex(d.index) == 1]
        bins = np.linspace(np.percentile(np.concatenate([d_n, d_a]), 0.5),
                           np.percentile(np.concatenate([d_n, d_a]), 99.5), 34)
        ax.hist(d_n, bins=bins, alpha=0.65, color="#2b6cb0", label="normal")
        ax.hist(d_a, bins=bins, alpha=0.65, color="#c53030", label="defect")
        ax.axvline(0, color="k", lw=1)
        ax.set_title(SHADOW_LABEL[c], fontsize=9)
        ax.set_xlabel("Δ score vs Original")
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("# images")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Per-image score change induced by illumination perturbation", fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "fig04_delta_hist.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------- fig05/06
def fig_heatmaps(npz, per: pd.DataFrame, thr_val: float, kind: str, n_cases: int = 3) -> list[Path]:
    """挑案例并画统一色标的多条件热力图。"""
    plt = mpl()
    conds = list(npz["conditions"])
    maps = npz["maps"]
    scores = npz["scores"]
    labels = npz["labels"]
    paths = npz["paths"]

    kinds = ["original"] + [c for c in COND_ORDER if c != "original" and c in conds]
    idx_orig = conds.index("original")

    label_want = 0 if kind == "normal" else 1
    cand = np.where(labels == label_want)[0]

    # 选择标准（预先声明，避免事后挑图）
    base_score = scores[idx_orig]
    if kind == "normal":
        # 选"扰动后分数上升最多"的正常图 —— 即最容易被光照骗到的样本
        delta = np.max([scores[conds.index(c)] for c in kinds[1:]], axis=0) - base_score
        order = cand[np.argsort(-delta[cand])][:n_cases]
    else:
        # 选"扰动后分数下降最多"的缺陷图 —— 即最容易被漏检的样本
        delta = np.min([scores[conds.index(c)] for c in kinds[1:]], axis=0) - base_score
        order = cand[np.argsort(delta[cand])][:n_cases]

    out = []
    for rank, img_i in enumerate(order, 1):
        fig, axes = plt.subplots(2, len(kinds), figsize=(2.5 * len(kinds), 5.4))
        # 统一色标：该图在所有条件下的最大值
        vmax = max(float(maps[conds.index(c), img_i].max()) for c in kinds)
        for j, c in enumerate(kinds):
            ci = conds.index(c)
            amap = maps[ci, img_i].astype(np.float32)
            ax = axes[0, j]
            im = ax.imshow(amap, cmap="jet", vmin=0, vmax=vmax)
            sc = float(scores[ci, img_i])
            flag = "ABN" if sc >= thr_val else "OK "
            ax.set_title(f"{SHADOW_LABEL[c]}\nscore={sc:.1f} [{flag}]", fontsize=8)
            ax.axis("off")
            axes[1, j].hist(amap.reshape(-1), bins=60, color="#4a5568")
            axes[1, j].set_xlim(0, vmax)
            axes[1, j].tick_params(labelsize=6)
            axes[1, j].grid(alpha=0.2)
        fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01, label="Mahalanobis distance")
        fname = Path(paths[img_i]).parent.name + "__" + Path(paths[img_i]).stem
        fig.suptitle(
            f"{'NORMAL' if kind=='normal' else 'DEFECT'} case {rank}: {fname}   "
            f"(shared colour scale vmax={vmax:.1f}, threshold={thr_val:.1f})",
            fontsize=10,
        )
        p = FIG_DIR / f"fig0{5 if kind=='normal' else 6}_heatmap_{kind}_{rank}_{fname}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        out.append(p)
    return out


# --------------------------------------------------------------- fig07
def fig_flip_summary(per: pd.DataFrame) -> tuple[Path, dict]:
    plt = mpl()
    conds = [c for c in COND_ORDER if c in set(per["condition"])]
    base = per[per["condition"] == "original"].set_index("image_path")
    rows = []
    for c in conds:
        cur = per[per["condition"] == c].set_index("image_path")
        normal = base["gt_label"] == 0
        defect = base["gt_label"] == 1
        fp_new = int(((base["pred_at_val_thr"] == 0) & (cur["pred_at_val_thr"] == 1) & normal).sum())
        fp_removed = int(((base["pred_at_val_thr"] == 1) & (cur["pred_at_val_thr"] == 0) & normal).sum())
        fn_new = int(((base["pred_at_val_thr"] == 1) & (cur["pred_at_val_thr"] == 0) & defect).sum())
        fn_fixed = int(((base["pred_at_val_thr"] == 0) & (cur["pred_at_val_thr"] == 1) & defect).sum())
        rows.append(
            {"condition": c, "new_false_positive": fp_new, "false_positive_fixed": fp_removed,
             "new_missed_defect": fn_new, "defect_recovered": fn_fixed}
        )
    df = pd.DataFrame(rows).set_index("condition").loc[conds]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(conds))
    w = 0.2
    metrics = ["new_false_positive", "false_positive_fixed", "new_missed_defect", "defect_recovered"]
    colors = ["#c53030", "#2f855a", "#dd6b20", "#2b6cb0"]
    labels_cn = ["新增误报", "误报消失", "新增漏检", "漏检被救回"]
    for k, (col, col_c, nm) in enumerate(zip(metrics, colors, labels_cn)):
        ax.bar(x + (k - 1.5) * w, df[col].values, width=w, color=col_c, label=nm)
    ax.set_xticks(x)
    ax.set_xticklabels([SHADOW_LABEL[c] for c in conds], rotation=30, ha="right")
    ax.set_ylabel("# images (vs Original, fixed val threshold)")
    ax.set_title("Label flips relative to Original (normal n=33, defect n=84)")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = FIG_DIR / "fig07_flip_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p, df.to_dict(orient="index")


# --------------------------------------------------------------- main
def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    met, per, npz = load()

    run_cfg = json.loads((PROJECT_ROOT / "results" / "logs" / "day2_run.json").read_text(encoding="utf-8"))
    thr_val = run_cfg["thresholds"]["image_threshold_mean_plus_3std"]

    paths = []
    paths.append(fig_condition_metrics(met, run_cfg["thresholds"]))
    paths.append(fig_score_shift(per, thr_val, "normal"))
    paths.append(fig_score_shift(per, thr_val, "defect"))
    paths.append(fig_delta_hist(per))
    paths += fig_heatmaps(npz, per, thr_val, "normal", 3)
    paths += fig_heatmaps(npz, per, thr_val, "defect", 3)
    p7, flips = fig_flip_summary(per)
    paths.append(p7)

    # ---- 文字分析 ----
    analysis: dict = {"threshold": thr_val, "flips": flips, "conditions": {}}
    base = per[per["condition"] == "original"].set_index("image_path")

    for c in [x for x in COND_ORDER if x in set(per["condition"])]:
        cur = per[per["condition"] == c].set_index("image_path")
        d = (cur["score_raw"] - base["score_raw"]).dropna()
        lab = base["gt_label"].reindex(d.index)
        analysis["conditions"][c] = {
            "delta_mean_normal": float(d[lab == 0].mean()),
            "delta_median_normal": float(d[lab == 0].median()),
            "delta_mean_defect": float(d[lab == 1].mean()),
            "delta_median_defect": float(d[lab == 1].median()),
            "n_normal_up": int((d[lab == 0] > 0).sum()),
            "n_normal_down": int((d[lab == 0] < 0).sum()),
            "n_defect_up": int((d[lab == 1] > 0).sum()),
            "n_defect_down": int((d[lab == 1] < 0).sum()),
            "mean_gt_region_score_shift": float(
                (cur["anomaly_map_mean_in_gt"] - base["anomaly_map_mean_in_gt"]).dropna().mean()
            ),
        }

    (MET_DIR / "day2_analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== 各条件分数位移（原图 → 条件）===")
    print(f"{'条件':16s} {'正常Δ均值':>10s} {'缺陷Δ均值':>10s} {'缺陷区响应Δ':>12s}")
    for c, v in analysis["conditions"].items():
        print(f"{c:16s} {v['delta_mean_normal']:+10.2f} {v['delta_mean_defect']:+10.2f} "
              f"{v['mean_gt_region_score_shift']:+12.3f}")
    print("\n=== 标签翻转（相对 Original）===")
    print(pd.DataFrame(flips).T.to_string())
    print(f"\n[OK] 生成 {len(paths)} 张图 -> {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
