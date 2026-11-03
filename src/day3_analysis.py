# -*- coding: utf-8 -*-
"""AD2 结果分析：误报/漏检案例、缺陷面积分组、热力图。

用法:
    python src/day3_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MET = PROJECT_ROOT / "results" / "metrics"
FIG = PROJECT_ROOT / "figures" / "day3_real_illumination"


def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def load_any():
    """优先读 aspect，缺则读 squash。"""
    out = {}
    for tag, name in (("", "main"), ("_aspect", "aspect")):
        p = MET / f"day3_ad2_per_image{tag}.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
    return out


def fig_score_dist(per: pd.DataFrame, mode: str = "") -> Path:
    plt = mpl()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for lab, color, nm in ((0, "#2b6cb0", "normal"), (1, "#c53030", "defect")):
        v = per[per["gt_label"] == lab]["score_raw"].values
        ax.hist(v, bins=40, alpha=0.65, color=color, label=f"{nm} (n={len(v)})")
    ax.set_xlabel("raw anomaly score (max Mahalanobis)")
    ax.set_ylabel("# images")
    ax.set_title("AD2 sheet_metal: image score distribution")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = FIG / f"fig04_score_distribution{mode}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_area_vs_recall(per: pd.DataFrame, thr: float, mode: str = "") -> tuple[Path, dict]:
    """缺陷面积 vs 是否被检出（微小缺陷分析）。"""
    plt = mpl()
    d = per[per["gt_label"] == 1].copy()
    d = d[d["gt_area_ratio"] > 0]
    if d.empty:
        return FIG / f"fig05_defect_area_vs_recall{mode}.png", {}

    bins = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    labels = ["<0.01%", "0.01-0.1%", "0.1-1%", "1-10%", ">10%"]
    d["area_bin"] = pd.cut(d["gt_area_ratio"], bins=bins, labels=labels, include_lowest=True)
    grp = d.groupby("area_bin", observed=False).agg(
        n=("score_raw", "size"),
        recall=("pred_at_thr", "mean"),
        mean_score=("score_raw", "mean"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(len(grp)), grp["n"].values, color="#4a5568")
    axes[0].set_xticks(range(len(grp)))
    axes[0].set_xticklabels([str(i) for i in grp.index], rotation=20, fontsize=8)
    axes[0].set_title("Defect count by area ratio")
    axes[0].set_ylabel("# images")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(range(len(grp)), grp["recall"].values, color="#c53030")
    axes[1].set_xticks(range(len(grp)))
    axes[1].set_xticklabels([str(i) for i in grp.index], rotation=20, fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"Recall at fixed threshold ({thr:.2f}) by defect area")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    p = FIG / f"fig05_defect_area_vs_recall{mode}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p, {str(k): {"n": int(v["n"]), "recall": float(v["recall"])} for k, v in grp.iterrows()}


def fig_cases(per: pd.DataFrame, npz_path: Path, thr: float, kind: str, n: int = 3, mode: str = "") -> list[Path]:
    """正常误报 / 缺陷漏检 / 缺陷正确定位 的 4 联图。"""
    plt = mpl()
    z = np.load(npz_path, allow_pickle=True)
    maps = z["maps"]
    scores = z["scores"]
    paths = [str(p) for p in z["paths"]]
    labels = z["labels"]
    stem2idx = {Path(p).stem: i for i, p in enumerate(paths)}

    out = []
    if kind == "normal":
        cand = [i for i in range(len(labels)) if labels[i] == 0]
        order = sorted(cand, key=lambda i: -scores[i])[:n]
        titles = ["[NORMAL / highest score (false-positive risk)]"] * len(order)
    elif kind == "miss":
        cand = [i for i in range(len(labels)) if labels[i] == 1]
        order = sorted(cand, key=lambda i: scores[i])[:n]
        titles = ["[DEFECT / lowest score (missed)]"] * len(order)
    else:
        cand = [i for i in range(len(labels)) if labels[i] == 1]
        order = sorted(cand, key=lambda i: -scores[i])[:n]
        titles = ["[DEFECT / highest score (detected)]"] * len(order)

    from torchvision.io import ImageReadMode, read_image

    for rank, i in enumerate(order, 1):
        img = read_image(paths[i], mode=ImageReadMode.RGB).permute(1, 2, 0).numpy() / 255.0
        amap = maps[i].astype(np.float32)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        axes[0].imshow(img)
        axes[0].set_title("Input (AD2 real capture)")
        im = axes[1].imshow(amap, cmap="jet", vmin=0, vmax=max(amap.max(), 1e-6))
        axes[1].set_title(f"Anomaly map\nscore={scores[i]:.2f} (thr={thr:.2f})")
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        axes[2].imshow(img)
        axes[2].imshow(amap, cmap="jet", alpha=0.45)
        axes[2].set_title("Overlay")
        for a in axes:
            a.axis("off")
        fig.suptitle(f"{titles[rank-1]} {Path(paths[i]).parent.name}/{Path(paths[i]).name}", fontsize=10)
        fig.tight_layout()
        p = FIG / f"fig_{kind}{mode}_{rank:02d}_{Path(paths[i]).stem}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        out.append(p)
    return out


def main() -> int:
    from paths import require_artifacts

    require_artifacts({
        MET / "day3_ad2_per_image.csv": "python src/run_ad2.py --category sheet_metal --mode squash",
        MET / "day3_ad2_maps.npz": "python src/run_ad2.py --category sheet_metal --mode squash",
    })
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_any()
    if not runs:
        print("[ERR] 没有 day3_ad2_per_image*.csv，请先运行 run_ad2.py")
        return 1

    analysis = {}
    figs = []
    for name, per in runs.items():
        suffix = "" if name == "main" else "_aspect"
        thr_file = PROJECT_ROOT / "results" / "logs" / f"day3_ad2_run{suffix}.json"
        thr = json.loads(thr_file.read_text(encoding="utf-8"))["thresholds"]["image_mean_plus_3std"] if thr_file.exists() else 0.0

        mode = "" if name == "main" else "_aspect"
        figs.append(fig_score_dist(per, mode))
        p, area = fig_area_vs_recall(per, thr, mode)
        figs.append(p)
        figs += fig_cases(per, MET / f"day3_ad2_maps{suffix}.npz", thr, "normal", 3, mode)
        figs += fig_cases(per, MET / f"day3_ad2_maps{suffix}.npz", thr, "miss", 3, mode)
        figs += fig_cases(per, MET / f"day3_ad2_maps{suffix}.npz", thr, "hit", 2, mode)

        normal = per[per.gt_label == 0]
        defect = per[per.gt_label == 1]
        analysis[name] = {
            "threshold": thr,
            "n_normal": int(len(normal)),
            "n_defect": int(len(defect)),
            "normal_score_mean": float(normal.score_raw.mean()),
            "normal_score_max": float(normal.score_raw.max()),
            "defect_score_mean": float(defect.score_raw.mean()),
            "defect_score_median": float(defect.score_raw.median()),
            "n_false_positive": int((normal.pred_at_thr == 1).sum()),
            "n_missed": int((defect.pred_at_thr == 0).sum()),
            "recall_by_area": area,
            "defect_type_stats": {
                str(k): {"n": int(len(g)), "recall": float(g.pred_at_thr.mean()),
                         "mean_score": float(g.score_raw.mean())}
                for k, g in defect.groupby("defect_type")
            },
        }

    (MET / "day3_analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("recall_by_area", "defect_type_stats")}
                      for k, v in analysis.items()}, indent=2, ensure_ascii=False))
    print(f"\n[OK] {len(figs)} 张图 -> {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
