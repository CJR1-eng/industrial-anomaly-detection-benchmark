# -*- coding: utf-8 -*-
"""Day 4 图表：聚合对比、Bootstrap CI、极值随位置数增长、边界定位、案例。

用法:
    python src/day4_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from day4_aggregation import aggregate  # noqa: E402

MET = PROJECT_ROOT / "results" / "metrics"
FIG = PROJECT_ROOT / "figures" / "day4_aggregation"
RULE_LABEL = {"max": "Max", "top1": "Top-1%", "top5": "Top-5%", "p99": "P99", "gmean": "Global mean"}


def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def fig_auroc(df: pd.DataFrame) -> Path:
    plt = mpl()
    rules = ["max", "top1", "top5", "p99", "gmean"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(rules))
    w = 0.36
    for k, (cfg, color) in enumerate((("squash", "#2b6cb0"), ("aspect", "#dd6b20"))):
        sub = df[df.config == cfg].set_index("rule").loc[rules]
        bars = ax.bar(x + (k - 0.5) * w, sub["image_auroc"].values, width=w,
                      color=color, label=f"{cfg} ({int(sub['n_positions'].iloc[0])} positions)")
        for b, v in zip(bars, sub["image_auroc"].values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([RULE_LABEL[r] for r in rules])
    ax.set_ylabel("Image AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Controlled aggregation experiment — identical maps, only the image-level rule changes")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig01_aggregation_auroc.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_bootstrap(extras: dict) -> Path:
    plt = mpl()
    fig, ax = plt.subplots(figsize=(9, 4))
    rules = ["top1", "top5", "p99", "gmean"]
    y = np.arange(len(rules))
    for k, (cfg, color) in enumerate((("squash", "#2b6cb0"), ("aspect", "#dd6b20"))):
        d = extras[cfg]["bootstrap"]["delta_vs_max"]
        pts = [d[r]["point"] for r in rules]
        lo = [d[r]["ci95"][0] for r in rules]
        hi = [d[r]["ci95"][1] for r in rules]
        off = (k - 0.5) * 0.22
        ax.errorbar(pts, y + off, xerr=[np.array(pts) - np.array(lo), np.array(hi) - np.array(pts)],
                    fmt="o", color=color, capsize=4, label=f"{cfg} (grouped bootstrap, 19 scenes)")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([RULE_LABEL[r] for r in rules])
    ax.set_xlabel("Δ Image AUROC vs Max   (95% CI, scene-level bootstrap)")
    ax.set_title("Paired comparison against Max on identical maps")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig02_bootstrap_delta.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_positions(extras: dict) -> Path:
    plt = mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, cfg in zip(axes, ("squash", "aspect")):
        g = extras[cfg]["extremes"]["max_vs_positions"]
        ns = [int(k) for k in g]
        nm = [g[str(n)]["normal_mean_max"] for n in ns]
        dm = [g[str(n)]["defect_mean_max"] for n in ns]
        ax.plot(ns, nm, "o-", color="#2b6cb0", label="normal images")
        ax.plot(ns, dm, "s-", color="#c53030", label="defect images")
        ax.fill_between(ns, nm, dm, color="gray", alpha=0.18)
        ax.set_xscale("log")
        ax.set_xlabel("# sampled positions (log)")
        ax.set_ylabel("mean of per-image max")
        ax.set_title(f"{cfg}: max inflates with position count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        for n, a, b in zip(ns, nm, dm):
            ax.annotate(f"gap {b-a:.1f}", (n, (a + b) / 2), fontsize=7, ha="center")
    fig.suptitle("Controlled position subsampling on identical anomaly maps", fontsize=11)
    fig.tight_layout()
    p = FIG / "fig03_max_vs_positions.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_argmax_border(extras: dict) -> Path:
    plt = mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cfg in zip(axes, ("squash", "aspect")):
        z = np.load(MET / "day4_maps" / f"{cfg}.npz", allow_pickle=True)
        maps = z["test_maps"].astype(np.float32)
        labels = z["labels"].astype(int)
        H, W = maps.shape[1:]
        d = []
        for i in np.where(labels == 0)[0]:
            r, c = np.unravel_index(int(np.argmax(maps[i])), (H, W))
            d.append(min(c, W - 1 - c) / (W - 1))
        ax.hist(d, bins=25, color="#4a5568")
        ax.axvline(0.1, color="#c53030", ls="--", lw=1.4, label="10% border band")
        ax.set_xlabel("normalised distance of argmax to LEFT/RIGHT border")
        ax.set_ylabel("# normal images")
        frac = float(np.mean(np.array(d) < 0.1))
        ax.set_title(f"{cfg}: {frac*100:.0f}% of normals peak at the border")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Where does Max look? (normal test images)", fontsize=11)
    fig.tight_layout()
    p = FIG / "fig04_argmax_border.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_score_dist() -> Path:
    plt = mpl()
    z = np.load(MET / "day4_maps" / "aspect.npz", allow_pickle=True)
    maps = z["test_maps"].astype(np.float32)
    labels = z["labels"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, rule in zip(axes, ("max", "gmean")):
        s = np.array([aggregate(m, rule) for m in maps])
        for lab, color, nm in ((0, "#2b6cb0", "normal"), (1, "#c53030", "defect")):
            ax.hist(s[labels == lab], bins=25, alpha=0.65, color=color,
                    label=f"{nm} (n={int((labels==lab).sum())})")
        ax.set_title(f"aspect — {RULE_LABEL[rule]}")
        ax.set_xlabel("image score")
        ax.set_ylabel("# images")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Same maps, different aggregation: separation changes drastically", fontsize=11)
    fig.tight_layout()
    p = FIG / "fig05_score_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_case() -> Path:
    """正常图的最大值案例：在地图上标出 argmax 位置。"""
    from torchvision.io import ImageReadMode, read_image

    plt = mpl()
    z = np.load(MET / "day4_maps" / "aspect.npz", allow_pickle=True)
    maps = z["test_maps"].astype(np.float32)
    labels = z["labels"].astype(int)
    paths = [str(p) for p in z["paths"]]
    s = np.array([aggregate(m, "max") for m in maps])
    cand = np.where(labels == 0)[0]
    i = cand[int(np.argmax(s[cand]))]

    img = read_image(paths[i], mode=ImageReadMode.RGB).permute(1, 2, 0).numpy() / 255.0
    amap = maps[i]
    H, W = amap.shape
    r, c = np.unravel_index(int(np.argmax(amap)), (H, W))

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    axes[0].imshow(img)
    axes[0].set_title("Input (wide strip, 4:1)")
    im = axes[1].imshow(amap, cmap="jet")
    axes[1].plot(c, r, "wx", ms=14, mew=2.5)
    axes[1].set_title(f"Anomaly map (max at col {c}/{W})\nscore={amap.max():.1f}")
    plt.colorbar(im, ax=axes[1], fraction=0.035)
    axes[2].imshow(amap, cmap="jet")
    axes[2].axvline(W * 0.02, color="w", ls=":", lw=1)
    axes[2].axvline(W * 0.98, color="w", ls=":", lw=1)
    axes[2].set_title("Same map: hot columns sit on the strip edges")
    for a in axes:
        a.axis("off")
    fig.suptitle(f"NORMAL image, highest Max score — argmax pinned to the strip border "
                 f"({paths[i].split('/')[-1]})", fontsize=10)
    fig.tight_layout()
    p = FIG / "fig06_case_border_argmax.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> int:
    from paths import require_artifacts

    require_artifacts({
        MET / "day4_aggregation.csv": "python src/day4_aggregation.py",
        MET / "day4_extremes_bootstrap.json": "python src/day4_aggregation.py",
        MET / "day4_maps" / "squash.npz": "python src/day4_aggregation.py",
        MET / "day4_maps" / "aspect.npz": "python src/day4_aggregation.py",
    })
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MET / "day4_aggregation.csv")
    extras = json.loads((MET / "day4_extremes_bootstrap.json").read_text(encoding="utf-8"))
    out = [fig_auroc(df), fig_bootstrap(extras), fig_positions(extras),
           fig_argmax_border(extras), fig_score_dist(), fig_case()]
    for p in out:
        print(" ", p.name)
    print(f"[OK] {len(out)} 张 -> {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
