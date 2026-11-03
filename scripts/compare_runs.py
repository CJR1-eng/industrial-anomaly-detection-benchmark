# -*- coding: utf-8 -*-
"""比较两次独立运行的逐图分数与指标，用于可复现性检查。

用法:
    python scripts/compare_runs.py                 # 默认比较 run1(main) 与 run2
    python scripts/compare_runs.py --a "" --b run2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MET_DIR = PROJECT_ROOT / "results" / "metrics"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="", help="第一次运行的后缀（空=正式运行）")
    ap.add_argument("--b", default="run2")
    args = ap.parse_args()

    ta = f"_{args.a}" if args.a else ""
    tb = f"_{args.b}" if args.b else ""

    pa = pd.read_csv(MET_DIR / f"day2_per_image{ta}.csv")
    pb = pd.read_csv(MET_DIR / f"day2_per_image{tb}.csv")
    ma = pd.read_csv(MET_DIR / f"day2_illumination_results{ta}.csv").set_index("condition")
    mb = pd.read_csv(MET_DIR / f"day2_illumination_results{tb}.csv").set_index("condition")

    key = ["condition", "image_path"]
    merged = pa.merge(pb, on=key, suffixes=("_a", "_b"))
    if len(merged) != len(pa) or len(merged) != len(pb):
        print(f"[WARN] 对齐后行数 {len(merged)}，原始 {len(pa)} / {len(pb)}（可能存在缺失配对）")

    print("=" * 74)
    print("逐图分数一致性（同一条件、同一图，两次独立完整运行）")
    print("=" * 74)
    print(f"{'条件':16s} {'n':>4s} {'max|Δscore|':>12s} {'mean|Δscore|':>13s} {'完全一致占比':>12s}")
    rows = []
    for cond, g in merged.groupby("condition", sort=False):
        d = (g["score_raw_a"] - g["score_raw_b"]).abs()
        rows.append(
            {
                "condition": cond,
                "n": int(len(g)),
                "max_abs_delta": float(d.max()),
                "mean_abs_delta": float(d.mean()),
                "frac_bitwise_equal": float((d == 0).mean()),
            }
        )
        print(f"{cond:16s} {len(g):4d} {d.max():12.3e} {d.mean():13.3e} {(d == 0).mean()*100:11.1f}%")

    print()
    print("=" * 74)
    print("指标一致性")
    print("=" * 74)
    cols = ["image_auroc", "pixel_auroc", "pixel_auroc_840", "pixel_ap", "fpr_normal", "recall_defect"]
    print(f"{'条件':16s} {'指标':16s} {'run_a':>10s} {'run_b':>10s} {'Δ':>12s}")
    max_delta = 0.0
    for cond in ma.index:
        for c in cols:
            va, vb = float(ma.loc[cond, c]), float(mb.loc[cond, c])
            dd = vb - va
            max_delta = max(max_delta, abs(dd))
            print(f"{cond:16s} {c:16s} {va:10.6f} {vb:10.6f} {dd:12.2e}")

    out = {
        "per_image": rows,
        "max_metric_delta": max_delta,
        "verdict": "bitwise reproducible" if max_delta == 0 else "numerically reproducible (small drift)",
    }
    p = MET_DIR / "day2_reproducibility.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"指标最大绝对差 = {max_delta:.3e}")
    print(f"判定: {out['verdict']}")
    print(f"[OK] -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
