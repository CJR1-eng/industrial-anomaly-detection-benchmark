# -*- coding: utf-8 -*-
"""Day 4 跨类别核查：在 MVTec AD / tile 的已保存热力图上重复聚合实验。

数据来源：Day 2 保存的 `results/metrics/day2_maps.npz`（条件 original，117 张）。
限制：Day 2 **未保存验证集热力图**，因此本脚本只报告无阈值指标
（Image AUROC / AP），不报告误报率/召回率。这一点必须如实标注。

用法:
    python src/day4_tile_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from day4_aggregation import RULES, aggregate  # noqa: E402

MET = PROJECT_ROOT / "results" / "metrics"


def main() -> int:
    from paths import require_artifacts

    require_artifacts({
        MET / "day2_maps.npz": "python src/run_illumination.py",
    })
    z = np.load(MET / "day2_maps.npz", allow_pickle=True)
    conds = [str(c) for c in z["conditions"]]
    i = conds.index("original") if "original" in conds else 0
    maps = z["maps"][i].astype(np.float32)
    labels = z["labels"].astype(int)
    print(f"tile / original: maps={maps.shape} 正常={int((labels==0).sum())} 缺陷={int((labels==1).sum())}")

    rows = []
    scores = {}
    for rule in RULES:
        s = np.array([aggregate(m, rule) for m in maps])
        scores[rule] = s
        rows.append({
            "dataset": "mvtec_ad_tile", "config": f"256x256", "rule": rule,
            "n": len(s), "n_normal": int((labels == 0).sum()), "n_defect": int((labels == 1).sum()),
            "image_auroc": float(roc_auc_score(labels, s)),
            "image_ap": float(average_precision_score(labels, s)),
            "threshold": float("nan"), "fpr_normal": float("nan"), "recall_defect": float("nan"),
            "note": "验证集热力图未保存，无阈值指标",
        })

    df = pd.DataFrame(rows)
    print()
    print(df[["rule", "image_auroc", "image_ap"]].round(4).to_string(index=False))

    # 分场景/配对：tile 无光照分组，这里按"文件名编号"做一个小规模分组 bootstrap（仅作参考）
    from sklearn.metrics import roc_auc_score as auc

    rng = np.random.default_rng(42)
    n = len(labels)
    boot = {r: [] for r in RULES if r != "max"}
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        if len(np.unique(y)) < 2:
            continue
        a0 = auc(y, scores["max"][idx])
        for r in boot:
            boot[r].append(auc(y, scores[r][idx]) - a0)
    print("\n单图 bootstrap ΔAUROC vs max（仅供参考，tile 无分组结构）:")
    for r, v in boot.items():
        v = np.asarray(v)
        print(f"  {r:6s} {np.mean(v):+.4f}  95%CI[{np.percentile(v,2.5):+.4f}, {np.percentile(v,97.5):+.4f}]")

    out = MET / "day4_aggregation_tile.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    (MET / "day4_tile_bootstrap.json").write_text(
        json.dumps({r: {"mean": float(np.mean(v)),
                        "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
                    for r, v in boot.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
