# -*- coding: utf-8 -*-
"""消融：拟合样本量对 PaDiM 指标的影响。

协议 A（主实验，无泄漏）：train/good 切 80/20，184 张拟合高斯，46 张定阈值。
协议 B（文献口径）：用全部 230 张 train/good 拟合，只报告无阈值指标。

目的：量化"为了留出验证集而定阈值"付出的精度代价，
避免把样本量损失误判为模型/实现问题。

用法:
    python src/ablation_full_train.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from anomalib.data import MVTecAD  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.models.image.padim import Padim  # noqa: E402

from run_padim import infer_raw, per_image_pixel_auroc  # noqa: E402


def run_protocol(val_split_mode: str, tag: str) -> dict:
    dm = MVTecAD(
        root="datasets/raw/mvtec_ad", category="tile",
        train_batch_size=8, eval_batch_size=8, num_workers=0,
        val_split_mode=val_split_mode, val_split_ratio=0.2,
        test_split_mode="from_dir", seed=42,
    )
    model = Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"],
                  pre_trained=True, n_features=100)
    engine = Engine(
        default_root_dir=f"results/runs/ablation_{tag}", max_epochs=1,
        accelerator="gpu", devices=1,
        # val_split_mode=none 时没有验证集，必须显式关闭，否则 Lightning 仍会请求 val_dataloader
        limit_val_batches=0 if val_split_mode == "none" else 1.0,
    )
    engine.fit(model=model, datamodule=dm)
    dm.setup()

    imgs, masks, scores, maps = infer_raw(model, dm.test_dataloader(), return_images=True)
    labels = np.array([1 if m.sum() > 0 else 0 for m in masks])
    y = masks.reshape(-1).astype(np.uint8)
    s = maps.reshape(-1).astype(np.float64)
    res = {
        "protocol": tag,
        "val_split_mode": val_split_mode,
        "n_train_fit": len(dm.train_data),
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_ap": float(average_precision_score(labels, scores)),
        "pixel_auroc": float(roc_auc_score(y, s)),
        "pixel_ap": float(average_precision_score(y, s)),
        "pixel_auroc_per_image_mean": per_image_pixel_auroc(maps, masks),
    }
    del engine, model
    torch.cuda.empty_cache()
    return res


def main() -> int:
    out = []
    for mode, tag in [("from_train", "A_fit184"), ("none", "B_fit230")]:
        print(f"\n=== 协议 {tag} (val_split_mode={mode}) ===")
        out.append(run_protocol(mode, tag))
        print(json.dumps(out[-1], indent=2, ensure_ascii=False))

    p = PROJECT_ROOT / "results" / "metrics" / "ablation_train_size.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] -> {p}")

    print("\n| 协议 | 拟合样本 | Image AUROC | Pixel AUROC | Pixel AUROC(逐图) | Pixel AP |")
    print("|---|---|---|---|---|---|")
    for r in out:
        print(f"| {r['protocol']} | {r['n_train_fit']} | {r['image_auroc']:.4f} | "
              f"{r['pixel_auroc']:.4f} | {r['pixel_auroc_per_image_mean']:.4f} | {r['pixel_ap']:.4f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
