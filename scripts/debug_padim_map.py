# -*- coding: utf-8 -*-
"""Paind 调试：对比 PaDiM 原始马氏距离图与后处理后的 anomaly_map。

目的：定位"热力图 95% 像素为 0、Pixel AUROC 接近随机"的原因，
判断问题出在模型输出、后处理归一化，还是评估口径。

用法:
    python scripts/debug_padim_map.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from anomalib.data import MVTecAD  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.models.image.padim import Padim  # noqa: E402


def stats(name: str, t: torch.Tensor) -> None:
    x = t.detach().float().cpu()
    flat = x.reshape(-1)
    print(
        f"  {name:22s} shape={tuple(x.shape)} min={flat.min():.6g} max={flat.max():.6g} "
        f"mean={flat.mean():.6g} >0_ratio={float((flat > 1e-6).float().mean()):.4f}"
    )


def main() -> int:
    dm = MVTecAD(
        root="datasets/raw/mvtec_ad", category="tile",
        train_batch_size=8, eval_batch_size=8, num_workers=0,
        val_split_mode="from_train", val_split_ratio=0.2,
        test_split_mode="from_dir", seed=42,
    )
    model = Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"],
                  pre_trained=True, n_features=100)
    engine = Engine(default_root_dir="results/runs/debug", max_epochs=1,
                    accelerator="gpu", devices=1)
    engine.fit(model=model, datamodule=dm)

    dm.setup()
    loader = dm.test_dataloader()
    batch = next(iter(loader))
    print("\n[batch] image", tuple(batch.image.shape),
          "| gt_label", batch.gt_label.tolist(),
          "| gt_mask nonzero per img", [int(x.sum()) for x in batch.gt_mask])

    model.eval()
    model.to("cuda")  # engine.fit 结束后模块可能已回到 CPU，显式搬回 GPU
    with torch.no_grad():
        # 1) 模型原始输出（不经过 post_processor）
        raw = model.model(batch.image.cuda())
        print("\n[1] 模型原始输出")
        stats("anomaly_map(raw)", raw.anomaly_map)
        stats("pred_score(raw)", raw.pred_score)

        # 2) 经过 post_processor 后的完整模块输出
        full = model(batch.image.cuda())
        print("\n[2] 经 post_processor")
        stats("anomaly_map(post)", full.anomaly_map)
        stats("pred_score(post)", full.pred_score)
        if getattr(full, "pred_mask", None) is not None:
            stats("pred_mask", full.pred_mask.float())

        # 3) 单张异常图：原始距离 vs GT 掩码的空间对应
        gt = batch.gt_mask
        idx = None
        for i in range(len(batch.gt_label)):
            if float(batch.gt_label[i]) == 1 and float(gt[i].sum()) > 0:
                idx = i
                break
        if idx is not None:
            m_raw = raw.anomaly_map[idx, 0].float().cpu().numpy()
            m_post = full.anomaly_map[idx, 0].float().cpu().numpy()
            g = (gt[idx, 0].cpu().numpy() > 0.5)
            print(f"\n[3] 异常样本 idx={idx}, GT 缺陷像素占比={g.mean():.4f}")
            print(f"  raw  图: 缺陷区均值={m_raw[g].mean():.4g} / 正常区均值={m_raw[~g].mean():.4g}"
                  f"  比值={m_raw[g].mean() / max(m_raw[~g].mean(), 1e-12):.2f}")
            print(f"  post 图: 缺陷区均值={m_post[g].mean():.4g} / 正常区均值={m_post[~g].mean():.4g}"
                  f"  比值={m_post[g].mean() / max(m_post[~g].mean(), 1e-12):.2f}")
        else:
            print("\n[3] 本 batch 无缺陷图，跳过空间对应检查")

        # 4) post_processor 的归一化配置
        pp = model.post_processor
        print("\n[4] post_processor 配置")
        for attr in ("normalization", "threshold", "image_threshold", "pixel_threshold"):
            print(f"  {attr} = {getattr(pp, attr, '<无>')}")

        # 5) 存图：原图 / 原始距离图 / 后处理图 / GT 掩码
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = PROJECT_ROOT / "figures" / "debug"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for i in range(min(3, len(batch.gt_label))):
            img = batch.image[i].permute(1, 2, 0).float().cpu().numpy()
            img = np.clip(img * 0.226 + 0.449, 0, 1)  # ImageNet 反归一化，仅用于查看
            m_raw = raw.anomaly_map[i, 0].float().cpu().numpy()
            m_post = full.anomaly_map[i, 0].float().cpu().numpy()
            g = batch.gt_mask[i].float().cpu().numpy()
            fig, ax = plt.subplots(1, 4, figsize=(15, 4))
            ax[0].imshow(img); ax[0].set_title("image (denorm)")
            im1 = ax[1].imshow(m_raw, cmap="jet"); ax[1].set_title(f"raw distance\n[{m_raw.min():.1f},{m_raw.max():.1f}]")
            plt.colorbar(im1, ax=ax[1], fraction=0.046)
            ax[2].imshow(m_post, cmap="jet"); ax[2].set_title("post-processed")
            ax[3].imshow(g, cmap="gray"); ax[3].set_title("GT mask")
            for a in ax:
                a.axis("off")
            fig.tight_layout()
            p = fig_dir / f"debug_map_img{i}.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            print(f"  已保存 {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
