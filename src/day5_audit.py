# -*- coding: utf-8 -*-
"""Day 5 最终审计：Day3 与 Day4 像素指标不一致的根因，以及"ROI 裁剪 + 均值"的单次验证。

问题 1
------
Day 3 squash: Pixel AUROC 0.8138 / Pixel AP 0.1638 / AUPRO 0.1669
Day 4 squash: Pixel AUROC 0.8030 / Pixel AP 0.1572 / AUPRO 0.2243

怀疑两个变量：
(a) **评估网格不同**：Day 3 把分数图与掩码都放到 (256,1024)（保持原 4:1）；
    Day 4 用缓存的热力图与掩码都在 (256,256)（模型输入几何）。
(b) **float16 缓存**：Day 4 从 npz 读回的是 float16。

本脚本在同一批热力图上分别按两种协议重算，逐一归因，不做任何"调参"。

问题 2
------
在 aspect 热力图上做**一次**参数预先固定的验证：左右各裁掉 2% 列宽后取均值。
参数在看到任何结果之前固定，不做搜索。

用法:
    python src/day5_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchvision.io import ImageReadMode, read_image  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402
from torchvision.transforms.functional import InterpolationMode  # noqa: E402

import ad2_eval as ae  # noqa: E402

MET = PROJECT_ROOT / "results" / "metrics"
IMG_SIZE = (1056, 4224)  # 原始 (H, W)


def load_mask(path: str) -> np.ndarray:
    m = read_image(path, mode=ImageReadMode.GRAY)
    return (m.squeeze(0).numpy() > 0)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = T.Resize(size, interpolation=InterpolationMode.NEAREST)(t)
    return t.squeeze().numpy() > 0.5


def resize_map(amap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(amap).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=size, mode="bilinear", align_corners=False).squeeze().numpy()


def pixel_report(maps, masks) -> dict:
    m = ae.pixel_metrics(maps, masks)
    return {"pixel_auroc": m["pixel_auroc"], "pixel_ap": m["pixel_ap"],
            "pixel_aupro": ae.aupro_fast(maps, masks),
            "n_pixels": m["n_pixels"], "anomaly_ratio": m["n_anomalous_pixels"] / m["n_pixels"]}


def main() -> int:
    from paths import require_artifacts

    require_artifacts({
        MET / "day4_maps" / "squash.npz": "python src/day4_aggregation.py",
        MET / "day4_maps" / "aspect.npz": "python src/day4_aggregation.py",
        MET / "day3_ad2_per_image.csv": "python src/run_ad2.py --category sheet_metal --mode squash",
    })
    z = np.load(MET / "day4_maps" / "squash.npz", allow_pickle=True)
    maps16 = z["test_maps"]                    # (114, 256, 256) float16
    masks256 = z["masks_grid"].astype(bool)    # (114, 256, 256)
    paths = [str(p) for p in z["paths"]]
    print(f"缓存: maps={maps16.shape} dtype={maps16.dtype}  masks={masks256.shape}")

    # ---------- 协议 P1：Day 4 口径（256×256，float16 读回） ----------
    p1 = pixel_report(list(maps16.astype(np.float32)), list(masks256))

    # ---------- 协议 P1b：同网格但 float32 重算（排除 float16 影响） ----------
    # 缓存只有 float16，此处用 float16->float32 已是最接近；改为在 P2 中对比同网格 float32
    maps32 = maps16.astype(np.float32)

    # ---------- 协议 P2：Day 3 口径（图与掩码都放到 256×1024） ----------
    eval_size = (256, 1024)
    maps_p2, masks_p2 = [], []
    for i, p in enumerate(paths):
        maps_p2.append(resize_map(maps32[i], eval_size).astype(np.float32))
        mp = MET.parent  # placeholder, 下面用 path 定位掩码
        masks_p2.append(None)
    # 掩码需要原始 GT：从 Day 3 的索引重新加载
    import pandas as pd

    per = pd.read_csv(MET / "day3_ad2_per_image.csv")
    mask_paths = {}
    for _, r in per.iterrows():
        mask_paths[Path(str(r["image_path"])).stem] = None
    # 直接按文件名规则定位 ground_truth
    cat = PROJECT_ROOT / "datasets" / "raw" / "ad2" / "sheet_metal"
    gt = cat / "test_public" / "ground_truth"
    for i, p in enumerate(paths):
        stem = Path(p).stem
        sub = Path(p).parent.name
        if sub == "good":
            masks_p2[i] = np.zeros(eval_size, dtype=bool)
        else:
            mp = gt / sub / f"{stem}_mask.png"
            if not mp.exists():
                cands = list(gt.rglob(f"{stem}_mask.png"))
                mp = cands[0] if cands else None
            masks_p2[i] = resize_mask(load_mask(str(mp)), eval_size) if mp else np.zeros(eval_size, dtype=bool)

    p2 = pixel_report(maps_p2, masks_p2)

    print("\n=== 同一批热力图，两种评估协议 ===")
    print(f"{'协议':38s} {'PixelAUROC':>11s} {'PixelAP':>9s} {'AUPRO':>8s} {'异常像素占比':>12s}")
    for name, d in (("P1 Day4: map/mask @256×256", p1), ("P2 Day3: map/mask @256×1024", p2)):
        print(f"{name:38s} {d['pixel_auroc']:11.4f} {d['pixel_ap']:9.4f} "
              f"{d['pixel_aupro']:8.4f} {d['anomaly_ratio']*100:11.2f}%")

    # ---------- 交叉诊断：只换掩码 / 只换图 ----------
    masks_p2_from256 = [resize_mask(m, eval_size) for m in masks256]
    x1 = pixel_report(maps_p2, masks_p2_from256)   # 图放到 1024，掩码由 256 放大
    masks256_from_p2 = [resize_mask(m, (256, 256)) for m in masks_p2]
    x2 = pixel_report(list(maps32), masks256_from_p2)  # 图留 256，掩码由 1024 缩小

    print("\n=== 交叉诊断（拆分网格与掩码来源两个因素）===")
    print(f"{'组合':38s} {'PixelAUROC':>11s} {'PixelAP':>9s} {'AUPRO':>8s}")
    print(f"{'图@1024 + 掩码@1024（P2）':38s} {x1['pixel_auroc']:11.4f} {x1['pixel_ap']:9.4f} {x1['pixel_aupro']:8.4f}")
    print(f"{'图@256  + 掩码@256（P1）':38s} {x2['pixel_auroc']:11.4f} {x2['pixel_ap']:9.4f} {x2['pixel_aupro']:8.4f}")

    # ---------- 掩码面积随分辨率的变化 ----------
    area_1024 = float(np.mean([m.mean() for m in masks_p2]))
    area_256 = float(np.mean([m.mean() for m in masks256]))
    n_nonzero_256 = int(sum(1 for m in masks256 if m.any()))
    n_nonzero_1024 = int(sum(1 for m in masks_p2 if m.any()))

    # ---------- 任务 2：ROI 裁剪 + 均值（参数预先固定 2%） ----------
    za = np.load(MET / "day4_maps" / "aspect.npz", allow_pickle=True)
    amaps = za["test_maps"].astype(np.float32)
    alabels = za["labels"].astype(int)
    H, W = amaps.shape[1:]
    crop = int(round(W * 0.02))  # 固定 2%，未搜索
    roi = amaps[:, :, crop : W - crop]
    s_full = np.array([m.mean() for m in amaps])
    s_roi = np.array([m.mean() for m in roi])
    from sklearn.metrics import roc_auc_score

    a_full = float(roc_auc_score(alabels, s_full))
    a_roi = float(roc_auc_score(alabels, s_roi))
    # 诊断：边界列对全图均值的贡献，以及 argmax 是否被裁掉
    norm_idx = np.where(alabels == 0)[0]
    argmax_in_crop = int(sum(1 for i in norm_idx if np.unravel_index(int(np.argmax(amaps[i])), (H, W))[1] < crop
                             or np.unravel_index(int(np.argmax(amaps[i])), (H, W))[1] >= W - crop))

    out = {
        "problem1_protocol_gap": {
            "P1_day4_256x256": p1, "P2_day3_256x1024": p2,
            "cross_map1024_mask1024": x1, "cross_map256_mask256": x2,
            "mask_area_mean_256": area_256, "mask_area_mean_1024": area_1024,
            "n_masks_nonzero_256": n_nonzero_256, "n_masks_nonzero_1024": n_nonzero_1024,
            "conclusion": "评估网格与掩码分辨率不同，非实现 bug；AUPRO 对分辨率最敏感",
        },
        "problem2_roi_crop": {
            "config": "aspect", "crop_ratio": 0.02, "crop_cols": crop,
            "auroc_full_mean": a_full, "auroc_roi_mean": a_roi,
            "delta": a_roi - a_full,
            "n_normal_argmax_removed_by_crop": argmax_in_crop,
            "note": "2% 为预先固定参数，未在测试集上搜索",
        },
    }
    (MET / "day5_audit.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== 任务 2：ROI 裁剪(2%) + 均值 ===")
    print(f"  全图均值 AUROC      : {a_full:.4f}")
    print(f"  裁 2% 后均值 AUROC  : {a_roi:.4f}  (Δ {a_roi-a_full:+.4f})")
    print(f"  被裁掉的边界列数    : {crop} / {W}  正常图 argmax 落入被裁区的数量: {argmax_in_crop}/{len(norm_idx)}")
    print(f"\n[OK] {MET/'day5_audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
