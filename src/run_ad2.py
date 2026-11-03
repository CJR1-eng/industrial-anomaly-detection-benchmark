# -*- coding: utf-8 -*-
"""在 MVTec AD 2 真实光照数据上运行 PaDiM（冻结骨干，原始异常分数）。

与 Day 2 的差异
---------------
1. 数据源换成官方 AD2（train / validation / test_public 三段）。
   `test_private` 的 GT 不公开，**不参与任何评估**。
2. AD2 图像非正方形且分辨率高，因此提供两种输入几何：
   - `squash`  : 直接压到 256×256（有信息损失，作为调试对照）
   - `aspect`  : 短边缩到 256，长边按比例（保持几何，位置数变多）
3. 像素指标在 **评估尺度** 上计算（默认长边 ≤1024，可 `--max-eval-side 0` 用原图），
   分数图双线性上采样、掩码最近邻，与任务要求一致。

用法:
    python src/run_ad2.py --category sheet_metal --mode squash
    python src/run_ad2.py --category sheet_metal --mode aspect --tag aspect
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
from paths import MVTEC_AD2_URL, ad2_root  # noqa: E402

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
AD2_ROOT = ad2_root()


# ------------------------------------------------------------------ indexing
def find_category(root: Path, name: str) -> Path:
    """Locate ``<category>/`` under the AD2 root (layout-tolerant)."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"\n\nMVTec AD 2 root not found: {root}\n"
            f"Download the dataset from the official source and extract it there:\n"
            f"  {MVTEC_AD2_URL}\n"
            f"Or set IDAD_DATA_ROOT to a directory containing 'ad2/'.\n"
        )
    for p in root.rglob("*"):
        if p.is_dir() and p.name == name and (p / "train").is_dir():
            return p
    available = sorted({q.name for q in root.rglob("*") if q.is_dir() and (q / "train").is_dir()})
    raise FileNotFoundError(
        f"\n\nCategory '{name}' not found under {root}\n"
        f"Available categories: {available or 'none'}\n"
    )


def list_imgs(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


LIGHT_ORDER = ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3"]


def parse_lighting(stem: str) -> str:
    """AD2 文件名形如 000_overexposed / 000_shift_1，后缀即官方光照条件标签。"""
    parts = stem.split("_", 1)
    if len(parts) == 2:
        return parts[1]
    return "unknown"


def build_index(cat_dir: Path) -> dict:
    """构造 train / val / test_public 索引，并核对 GT 对应关系。"""
    idx: dict = {"train": [], "val": [], "test": [], "issues": []}

    for split_name, key in (("train", "train"), ("validation", "val")):
        sp = cat_dir / split_name
        for sub in sorted(p for p in sp.iterdir() if p.is_dir()):
            for f in list_imgs(sub):
                idx[key].append({"path": str(f), "label": 0, "defect": sub.name, "mask": None,
                                 "lighting": parse_lighting(f.stem), "scene": f.stem.split("_")[0]})

    tp = cat_dir / "test_public"
    # ground_truth 可能挂在 test_public/ 或类别根目录下，自动定位
    gt_root = tp / "ground_truth"
    if not gt_root.is_dir():
        cands = list(cat_dir.rglob("ground_truth"))
        if cands:
            gt_root = cands[0]
            idx["issues"].append(f"ground_truth 定位到: {gt_root}")
    for sub in sorted(p for p in tp.iterdir() if p.is_dir()):
        if sub.name == "ground_truth":
            continue
        label = 0 if sub.name == "good" else 1
        for f in list_imgs(sub):
            mask = None
            if label == 1:
                cands = [
                    gt_root / sub.name / f"{f.stem}_mask{f.suffix}",
                    gt_root / sub.name / f"{f.stem}{f.suffix}",
                ]
                for c in cands:
                    if c.exists():
                        mask = str(c)
                        break
                if mask is None:
                    idx["issues"].append(f"缺少掩码: {f}")
            idx["test"].append({"path": str(f), "label": label, "defect": sub.name, "mask": mask,
                                "lighting": parse_lighting(f.stem), "scene": f.stem.split("_")[0]})

    idx["counts"] = {k: len(v) for k, v in idx.items() if isinstance(v, list)}
    idx["defect_types"] = sorted({x["defect"] for x in idx["test"]})
    return idx


# ------------------------------------------------------------------ io / transform
def load_u8(path: str) -> torch.Tensor:
    return read_image(path, mode=ImageReadMode.RGB)


def apply_preproc(img_u8: torch.Tensor, kind: str, target_mean: float | None = None) -> torch.Tensor:
    """测试期简单预处理（任务 5 的对照，全部**非学习、参数预先固定**）。

    - none            : 原图
    - brightness_norm : 把该图全局均值线性拉到训练集均值（不用任何标签）
    - clahe           : LAB 的 L 通道做 CLAHE，clipLimit=2.0, tile=8
    - gamma_fixed     : x^0.85（固定，不按条件调参）

    注意：模型始终是在**原图**上拟合的，这里只改测试输入，
    因此回答的是"仅靠测试期预处理能否救回来"。
    """
    if kind == "none":
        return img_u8

    x = img_u8.to(torch.float32) / 255.0
    if kind == "brightness_norm":
        if target_mean is None:
            return img_u8
        mu = float(x.mean())
        scale = target_mean / max(mu, 1e-3)
        y = (x * scale).clamp(0, 1)
    elif kind == "gamma_fixed":
        y = torch.pow(x.clamp_min(0), 0.85)
    elif kind == "clahe":
        import cv2

        arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        arr2 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return torch.from_numpy(arr2).permute(2, 0, 1).contiguous()
    else:
        msg = f"未知 preproc: {kind}"
        raise ValueError(msg)
    return torch.round(y.clamp(0, 1) * 255).to(torch.uint8)


def to_input(img_u8: torch.Tensor, mode: str, preproc: str = "none",
             target_mean: float | None = None) -> torch.Tensor:
    img_u8 = apply_preproc(img_u8, preproc, target_mean)
    x = img_u8.to(torch.float32).div(255.0)
    if mode == "squash":
        size = [256, 256]
    elif mode == "aspect":
        h, w = x.shape[-2:]
        if h <= w:
            size = [256, int(round(256 * w / h))]
        else:
            size = [int(round(256 * h / w)), 256]
        size = [((s + 15) // 16) * 16 for s in size]  # 保证能被 16 整除
    else:
        msg = f"未知 mode: {mode}"
        raise ValueError(msg)
    x = T.Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=True)(x)
    return T.Normalize(mean=MEAN, std=STD)(x)


def load_mask(path: str) -> np.ndarray:
    m = read_image(path, mode=ImageReadMode.GRAY)
    return (m.squeeze(0).numpy() > 0)


def resize_map(amap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(amap).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = T.Resize(size, interpolation=InterpolationMode.NEAREST)(t)
    return t.squeeze().numpy() > 0.5


# ------------------------------------------------------------------ model
def fit_padim(train_items, mode: str, device: str, seed: int, batch_size: int = 4,
              n_features: int = 100, max_train: int | None = None):
    from anomalib.models.image.padim.torch_model import PadimModel

    # n_features 的通道子集是 randperm 选的，必须在构造前固定种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PadimModel(backbone="resnet18", layers=["layer1", "layer2", "layer3"],
                       pre_trained=True, n_features=n_features)
    model = model.to(device).eval()

    items = train_items[:max_train] if max_train else train_items
    t0 = time.time()
    for i in range(0, len(items), batch_size):
        xs = torch.stack([to_input(load_u8(it["path"]), mode) for it in items[i : i + batch_size]])
        with torch.no_grad():
            # 与 PadimModel.forward 内部完全一致：feature_extractor -> generate_embedding
            feats = model.feature_extractor(xs.to(device))
            emb = model.generate_embedding(feats)
        model.memory_bank.append(emb)
        if i % 40 == 0:
            print(f"    fit {i}/{len(items)}", flush=True)
    model.fit()
    peek = model.gaussian.mean.shape
    return model, {"n_fit": len(items), "fit_seconds": round(time.time() - t0, 1),
                   "gaussian_mean_shape": list(peek)}


@torch.no_grad()
def infer(model, items, mode: str, device: str, batch_size: int = 4,
          preproc: str = "none", target_mean: float | None = None):
    scores, maps = [], []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        xs = torch.stack([to_input(load_u8(it["path"]), mode, preproc, target_mean) for it in chunk]).to(device)
        out = model(xs)
        s = out.pred_score.reshape(-1).float().cpu().numpy()
        m = out.anomaly_map[:, 0].float().cpu().numpy()
        scores.extend(s.tolist())
        maps.extend(list(m))
    return np.asarray(scores, dtype=np.float64), maps


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="sheet_metal")
    ap.add_argument("--mode", default="squash", choices=["squash", "aspect"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=None, help="调试用，限制训练图数量")
    ap.add_argument("--max-eval-side", type=int, default=1024,
                    help="像素评估时长边上限；0 表示用原图分辨率")
    ap.add_argument("--preproc", default="none",
                    choices=["none", "brightness_norm", "clahe", "gamma_fixed"],
                    help="测试期简单预处理对照（模型始终在原图上拟合）")
    args = ap.parse_args()

    tag = f"_{args.tag}" if args.tag else ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cat_dir = find_category(AD2_ROOT, args.category)
    print(f"类别目录: {cat_dir}")
    idx = build_index(cat_dir)
    print(f"索引: {idx['counts']}  缺陷类型: {idx['defect_types']}")
    if idx["issues"]:
        print(f"[WARN] {len(idx['issues'])} 个问题，例如: {idx['issues'][:3]}")

    print(f"[1/4] 拟合 PaDiM（mode={args.mode}）...")
    model, fit_info = fit_padim(idx["train"], args.mode, device, args.seed, max_train=args.max_train)
    print(f"      {fit_info}")

    print("[2/4] 验证集定阈值...")
    # brightness_norm 的目标均值：只用**训练集**统计，不看测试标签
    train_mean = float(np.mean([float(load_u8(it["path"]).float().mean()) / 255.0 for it in idx["train"][:60]]))
    val_scores, val_maps = infer(model, idx["val"], args.mode, device,
                                 preproc=args.preproc, target_mean=train_mean)
    thr_img = float(val_scores.mean() + 3 * val_scores.std())
    thr_pix = float(np.quantile(np.concatenate([m.reshape(-1) for m in val_maps]), 0.995))
    print(f"      val n={len(val_scores)} mean={val_scores.mean():.3f} std={val_scores.std():.3f} "
          f"max={val_scores.max():.3f} -> thr={thr_img:.3f}")

    print("[3/4] test_public 推理...")
    t0 = time.time()
    test_scores, test_maps = infer(model, idx["test"], args.mode, device,
                                   preproc=args.preproc, target_mean=train_mean)
    infer_s = round(time.time() - t0, 1)
    labels = np.array([it["label"] for it in idx["test"]])

    # ---- 评估尺度 ----
    maps_eval, masks_eval = [], []
    for it, m in zip(idx["test"], test_maps):
        mp = it["mask"]
        img = load_u8(it["path"])
        h, w = img.shape[-2:]
        scale = 1.0
        if args.max_eval_side and max(h, w) > args.max_eval_side:
            scale = args.max_eval_side / max(h, w)
        eh, ew = int(round(h * scale)), int(round(w * scale))
        maps_eval.append(resize_map(m, (eh, ew)).astype(np.float32))
        if mp is not None:
            masks_eval.append(resize_mask(load_mask(mp), (eh, ew)))
        else:
            masks_eval.append(np.zeros((eh, ew), dtype=bool))

    print(f"      评估分辨率示例: {maps_eval[0].shape}（上限 {args.max_eval_side}）")

    print("[4/4] 计算指标...")
    img_m = ae.image_metrics(test_scores, labels)
    pix_m = ae.pixel_metrics(maps_eval, masks_eval)
    aupro = ae.aupro_fast(maps_eval, masks_eval)
    thm = ae.threshold_metrics(test_scores, labels, thr_img)
    pi = ae.pixel_auroc_per_image(maps_eval, masks_eval)

    # 按缺陷类型分组
    per_type = {}
    for t in idx["defect_types"]:
        sel = [i for i, it in enumerate(idx["test"]) if it["defect"] == t]
        if not sel:
            continue
        y = labels[sel]
        entry = {"n": len(sel), "n_defect": int(y.sum())}
        if 0 < y.sum() < len(y):
            entry.update(ae.image_metrics(test_scores[sel], y))
        entry["recall_at_thr"] = float((test_scores[sel][y == 1] >= thr_img).mean()) if y.sum() else None
        entry["mean_score"] = float(test_scores[sel].mean())
        per_type[t] = entry

    # ---- 按光照条件分组 ----
    per_light = {}
    for L in LIGHT_ORDER + sorted({it["lighting"] for it in idx["test"]} - set(LIGHT_ORDER)):
        sel = [i for i, it in enumerate(idx["test"]) if it["lighting"] == L]
        if not sel:
            continue
        y = labels[sel]
        sc = test_scores[sel]
        entry = {"n": len(sel), "n_normal": int((y == 0).sum()), "n_defect": int((y == 1).sum())}
        entry["score_mean_normal"] = float(sc[y == 0].mean()) if (y == 0).any() else None
        entry["score_mean_defect"] = float(sc[y == 1].mean()) if (y == 1).any() else None
        entry["recall_at_thr"] = float((sc[y == 1] >= thr_img).mean()) if (y == 1).any() else None
        entry["fpr_at_thr"] = float((sc[y == 0] >= thr_img).mean()) if (y == 0).any() else None
        per_light[L] = entry

    # ---- 同一场景不同光照的配对分析（scene 前缀相同 = 同一物体同一位置）----
    paired = {}
    scene_idx = {}
    for i, it in enumerate(idx["test"]):
        scene_idx.setdefault((it["scene"], it["label"]), {})[it["lighting"]] = i
    for (scene, lab), d in sorted(scene_idx.items()):
        if "regular" not in d or len(d) < 2:
            continue
        base = d["regular"]
        paired[f"{scene}_{'defect' if lab else 'normal'}"] = {
            L: round(float(test_scores[d[L]] - test_scores[base]), 3)
            for L in d if L != "regular"
        }

    row = {
        "category": args.category, "mode": args.mode, "preproc": args.preproc,
        "eval_max_side": args.max_eval_side,
        "n_train_fit": fit_info["n_fit"], "n_val": len(val_scores), "n_test": len(test_scores),
        "n_test_normal": int((labels == 0).sum()), "n_test_defect": int((labels == 1).sum()),
        "image_auroc": img_m["image_auroc"], "image_ap": img_m["image_ap"],
        "pixel_auroc": pix_m["pixel_auroc"], "pixel_ap": pix_m["pixel_ap"],
        "pixel_aupro": aupro, "pixel_auroc_per_image": pi,
        "fpr_normal": thm["fpr_normal"], "recall_defect": thm["recall_defect"],
        "precision": thm["precision"], "f1": thm["f1"],
        "tp": thm["tp"], "fp": thm["fp"], "fn": thm["fn"], "tn": thm["tn"],
        "threshold_image": thr_img, "threshold_pixel": thr_pix,
        "fit_seconds": fit_info["fit_seconds"], "infer_seconds": infer_s,
        "gaussian_positions": int(np.prod(fit_info["gaussian_mean_shape"][1:])),
    }

    import pandas as pd

    mdir = PROJECT_ROOT / "results" / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(mdir / f"day3_ad2_results{tag}.csv", index=False, float_format="%.6f")

    per_rows = []
    for it, s, m, ev in zip(idx["test"], test_scores, test_maps, maps_eval):
        per_rows.append({
            "image_path": it["path"], "defect_type": it["defect"], "gt_label": it["label"],
            "lighting": it["lighting"], "scene": it["scene"],
            "score_raw": float(s), "pred_at_thr": int(s >= thr_img),
            "map_max": float(m.max()), "map_mean": float(m.mean()),
            "map_mean_in_gt": float(ev[masks_eval[len(per_rows)]].mean()) if masks_eval[len(per_rows)].any() else float("nan"),
            "gt_area_ratio": float(masks_eval[len(per_rows)].mean()),
            "eval_h": int(ev.shape[0]), "eval_w": int(ev.shape[1]),
        })
    pd.DataFrame(per_rows).to_csv(mdir / f"day3_ad2_per_image{tag}.csv", index=False, float_format="%.6f")

    np.savez_compressed(
        mdir / f"day3_ad2_maps{tag}.npz",
        scores=test_scores, labels=labels.astype(np.uint8),
        maps=np.asarray(test_maps, dtype=np.float16),
        paths=np.array([it["path"] for it in idx["test"]]),
        defect_types=np.array([it["defect"] for it in idx["test"]]),
    )

    summary = {
        "args": vars(args), "index_counts": idx["counts"], "defect_types": idx["defect_types"],
        "thresholds": {"image_mean_plus_3std": thr_img, "pixel_p99.5": thr_pix,
                       "val_mean": float(val_scores.mean()), "val_std": float(val_scores.std()),
                       "val_max": float(val_scores.max())},
        "metrics": row, "per_defect_type": per_type,
        "per_lighting": per_light, "paired_score_delta_vs_regular": paired,
        "notes": [
            "test_private / test_private_mixed 的 GT 不公开，未使用",
            "分数为模型原始马氏距离，未经 PostProcessor 归一化",
            f"像素指标评估尺度：长边上限 {args.max_eval_side}（0=原图）",
        ],
    }
    (PROJECT_ROOT / "results" / "logs" / f"day3_ad2_run{tag}.json").parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "results" / "logs" / f"day3_ad2_run{tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n=== 结果 ===")
    for k in ("image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "pixel_aupro",
              "fpr_normal", "recall_defect", "f1"):
        print(f"  {k:22s} {row[k]:.4f}")
    print(f"\n[OK] CSV -> {mdir / f'day3_ad2_results{tag}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
