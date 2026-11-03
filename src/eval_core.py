# -*- coding: utf-8 -*-
"""Day 2 统一评估核心：确定性设置、数据管线对齐、冻结模型推理、双尺度指标。

为什么要自己写推理循环
----------------------
anomalib 2.x 的评估链路里有两个会干扰科研结论的行为：
1. 数据集层只做 uint8 -> [0,1]；`Resize(256)+Normalize` 由模型 PreProcessor 注入
   数据集 augmentations，**在 fit 之前调用 setup() 拿到的样本是 840×840**。
2. `engine.predict()` 返回的 anomaly_map 已被 PostProcessor 做**全局仿射+clamp**，
   分数跨条件不可比（Day 1 已实证）。

所以本模块显式重建管线：
    uint8 图像 -> [扰动] -> Resize(256,bilinear,antialias) -> /255 -> Normalize(ImageNet)
与 anomalib 的 PreProcessor 逐元素一致（由 audit_postprocess.py 校验）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paths import MVTEC_AD_URL, mvtec_ad_root, require_category  # noqa: E402

DATA_ROOT = mvtec_ad_root()
CATEGORY = "tile"
IMG_SIZE = 840
MODEL_SIZE = 256

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 与 anomalib PreProcessor 默认 transform 完全一致
TRANSFORM = T.Compose(
    [
        T.Resize([MODEL_SIZE, MODEL_SIZE], interpolation=InterpolationMode.BILINEAR, antialias=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# ------------------------------------------------------------------ determinism
def setup_determinism(seed: int = 42) -> dict:
    """固定全部随机源，并返回实际生效的配置（写进报告）。"""
    import random

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001
        pass
    return {
        "seed": seed,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": True,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


# ------------------------------------------------------------------ data access
def load_image_u8(path: str | Path) -> torch.Tensor:
    """读 8-bit RGB 原图，返回 (3,H,W) uint8，不做任何缩放。"""
    from torchvision.io import ImageReadMode, read_image

    return read_image(str(path), mode=ImageReadMode.RGB)


def load_mask_bool(path: str | Path) -> np.ndarray:
    """读原始 GT 掩码，返回 (H,W) bool，保持原始分辨率与二值性。"""
    from torchvision.io import ImageReadMode, read_image

    m = read_image(str(path), mode=ImageReadMode.GRAY)
    return (m.squeeze(0).numpy() > 0)


def mask_to_size(mask: np.ndarray, size: int, interpolation: InterpolationMode) -> np.ndarray:
    """把二值掩码缩放，再按 >0.5 二值化。

    interpolation=NEAREST  -> 严格保持标签（任务要求的做法，P256 使用）
    interpolation=BILINEAR+antialias -> anomalib 数据集默认行为（Day 1 的实际做法）
    """
    t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = T.Resize([size, size], interpolation=interpolation, antialias=(interpolation != InterpolationMode.NEAREST))(t)
    return (t.squeeze().numpy() > 0.5)


def to_model_input(img_u8: torch.Tensor) -> torch.Tensor:
    """uint8 (3,H,W) -> 模型输入 (3,256,256) float32，与 anomalib PreProcessor 等价。"""
    x = img_u8.to(torch.float32).div(255.0)
    return TRANSFORM(x)


# ------------------------------------------------------------------ model
def build_datamodule(seed: int = 42, batch_size: int = 8):
    """构造与 Day 1 完全相同的划分：184 拟合 / 46 验证 / 117 测试。"""
    from anomalib.data import MVTecAD

    require_category(DATA_ROOT, CATEGORY, "MVTec AD", MVTEC_AD_URL)
    return MVTecAD(
        root=str(DATA_ROOT),
        category=CATEGORY,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,  # 单进程，保证顺序确定
        val_split_mode="from_train",
        val_split_ratio=0.2,
        test_split_mode="from_dir",
        seed=seed,
    )


def fit_padim(seed: int = 42, device: str = "cuda", n_features: int = 100):
    """训练（拟合高斯）PaDiM 并返回冻结模型 + datamodule + 引擎。"""
    from anomalib.engine import Engine
    from anomalib.models.image.padim import Padim

    dm = build_datamodule(seed=seed)
    model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=True,
        n_features=n_features,
    )
    engine = Engine(
        default_root_dir=str(PROJECT_ROOT / "results" / "runs" / "day2"),
        max_epochs=1,
        accelerator=device if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
    )
    engine.fit(model=model, datamodule=dm)
    model.eval()
    model.to(device)
    return model, dm, engine


def save_split_manifest(dm, seed: int, out_path: Path) -> dict:
    """保存 train/val/test 的文件清单，作为可复现性凭据。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["split,label,image_path,mask_path"]
    counts = {}
    for split, subset in (("train", "train_data"), ("val", "val_data"), ("test", "test_data")):
        data = getattr(dm, subset)
        df = data.samples
        counts[split] = int(len(df))
        for _, r in df.iterrows():
            rows.append(f"{split},{r['label']},{r['image_path']},{r['mask_path']}")
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"seed": seed, "counts": counts, "manifest": str(out_path)}


# ------------------------------------------------------------------ inference
@torch.no_grad()
def infer_tensors(model, tensors: list[torch.Tensor], device: str = "cuda", batch_size: int = 8):
    """对一组 (3,256,256) 张量做前向，返回原始马氏距离得分与热力图。"""
    scores, maps = [], []
    for i in range(0, len(tensors), batch_size):
        chunk = torch.stack(tensors[i : i + batch_size]).to(device)
        out = model.model(chunk)
        s = out.pred_score.detach().float().cpu().numpy().reshape(-1)
        m = out.anomaly_map.detach().float().cpu().numpy()
        if m.ndim == 4:
            m = m[:, 0]
        scores.extend(s.tolist())
        maps.extend(list(m))
    return np.asarray(scores, dtype=np.float64), np.asarray(maps, dtype=np.float32)


def upsample_map(amap: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """异常分数图 -> 原分辨率，双线性连续插值（不改变排序，仅插值）。"""
    t = torch.from_numpy(amap).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


# ------------------------------------------------------------------ metrics
def image_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_ap": float(average_precision_score(labels, scores)),
    }


def pixel_metrics(maps: list[np.ndarray], masks: list[np.ndarray]) -> dict:
    """全局池化像素级指标（跨图拼接后统一计算，MVTec 标准口径）。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.concatenate([m.reshape(-1) for m in masks]).astype(np.uint8)
    s = np.concatenate([a.reshape(-1) for a in maps]).astype(np.float32)
    return {
        "pixel_auroc": float(roc_auc_score(y, s)),
        "pixel_ap": float(average_precision_score(y, s)),
        "n_pixels": int(y.size),
        "n_anomalous_pixels": int(y.sum()),
    }


def pixel_auroc_per_image(maps: list[np.ndarray], masks: list[np.ndarray]) -> float:
    """逐图 pixel AUROC 均值（只统计含缺陷的图）。"""
    from sklearn.metrics import roc_auc_score

    vals = []
    for a, m in zip(maps, masks):
        y = m.reshape(-1).astype(np.uint8)
        if y.sum() == 0 or y.sum() == y.size:
            continue
        vals.append(roc_auc_score(y, a.reshape(-1)))
    return float(np.mean(vals)) if vals else float("nan")


def aupro(maps: list[np.ndarray], masks: list[np.ndarray], fpr_limit: float = 0.3, n_thresholds: int = 100) -> float:
    """AUPRO：逐连通域重叠率随 FPR 的曲线下面积（截断到 FPR<=limit）。

    实现要点：连通域只算一次；每个阈值只做向量化比较，避免逐图循环开销。
    """
    from scipy import ndimage

    regions: list[tuple[int, np.ndarray]] = []
    for i, m in enumerate(masks):
        if m.sum() == 0:
            continue
        lab, n = ndimage.label(m)
        for k in range(1, n + 1):
            idx = np.where((lab == k).reshape(-1))[0]
            if idx.size:
                regions.append((i, idx))
    if not regions:
        return float("nan")

    flat = [a.reshape(-1) for a in maps]
    normal_total = sum(int((~m).sum()) for m in masks)
    if normal_total == 0:
        return float("nan")

    all_scores = np.concatenate(flat)
    qs = np.unique(np.quantile(all_scores, np.linspace(0.5, 1.0, n_thresholds)))

    pro_vals, fpr_vals = [], []
    for thr in qs:
        overlaps = [float((flat[i][idx] >= thr).mean()) for i, idx in regions]
        fp = sum(int((flat[i][(~masks[i].reshape(-1))] >= thr).sum()) for i in range(len(maps)))
        pro_vals.append(float(np.mean(overlaps)))
        fpr_vals.append(fp / normal_total)

    pro = np.asarray(pro_vals)
    fpr = np.asarray(fpr_vals)
    order = np.argsort(fpr)
    fpr, pro = fpr[order], pro[order]
    keep = fpr <= fpr_limit
    if keep.sum() < 2:
        return float("nan")
    f, p = fpr[keep], pro[keep]
    f = np.concatenate([[0.0], f])
    p = np.concatenate([[p[0]], p])
    if f[-1] < fpr_limit:
        p = np.concatenate([p, [np.interp(fpr_limit, fpr, pro)]])
        f = np.concatenate([f, [fpr_limit]])
    return float(np.trapezoid(p, f) / fpr_limit)


def threshold_metrics(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    """固定阈值下的误报率/召回率（阈值必须来自验证集）。"""
    pred = scores >= thr
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    return {
        "threshold": float(thr),
        "fpr_normal": fp / max(fp + tn, 1),
        "recall_defect": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
