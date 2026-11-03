# -*- coding: utf-8 -*-
"""Day 2 光照扰动算子（确定性、无随机性）。

设计约定
--------
1. **作用域**：只作用于 8-bit sRGB 原始图像（840x840），在 Resize 到 256 之前。
   这样扰动模拟的是"相机拍到的光变"，而不是在特征尺度上做手脚。
2. **颜色空间**：直接在 sRGB 非线性值上做逐通道乘法/幂运算（业界常用近似，
   非物理光照模型）。不做线性化，因为论文关心的是"输入像素扰动"，不是成像物理。
3. **单步独立**：每种条件单独作用在原始图上，绝不叠加。
4. **确定性**：无随机种子、无随机参数；局部阴影的光照场是全图固定解析式。
5. **量化回 8-bit**：扰动后 round 回 uint8，再交给下游管线，与真实相机链路一致。

数学定义（x 为 [0,1] 的 sRGB 像素值，k/γ/A 为给定常数）
--------------------------------------------------------
- original        : y = x
- brightness_low  : y = clip(0.8 · x, 0, 1)
- brightness_high : y = clip(1.2 · x, 0, 1)
- gamma_low       : y = x^0.7     （γ<1 → 提亮中间调）
- gamma_high      : y = x^1.4     （γ>1 → 压暗中间调）
- local_shadow    : y = x · L(p)

  其中 L(p) = 1 − A · exp(−‖p − c‖² / (2σ²))，A=0.50，σ=0.30·H，
  c=(0.35H, 0.40W) 固定于全图统一位置（与图像内容无关，保证跨图可比）。
  L 是平滑高斯衰减场，范围 [0.50, 1.0]，**不是**叠加黑色矩形。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

IMG_SIZE = 840  # MVTec AD tile 原始分辨率


@dataclass(frozen=True)
class Condition:
    """一种光照条件的元数据。"""

    name: str
    family: str  # 用于分组统计
    description: str
    formula: str


CONDITIONS: list[Condition] = [
    Condition("original", "baseline", "无扰动", "y = x"),
    Condition("brightness_low", "brightness", "全局亮度系数 0.8", "y = clip(0.8·x, 0, 1)"),
    Condition("brightness_high", "brightness", "全局亮度系数 1.2", "y = clip(1.2·x, 0, 1)"),
    Condition("gamma_low", "gamma", "Gamma 0.7（提亮中间调）", "y = x^0.7"),
    Condition("gamma_high", "gamma", "Gamma 1.4（压暗中间调）", "y = x^1.4"),
    Condition(
        "local_shadow",
        "spatial",
        "平滑高斯局部阴影场",
        "y = x·L, L = 1 − 0.50·exp(−‖p−c‖²/(2σ²)), σ=0.30H, c=(0.35H,0.40W)",
    ),
]

CONDITION_NAMES = [c.name for c in CONDITIONS]


def shadow_field(h: int = IMG_SIZE, w: int = IMG_SIZE, amplitude: float = 0.50, sigma_ratio: float = 0.30,
                 center: tuple[float, float] = (0.35, 0.40)) -> np.ndarray:
    """生成固定的平滑光照衰减场 L∈[1-amplitude, 1]，形状 (h,w)。

    与图像内容无关，因此同一位置在 117 张图上完全一致，避免"每图随机阴影"
    带来的不可控变量。
    """
    cy, cx = center[0] * h, center[1] * w
    sigma = sigma_ratio * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    field = 1.0 - amplitude * np.exp(-d2 / (2.0 * sigma**2))
    return field.astype(np.float32)


_FIELD_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _get_field(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key not in _FIELD_CACHE:
        _FIELD_CACHE[key] = shadow_field(h, w)
    return _FIELD_CACHE[key]


def apply_condition(img_u8: torch.Tensor, condition: str) -> tuple[torch.Tensor, dict]:
    """对单张 uint8 图像 (3,H,W) 施加指定光照条件。

    返回 (perturbed_uint8, stats)。stats 记录截断比例等可审计信息。
    """
    if condition not in CONDITION_NAMES:
        msg = f"未知条件: {condition}，可选 {CONDITION_NAMES}"
        raise ValueError(msg)

    x = img_u8.to(torch.float32) / 255.0

    if condition == "original":
        y = x
    elif condition == "brightness_low":
        y = x * 0.8
    elif condition == "brightness_high":
        y = x * 1.2
    elif condition == "gamma_low":
        y = torch.pow(x.clamp_min(0.0), 0.7)
    elif condition == "gamma_high":
        y = torch.pow(x.clamp_min(0.0), 1.4)
    elif condition == "local_shadow":
        field = torch.from_numpy(_get_field(x.shape[-2], x.shape[-1])).unsqueeze(0)
        y = x * field
    else:  # pragma: no cover
        raise AssertionError

    y = y.clamp(0.0, 1.0)
    out = torch.round(y * 255.0).to(torch.uint8)

    stats = {
        "clip_low_ratio": float((out == 0).float().mean()),
        "clip_high_ratio": float((out == 255).float().mean()),
        "mean_intensity_before": float(x.mean()),
        "mean_intensity_after": float(out.to(torch.float32).div(255.0).mean()),
    }
    stats["mean_intensity_delta"] = stats["mean_intensity_after"] - stats["mean_intensity_before"]
    return out, stats


def condition_metadata() -> list[dict]:
    """导出条件定义，供报告与 CSV 引用。"""
    return [
        {"name": c.name, "family": c.family, "description": c.description, "formula": c.formula}
        for c in CONDITIONS
    ]
