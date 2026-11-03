# -*- coding: utf-8 -*-
"""Unit tests for the metric functions in ``src/eval_core.py``.

These run on synthetic arrays only - no dataset is required.
"""
from __future__ import annotations

import numpy as np
import pytest

import eval_core as ec


# ------------------------------------------------------------------ image level
def test_image_metrics_perfect_separation():
    scores = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    m = ec.image_metrics(scores, labels)
    assert m["image_auroc"] == 1.0
    assert m["image_ap"] == pytest.approx(1.0)


def test_image_metrics_inverted_scores():
    scores = np.array([10.0, 11.0, 12.0, 1.0, 2.0, 3.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert ec.image_metrics(scores, labels)["image_auroc"] == 0.0


def test_image_metrics_all_ties_is_chance():
    """A constant score cannot rank anything: AUROC must be 0.5, not 1.0."""
    scores = np.zeros(10)
    labels = np.array([0, 1] * 5)
    assert ec.image_metrics(scores, labels)["image_auroc"] == pytest.approx(0.5)


def test_clamp_creates_ties_and_destroys_ranking():
    """Regression test for the prototype's PostProcessor pitfall.

    A monotone clamp is *not* strictly monotone: it maps many distinct scores
    onto the same constant. Ties count as 0.5 in AUROC, so the metric collapses
    even though nothing was numerically "lost".

    The damage needs overlap between the two classes - that is exactly the
    regime anomaly maps live in (background and defect pixel scores overlap
    heavily). Here a large fraction of both classes falls below the affine
    midpoint and is clipped to the same value.
    """
    rng = np.random.default_rng(0)
    labels = np.array([0] * 200 + [1] * 200)
    scores = np.concatenate([rng.normal(20, 4, 200), rng.normal(23, 4, 200)])

    raw = ec.image_metrics(scores, labels)["image_auroc"]

    # global affine + clamp, mirroring anomalib's PostProcessor when the
    # validation split holds only normal images (threshold == max score)
    lo, hi = scores.min(), scores.max()
    clamped = np.clip((scores - hi) / (hi - lo) + 0.5, 0.0, 1.0)
    after = ec.image_metrics(clamped, labels)["image_auroc"]

    n_tied = int((clamped == 0.0).sum())
    assert n_tied > len(scores) * 0.3, "expected heavy saturation at zero"
    assert len(np.unique(clamped)) < len(np.unique(scores))
    # The exact drop depends on the overlap; what matters is that it is a loss,
    # not noise. On the real pixel case the same mechanism cost 0.32 AUROC.
    assert after < raw - 0.02


# ------------------------------------------------------------------ pixel level
def test_pixel_metrics_perfect_map():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True
    amap = np.zeros((8, 8), dtype=np.float32)
    amap[mask] = 10.0

    m = ec.pixel_metrics([amap], [mask])
    assert m["pixel_auroc"] == 1.0
    assert m["n_anomalous_pixels"] == 9
    assert m["n_pixels"] == 64


def test_pixel_auroc_per_image_skips_maskless_images():
    empty = np.zeros((4, 4), dtype=bool)
    full = np.ones((4, 4), dtype=bool)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True

    amap = np.zeros((4, 4), dtype=np.float32)
    amap[0, 0] = 5.0

    # only the third image has a partially-marked mask -> only it contributes
    val = ec.pixel_auroc_per_image([amap, amap, amap], [empty, full, mask])
    assert val == pytest.approx(1.0)


def test_threshold_metrics_confusion_matrix():
    scores = np.array([1.0, 2.0, 5.0, 8.0])
    labels = np.array([0, 0, 1, 1])
    r = ec.threshold_metrics(scores, labels, thr=3.0)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 0, 0, 2)
    assert r["fpr_normal"] == 0.0
    assert r["recall_defect"] == 1.0

    r2 = ec.threshold_metrics(scores, labels, thr=9.0)
    assert (r2["tp"], r2["fp"], r2["fn"], r2["tn"]) == (0, 0, 2, 2)
    assert r2["recall_defect"] == 0.0


# ------------------------------------------------------------------ AUPRO
def test_aupro_perfect_localization_is_one():
    mask = np.zeros((16, 16), dtype=bool)
    mask[6:10, 6:10] = True
    amap = np.zeros((16, 16), dtype=np.float32)
    amap[mask] = 1.0

    score = ec.aupro([amap], [mask], fpr_limit=0.3)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_aupro_inverted_localization_is_low():
    """If the defect scores *lower* than the background, PRO stays near zero."""
    rng = np.random.default_rng(3)
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:20, 12:20] = True

    amap = rng.random((32, 32)).astype(np.float32)
    amap[mask] = 0.0  # defect deliberately ranked below everything else

    score = ec.aupro([amap], [mask], fpr_limit=0.3)
    assert not np.isnan(score)
    assert score < 0.2


def test_aupro_returns_nan_when_no_operating_point_fits_the_fpr_limit():
    """A constant score gives a single operating point; if FPR exceeds the
    limit there is nothing to integrate, and NaN is the honest answer."""
    mask = np.ones((16, 16), dtype=bool)
    mask[0, 0] = False
    amap = np.zeros((16, 16), dtype=np.float32)

    assert np.isnan(ec.aupro([amap], [mask], fpr_limit=0.3))


def test_aupro_returns_nan_without_regions():
    empty = np.zeros((8, 8), dtype=bool)
    amap = np.zeros((8, 8), dtype=np.float32)
    assert np.isnan(ec.aupro([amap], [empty]))


# ------------------------------------------------------------------ determinism
def test_setup_determinism_reports_effective_flags():
    info = ec.setup_determinism(seed=1234)
    assert info["seed"] == 1234
    assert info["cudnn_deterministic"] is True
    assert info["allow_tf32_matmul"] is False
