# -*- coding: utf-8 -*-
"""Unit tests for the image-level aggregation rules in ``src/day4_aggregation.py``.

The aggregator is the part of the pipeline that was held fixed while the model,
input geometry and anomaly maps were frozen, so its numerics matter: a silent
change here would invalidate every conclusion in ``docs/reports/Day4_Report.md``.
"""
from __future__ import annotations

import numpy as np
import pytest

from day4_aggregation import RULES, aggregate, hash_maps


def test_rules_list_is_stable():
    assert RULES == ["max", "top1", "top5", "p99", "gmean"]


def test_max_and_mean_on_known_array():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert aggregate(a, "max") == pytest.approx(4.0)
    assert aggregate(a, "gmean") == pytest.approx(2.5)
    assert aggregate(a, "p99") == pytest.approx(float(np.quantile(a, 0.99)))


def test_topk_matches_manual_topk_mean():
    # 100 values, one dominant spike: Top-1% must be a single element
    a = np.zeros((10, 10), dtype=np.float32)
    a[0, 0] = 9.0
    assert aggregate(a, "top1") == pytest.approx(9.0)

    # 1000 values: Top-1% averaging over 10 elements
    b = np.arange(1000, dtype=np.float32).reshape(10, 100)
    expected = b.reshape(-1)[-10:].mean()
    assert aggregate(b, "top1") == pytest.approx(float(expected))


def test_topk_fraction_rounds_to_at_least_one_element():
    tiny = np.array([[1.0, 2.0]], dtype=np.float32)  # 2 elements
    assert aggregate(tiny, "top1") == pytest.approx(2.0)


def test_topk_degrades_to_mean_when_k_covers_everything():
    one = np.array([[5.0]], dtype=np.float32)
    for rule in ("top1", "top5"):
        assert aggregate(one, rule) == pytest.approx(5.0)


def test_unknown_rule_raises():
    with pytest.raises(ValueError, match="未知规则"):
        aggregate(np.zeros((2, 2)), "median")


def test_aggregation_does_not_mutate_the_map():
    """The controlled experiment relies on maps being immutable."""
    a = np.random.default_rng(0).normal(size=(16, 16)).astype(np.float32)
    before = a.copy()
    for rule in RULES:
        aggregate(a, rule)
    assert np.array_equal(a, before)


def test_hash_is_deterministic_and_factor_ordering_free():
    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    h1 = hash_maps(a)
    h2 = hash_maps(a.copy())
    assert h1 == h2
    assert len(h1) == 16

    # a non-contiguous view with identical contents must hash the same
    view = a.reshape(8, 8).T.T
    assert hash_maps(view) == h1
