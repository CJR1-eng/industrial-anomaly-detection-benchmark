# -*- coding: utf-8 -*-
"""Smoke tests: every module imports, every CLI answers ``--help``.

These are the checks a reader can trust before downloading multi-GB datasets.
Anything requiring the actual MVTec data is deliberately excluded and listed as
unverified in ``docs/REPRODUCTION_STATUS.md``.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

BIBLIOTECA = [
    "paths",
    "eval_core",
    "ad2_eval",
    "perturbations",
    "dataset_check",
    "ad2_check",
    "env_check",
    "audit_postprocess",
    "run_illumination",
    "day2_analysis",
    "run_ad2",
    "day3_analysis",
    "day4_aggregation",
    "day4_figures",
    "day5_audit",
    "day4_tile_check",
    "ablation_full_train",
]

CLI_SCRIPTS = [
    "src/dataset_check.py",
    "src/env_check.py",
    "src/run_padim.py",
    "src/run_ad2.py",
    "src/run_illumination.py",
    "src/ad2_check.py",
    "scripts/download_dataset.py",
    "scripts/prepare_dataset.py",
    "scripts/compare_runs.py",
]


@pytest.mark.parametrize("name", BIBLIOTECA)
def test_module_imports(name):
    assert importlib.import_module(name) is not None


@pytest.mark.parametrize("script", CLI_SCRIPTS)
def test_cli_help_exits_zero(script):
    out = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "usage" in out.stdout.lower()


def test_missing_dataset_message_is_not_a_bare_traceback():
    """Running the tile check with a bogus data root must explain the fix."""
    out = subprocess.run(
        [sys.executable, "src/dataset_check.py", "--data-root", "no-such-dir", "--category", "tile"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    blob = out.stdout + out.stderr
    assert out.returncode != 0
    # must name the category and point at the official source rather than crash silently
    assert "tile" in blob.lower()
    assert "mvtec" in blob.lower()


def test_repo_declares_its_dependencies():
    req = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for pkg in ("torch", "anomalib", "scikit-learn", "numpy"):
        assert pkg in req.lower()
