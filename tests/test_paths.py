# -*- coding: utf-8 -*-
"""Tests for dataset path resolution and the missing-data error message.

MVTec AD and MVTec AD 2 are not redistributed with this repository, so "the
user cloned it and has no data yet" is the *normal* first run. It has to fail
with a readable message, not a traceback from inside a dataloader.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import paths

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_roots_point_inside_the_repo():
    assert paths.mvtec_ad_root() == REPO_ROOT / "datasets" / "raw" / "mvtec_ad"
    assert paths.ad2_root() == REPO_ROOT / "datasets" / "raw" / "ad2"


def test_env_override_is_honoured(monkeypatch, tmp_path):
    monkeypatch.setenv("IDAD_DATA_ROOT", str(tmp_path))
    assert paths.dataset_root() == tmp_path.resolve()
    assert paths.mvtec_ad_root() == tmp_path.resolve() / "mvtec_ad"


def test_results_and_figures_are_overridable(monkeypatch, tmp_path):
    monkeypatch.setenv("IDAD_RESULTS_DIR", str(tmp_path / "r"))
    monkeypatch.setenv("IDAD_FIGURES_DIR", str(tmp_path / "f"))
    assert paths.results_dir() == (tmp_path / "r").resolve()
    assert paths.figures_dir() == (tmp_path / "f").resolve()


def test_require_category_returns_existing_dir(tmp_path):
    cat = tmp_path / "tile"
    cat.mkdir()
    assert paths.require_category(tmp_path, "tile", "MVTec AD", paths.MVTEC_AD_URL) == cat


def test_missing_category_error_is_actionable(tmp_path):
    (tmp_path / "bottle").mkdir()  # something exists, but not the asked-for category
    with pytest.raises(paths.DatasetNotFound) as excinfo:
        paths.require_category(tmp_path, "tile", "MVTec AD", paths.MVTEC_AD_URL)

    msg = str(excinfo.value)
    assert "tile" in msg
    assert paths.MVTEC_AD_URL in msg          # tells the user where to download
    assert "IDAD_DATA_ROOT" in msg            # tells the user how to relocate it
    assert "bottle" in msg                    # shows what *is* there


def test_dataset_not_found_subclasses_file_not_found():
    """Existing ``except FileNotFoundError`` handlers keep working."""
    assert issubclass(paths.DatasetNotFound, FileNotFoundError)


def test_require_artifacts_passes_when_everything_exists(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("x", encoding="utf-8")
    paths.require_artifacts({f: "python src/whatever.py"})  # no raise


def test_require_artifacts_names_the_producing_command(tmp_path):
    missing = tmp_path / "day4_aggregation.csv"
    with pytest.raises(FileNotFoundError) as excinfo:
        paths.require_artifacts({missing: "python src/day4_aggregation.py"})

    msg = str(excinfo.value)
    assert "day4_aggregation.csv" in msg
    assert "python src/day4_aggregation.py" in msg


def test_data_root_env_is_read_at_import_time(tmp_path):
    """``eval_core.DATA_ROOT`` is a module constant - verify it tracks the env."""
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "import eval_core;"
        "print(eval_core.DATA_ROOT)"
    )
    env = dict(os.environ)
    env["IDAD_DATA_ROOT"] = str(tmp_path)
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=300,
    )
    assert str(tmp_path) in out.stdout
