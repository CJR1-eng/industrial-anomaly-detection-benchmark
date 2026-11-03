# -*- coding: utf-8 -*-
"""Dataset path resolution.

Neither MVTec AD nor MVTec AD 2 is redistributed with this repository
(both are CC BY-NC-SA 4.0 and must be downloaded from the official source).
Every dataset location can therefore be overridden without editing code:

    IDAD_DATA_ROOT   directory that contains ``mvtec_ad/`` and ``ad2/``
                     default: ``<repo>/datasets/raw``
    IDAD_RESULTS_DIR output root for metrics and logs
                     default: ``<repo>/results``
    IDAD_FIGURES_DIR output root for figures
                     default: ``<repo>/figures``

A missing dataset raises :class:`DatasetNotFound` with the expected layout and
the download URL, instead of a bare ``FileNotFoundError`` deep inside a loader.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MVTEC_AD_URL = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"
MVTEC_AD2_URL = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2"


class DatasetNotFound(FileNotFoundError):
    """Raised when an expected dataset directory is absent."""


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def dataset_root() -> Path:
    """Root that should contain ``mvtec_ad/`` and/or ``ad2/``."""
    return _env_path("IDAD_DATA_ROOT", PROJECT_ROOT / "datasets" / "raw")


def results_dir() -> Path:
    return _env_path("IDAD_RESULTS_DIR", PROJECT_ROOT / "results")


def figures_dir() -> Path:
    return _env_path("IDAD_FIGURES_DIR", PROJECT_ROOT / "figures")


def mvtec_ad_root() -> Path:
    """Root passed to ``anomalib.data.MVTecAD`` (must contain ``<category>/``)."""
    return dataset_root() / "mvtec_ad"


def ad2_root() -> Path:
    """Extracted MVTec AD 2 root (must contain ``<category>/train/good``)."""
    return dataset_root() / "ad2"


def require_category(root: Path, category: str, dataset: str, url: str) -> Path:
    """Return ``root/category`` or raise a human-readable error."""
    cat_dir = root / category
    if cat_dir.is_dir():
        return cat_dir

    hint = (
        f"\n\n{dataset} / '{category}' not found.\n"
        f"  expected: {cat_dir}\n"
        f"  data root: {root}\n\n"
        f"Download the dataset from the official source and extract it there:\n"
        f"  {url}\n\n"
        f"Or point the code at an existing copy:\n"
        f"  set IDAD_DATA_ROOT=<dir containing mvtec_ad/ or ad2/>   (Windows)\n"
        f"  export IDAD_DATA_ROOT=<dir containing mvtec_ad/ or ad2/>  (Linux/macOS)\n"
    )
    if root.is_dir():
        existing = sorted(p.name for p in root.iterdir() if p.is_dir())
        hint += f"\nDirectories currently present under the data root: {existing or 'none'}\n"

    raise DatasetNotFound(hint)


def require_artifacts(items: dict[str | Path, str]) -> None:
    """Fail fast when cached results from an earlier stage are missing.

    ``items`` maps a required path to the command that produces it. The analysis
    scripts depend on intermediate ``.npz`` / ``.csv`` files that are not shipped
    (they are regenerated), so a fresh clone hits this immediately. One readable
    message beats a ``FileNotFoundError`` from inside ``np.load``.
    """
    missing = [(Path(p), cmd) for p, cmd in items.items() if not Path(p).exists()]
    if not missing:
        return

    lines = ["\n\nRequired intermediate results are missing.\n"]
    for path, cmd in missing:
        lines.append(f"  missing: {path}\n")
        lines.append(f"  produce: {cmd}\n\n")
    lines.append(
        "These files are generated, not shipped: run the producing command(s) "
        "above, in order, from the repository root.\n"
    )
    raise FileNotFoundError("".join(lines))
