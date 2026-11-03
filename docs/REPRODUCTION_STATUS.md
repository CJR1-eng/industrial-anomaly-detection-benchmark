# Reproduction status

What was actually re-executed from **this** directory while packaging the
repository, and what is only reported from the original runs. The point of this
file is that you can tell the difference without reading git history.

Verified on Windows 11 / Python 3.12.7 / torch 2.14.1+cu126 / anomalib 2.7.0,
single RTX 3050 Laptop 4 GB, data referenced through `IDAD_DATA_ROOT`.

## Re-executed here (results below are from these runs)

| Check | Command | Result |
|---|---|---|
| Unit tests | `python -m pytest tests -q` | **57 passed** in ~23 s |
| Every module imports | included in the tests | 17 modules |
| Every CLI answers `--help` | included in the tests | 9 scripts, exit 0 |
| Missing data gives an actionable message | included in the tests | non-zero exit, names the category and the download URL |
| Dataset integrity, MVTec AD tile | `python src/dataset_check.py --category tile` | **passes**: 230 train / 117 test (33 + 84) / 84 masks, all 840×840, mask pairing complete, content-level leakage 0 |
| Dataset integrity, MVTec AD 2 sheet_metal | `python src/ad2_check.py` | **passes**: 644 files indexed; train 137 / validation 19 / test_public 24 + 90 (+90 masks); test_private 142 / test_private_mixed 142 (no GT); all sampled images 4224×1056 grayscale; 6 lighting variants visible in file names |
| Tile baseline + 6 synthetic lighting conditions | `python src/run_illumination.py` | **329 s**, and **bit-identical to the archived original run** |
| Failure analysis + figures | `python src/day2_analysis.py` | **11 figures** written to `figures/day2_illumination/`, exit 0 |

### The reproducibility check that matters

A fresh process refitted PaDiM from scratch and re-ran all six conditions. Compared
against the archived table in `docs/results/tile_synthetic_illumination.csv`:

| Metric | max absolute difference |
|---|---|
| Image AUROC | `0.000e+00` |
| Image AP | `0.000e+00` |
| Pixel AUROC | `0.000e+00` |
| Pixel AP | `0.000e+00` |
| AUPRO (256) | `0.000e+00` |
| FPR / Recall | `0.000e+00` / `0.000e+00` |

This was done **twice**, in two separate processes, and both matched the archived
values exactly. Six rows identical to six decimals, including the baseline:

```
original         Image AUROC 0.942641   Pixel AUROC 0.904702   AUPRO 0.767547   FPR 0.0303   Recall 0.5833
gamma_low        Image AUROC 0.924964   Pixel AUROC 0.904630   AUPRO 0.774491   FPR 0.0303   Recall 0.5714
local_shadow     Image AUROC 0.931097   Pixel AUROC 0.902498   AUPRO 0.760334   FPR 0.0303   Recall 0.6190
```

Runtime note: 329 s here versus ~44 s per condition in the original run — the
difference is the first-run weight download and CUDA warm-up, not a behavioural
change (the numbers are identical).

## Reported from the original runs, **not** re-executed here

These need the MVTec AD 2 extraction plus ~8 minutes of GPU time, and the code path
was not modified during packaging. The numbers in `docs/results/ad2_*.csv` and
`docs/results/aggregation_*.csv` are therefore carried over, not re-measured:

| Experiment | Why not re-run | Residual risk |
|---|---|---|
| `run_ad2.py` squash / aspect | 1.5 GB extraction + AD2 GPU run | low: `run_ad2.py` was edited only in `find_category` (error message) and the path constant; scoring code untouched |
| `day4_aggregation.py` | depends on the AD2 map cache | low: `aggregate()` is covered by unit tests against hand-computed values |
| `day3_analysis.py`, `day4_figures.py` | plotting only | low |
| `run_padim.py` (Day 1) | deliberately not: its evaluation path is known-broken and documented as such | none — its numbers are not used |
| `tools/ablation_full_train.py` → `src/ablation_full_train.py` | 3 fits × 2 protocols | none — it only supports a Day 1 remark |

If you want to close this gap, run:

```bash
python src/ad2_check.py --archive datasets/raw/ad2/sheet_metal.tar.gz
python src/run_ad2.py --category sheet_metal --mode squash
python src/run_ad2.py --category sheet_metal --mode aspect --tag aspect
python src/day4_aggregation.py
```

and diff the resulting CSVs against `docs/results/`.

## Never verified, by anyone

- **Other hardware**, including CPU-only runtimes and non-NVIDIA GPUs.
- **Other categories** — MVTec AD 2 `fabric` was never downloaded.
- **Other models** — PatchCore, `wide_resnet50_2`.
- **Seed sensitivity** — only seed 42 was swept.
- **Private test splits** — `test_private` / `test_private_mixed` have no public
  ground truth and were never scored.
- **The `environment.yml` conda path** — written for convenience, untested.
