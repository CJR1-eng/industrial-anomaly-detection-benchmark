# Experimental protocol

What was actually run, in enough detail that a mismatch would be visible.
Anything not stated here (optimizer, augmentation, hyper-parameter search) does
not exist — PaDiM is fitted in closed form and nothing is trained.

Machine: Windows 11, Python 3.12.7, PyTorch 2.14.1+cu126, anomalib 2.7.0,
Lightning 2.6.6, single RTX 3050 Laptop (4 GB). Peak GPU memory for PaDiM fitting:
1097.7 MB.

---

## 1. Data splits

### MVTec AD / tile (used for the baseline and the synthetic-lighting study)

| Split | Count | Source | Role |
|---|---|---|---|
| fit | 184 | `train/good` minus 20% | fits the Gaussian (mean + covariance) |
| validation | 46 | 20% held out of `train/good` | **threshold only** |
| test | 117 (33 normal, 84 defect) | official `test/` | all reported metrics |

MVTec AD has no official validation split. Anomalib's default
(`val_split_mode=SAME_AS_TEST`) makes validation *equal to test*; using it to pick
a threshold is fitting on the test set. We used `val_split_mode=from_train,
val_split_ratio=0.2`, so the test split is never seen during configuration.

Cost of doing this honestly: fitting on 184 instead of 230 images lowers Image
AUROC from 0.926 to 0.855 under the Day 1 (unnormalized) protocol — the ablation is
in `docs/reports/Day1_Report.md`. It is a real cost, reported rather than hidden.

### MVTec AD 2 / sheet_metal (the real-lighting study)

| Split | Count | Lighting | Role |
|---|---|---|---|
| train | 137 | all `regular` | fits the Gaussian |
| validation | 19 | all `regular` | threshold only |
| test_public | 24 normal + 90 defect | 6 conditions | all reported metrics |
| test_private / test_private_mixed | 142 / 142 | mixed | **never used** (no public GT) |

Six lighting conditions per scene: `regular`, `overexposed`, `underexposed`,
`shift_1`, `shift_2`, `shift_3`. The condition is encoded in the file name, so
each scene yields six images that share a numeric prefix and each has its own
ground truth. That is what makes pairing legitimate — the pairing is not inferred
from filename similarity.

Consequence: 114 test images but only **19 independent scenes**, and only 4 normal
images per condition. Every confidence interval in this repository is computed
with the scene as the resampling unit for that reason.

---

## 2. Model

```python
Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"],
      pre_trained=True, n_features=100)
```

- Backbone frozen; weights are the upstream timm `resnet18.a1_in1k` ImageNet
  checkpoint, downloaded on first use.
- `n_features=100` is the value given in the PaDiM paper for ResNet-18 (random
  channel selection, seeded).
- Fitting is closed-form (per-position multivariate Gaussian, shared covariance,
  `eps=0.01` inside anomalib's `MultiVariateGaussian`). There is no optimizer, no
  epoch, no augmentation, no early stopping, and no hyper-parameter search.
- `precision=32`. PaDiM inverts a covariance matrix; fp16 was not attempted.

### Input geometry (MVTec AD 2 only)

Two fixed configurations, both at 256 px on the short side:

| Name | Transform | Score-map size | Grid cells |
|---|---|---|---|
| `squash` | resize to 256×256 | 256×256 | 4096 |
| `aspect` | resize short side to 256, keep ratio | 256×1024 | 16384 |

The official images are 4224×1056 (4:1). `squash` therefore compresses by 16.5×
horizontally; it is kept as a deliberately flawed debug baseline, and the aspect
configuration is the geometric control. Neither preserves the original scale, so
neither is an argument about resolution.

---

## 3. Preprocessing

Rebuilt explicitly in `src/eval_core.py` so that it is identical to, but
independent of, anomalib's `PreProcessor`:

```
uint8 image
  -> [optional perturbation, before any resizing]
  -> float / 255
  -> Resize((256, 256), bilinear, antialias=True)     # or aspect variant
  -> Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
```

Perturbations are applied to the **uint8 image before resizing** so that the
comparison is against the same spatial content, and so that clipping is measured
in the original 8-bit domain.

---

## 4. Anomaly score

The image score is the raw Mahalanobis distance map aggregated to a scalar; the
pixel score is the distance map itself.

```
map(x, y) = min_k  (e(x,y) - mu_k)^T  Sigma^-1  (e(x,y) - mu_k)
```

with `e` the concatenated layer1–3 features at that position, `mu_k` / `Sigma`
estimated from the fit split.

**Aggregation rules** (all pre-existing, none claimed as novel):

| Rule | Definition |
|---|---|
| `max` | maximum over all positions |
| `top1` | mean of the largest 1% of positions |
| `top5` | mean of the largest 5% of positions |
| `p99` | 99th percentile |
| `gmean` | mean over all positions (negative control) |

The aggregation study caches the maps to `.npz` first and asserts a SHA-256 hash of
the array is unchanged across rules — the "only the rule changed" claim is enforced
in code, not by convention.

---

## 5. Thresholds

`threshold = mean + 3σ` over the **raw scores of the validation normal images**.
Computed once per configuration and applied unchanged to every test condition;
nothing is re-tuned per condition or per lighting type.

Known weakness, stated in `docs/reports/Day2_Errata.md` (E5): the normal score
distribution is right-skewed, so `mean+3σ` sits *inside* the normal range and is not
a 3σ confidence bound. On MVTec AD tile it landed at 55.48 while the maximum normal
score was 64.60. A validation quantile (`p95`/`p99`) would behave better; we kept
`mean+3σ` because it was fixed in advance and is trivially reproducible.

Because of this, threshold-dependent numbers (FPR, recall, F1) are reported
*alongside* threshold-free numbers (AUROC, AP, AUPRO) and never as the headline.

---

## 6. Metrics

| Metric | Definition | Notes |
|---|---|---|
| Image AUROC / AP | sklearn, over the 117 (or 114) test images | threshold-free |
| Pixel AUROC / AP | sklearn, all pixels of all images pooled | MVTec convention |
| Pixel AUROC (per image) | mean of per-image AUROC, defect images only | insensitive to image size |
| AUPRO | per-connected-region overlap vs FPR, truncated at FPR ≤ 0.3, 100 thresholds | slow; `src/ad2_eval.py` has a `searchsorted` version for large images |
| FPR / Recall | at the validation threshold | report both, never F1 alone |

Pixel metrics are computed at the model output grid for MVTec AD tile (256×256,
with a 840×840 comparison showing Δ ≤ 0.0002), and at long side ≤ 1024 for MVTec
AD 2 (the native 4224×1056 evaluation needs 3.79 GB for one int64 array in
sklearn and OOMs on this machine).

**Every pixel-level number below must be read with its mask rasterization**, per
pitfall 3. In this repository: MVTec AD numbers use `nearest` masks at 256 and
originals at 840; MVTec AD 2 numbers use masks rasterized from the original GT to
the score-map grid.

---

## 7. Reproducibility

`src/eval_core.py:setup_determinism` sets:

```
CUBLAS_WORKSPACE_CONFIG=:4096:8   PYTHONHASHSEED=seed
random / numpy / torch / torch.cuda  seeds = 42
cudnn.deterministic = True        cudnn.benchmark = False
matmul.allow_tf32 = False         cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
num_workers = 0
```

Verified: two independent full runs (separate processes, refit from scratch) over
6 conditions × 117 images produced **bit-identical per-image scores** —
`max|Δscore| = 0.000e+00` for every condition, all metric deltas `0.00e+00`
(`scripts/compare_runs.py`, archived as `results/reproducibility.json` in the
original project).

The 0.01 AUROC spread seen before this was fixed had three causes acting together:
unnormalized inputs (pitfall 1), no cuDNN/TF32 determinism, and `num_workers > 0`.

Not verified: seed sensitivity. Only seed 42 was swept; a 3-seed study was listed
as outstanding work and never run.

---

## 8. What this protocol cannot support

- **No SOTA claim.** One model (PaDiM/ResNet-18), two categories. PatchCore and
  wide_resnet50_2 were deliberately not run.
- **No cross-category generalization claim.** MVTec AD 2 `fabric` (10 GB) was not
  downloaded; `sheet_metal` is the only AD2 category.
- **No precise effect sizes.** 19 scenes; see the CI width in pitfall 5.
- **No independent test set.** See pitfall 6.
- **Not a paper.** No novel method is proposed. `docs/reports/Industrial_AD_Research_Summary.md`
  explains why the direction was stopped.
