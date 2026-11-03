# Six evaluation pitfalls we hit, measured rather than assumed

Every item below is something **this project actually did wrong and then fixed**.
The numbers are from the runs archived in `docs/reports/` and `docs/results/`.

Scope warning: these are observations under specific framework versions and
evaluation protocols. They are not universal laws. In particular, items 1, 2 and 5
are tied to **anomalib 2.7.0 / PyTorch Lightning 2.6.6** and to the way this
project drives them; a future release may change the behaviour. Check the version
before transferring a conclusion.

---

## 1. Bypassing the model `PreProcessor` silently drops ImageNet normalization

**What we did.** Day 1, in order to get *un-clamped* anomaly scores, we wrote our
own loop:

```python
for batch in dataloader:
    out = model.model(batch.image)     # "skip the post-processor"
```

**Why it is wrong.** Anomalib 2.x splits preprocessing across two layers:

| Step | Where it happens |
|---|---|
| `uint8 -> [0,1]` | dataset `__getitem__` |
| `Resize(256, bilinear, antialias)` | injected into the dataset's `augmentations` |
| `Normalize(ImageNet mean/std)` | **model `PreProcessor`, a Lightning callback** |

`PreProcessor` only runs inside a Trainer loop. A plain `for batch in dataloader`
skips it, so the ResNet backbone received `[0,1]` inputs instead of normalized
ones. Nothing raises; the model still returns plausible-looking maps.

**How we caught it.** Compare the tensor statistics, not the metrics:

| Check | Value |
|---|---|
| DataLoader batch before fix | `min = 0.0578`, `max = 1.0000` — normalized inputs must contain negatives |
| Explicitly rebuilt pipeline | `min = -1.8069`, `max = 1.7143` |
| Max abs difference, batch vs pipeline | 1.878 |
| Image score, same image, same weights | **77.68** (unnormalized) vs **116.72** (normalized) |

**Cost.** On the same 117 test images, same model, same masks:

| Input | Image AUROC | Pixel AUROC | Pixel AP |
|---|---|---|---|
| `[0,1]`, not normalized | 0.8442 | 0.8387 | 0.2762 |
| ImageNet-normalized | **0.9426** | **0.9047** | **0.3279** |

Roughly 0.10 Image AUROC lost to a missing normalization step.

**Rule.** If you write your own inference loop over an anomalib model, rebuild
the transform explicitly (see `src/eval_core.py:TRANSFORM`) and assert it
element-wise against the framework path once, at the start.

---

## 2. The `PostProcessor` clamp creates ties and destroys rank-based metrics

**What we assumed.** Day 1 wrote that anomalib normalizes anomaly maps
"per image", so scores are not comparable across images. That was a guess.

**What the source says.** `anomalib/post_processing/post_processor.py`:

```python
normalized = clip((preds - threshold) / (norm_max - norm_min) + 0.5, 0, 1)
```

A **global affine map plus a hard clamp**, with `min` / `max` / `threshold` all
taken from the validation split. Reproducing that formula on our scores matched
to `2.6e-8` (image) and `0.0` (pixel) — so the formula, not an inference, is what
we were dealing with.

**Why it damages metrics.** `clip` is monotone but *not strictly* monotone:

| Symptom | Value |
|---|---|
| Distinct image scores | 117 -> **47** |
| Rank order preserved | **41.0%** |
| Pixels clipped to exactly 0 | **95.6%** |
| Pixels clipped to exactly 1 | 0.1% |

Ties are scored as 0.5 by AUROC, so:

| Metric | Raw scores | After PostProcessor |
|---|---|---|
| Image AUROC | 0.9426 | 0.8849 (−0.058) |
| Pixel AUROC | 0.9047 | 0.5889 (−0.316) |
| Pixel AP | 0.3279 | 0.1334 (−0.195) |

The mean+3σ-style threshold landed on the validation maximum (65.5), and test
pixels are far above it, so `(x − 65.5)/58 + 0.5` goes negative for almost every
background pixel and is clipped to the same value.

**Rule.** Rank-based metrics need strictly ordered scores. Use the raw Mahalanobis
distance for evaluation; keep the PostProcessor output for human-facing overlays
only. Regression test: `tests/test_metrics.py::test_clamp_creates_ties_and_destroys_ranking`.

---

## 3. How you rasterize the ground-truth mask changes the score

**What happened.** Day 3 reported Pixel AUROC 0.8138 / AP 0.1638 / AUPRO 0.1669.
Day 4, on the *same cached heatmaps*, reported 0.8030 / 0.1572 / 0.2243. Image
AUROC was identical (0.6931 vs 0.6928), so the discrepancy was isolated to the
pixel evaluation step.

**Diagnosis.** Recompute the same maps under both protocols and cross them:

| Score map | Mask source | Pixel AUROC | AUPRO |
|---|---|---|---|
| @256×1024 | rasterized from original 4224×1056 GT | 0.8138 | 0.1670 |
| @256×256 | from the 256 grid | 0.8030 | 0.2243 |
| @256×1024 | upsampled from the 256 grid | 0.8030 | 0.2242 |

The score-map grid is irrelevant; **the mask's rasterization is the whole story.**

**Which one is right?** Neither is "correct" — they measure different things. But
the 256-wide mask is *optimistic* for AUPRO: sheet-metal defects are hairline
scratches a few pixels wide at 4224 px, and downscaling by 16.5× thickens them,
so per-region overlap is easier to satisfy. AUPRO inflates by 0.057 (relative
+34%).

**Rule.** Any pixel-level number must state (a) the score-map grid, (b) where the
mask was rasterized from and with which interpolation. Image-level numbers can be
compared across days; pixel-level numbers cannot, unless the protocol matches.

---

## 4. Image-level aggregation can be dominated by border extremes

**What happened.** With identical weights and identical maps, changing only the
input geometry changed conclusions:

| Config | Image AUROC (max) | Pixel AUROC | AUPRO |
|---|---|---|---|
| squash 256×256 | 0.693 | 0.814 | 0.167 |
| aspect 256×1024 | 0.548 | 0.894 | 0.444 |

Localization got much better while ranking got worse. The controlled experiment
(freeze the maps, vary only the aggregation rule) explains it:

- Normal images whose maximum sits in the left/right 10% border band:
  **54.2%** (squash) and **62.5%** (aspect). Counting all four borders: 58.3% / 79.2%.
  Median normalized distance to the left/right border in the aspect config: **0.000**.
- Position-count subsampling on the same map: mean per-image max for normal images
  rises **36.7 -> 63.2 (+72%)** in the aspect config while the defect-minus-normal
  gap collapses **13.0 -> 3.3**.

This is ordinary extreme-value behaviour: `E[max]` grows with the number of
samples, and background positions outnumber defect positions, so more samples
means more chances for background to win.

**Rule.** For wide-format industrial images, crop the region of interest *before*
inference, or do not use max as the image score. Both are engineering hygiene, not
research contributions — see `docs/reports/Day4_Decision.md` for why we stopped
pursuing this.

---

## 5. Multiple lighting variants of one scene are not independent samples

MVTec AD 2 ships six lighting variants per scene (`regular`, `overexposed`,
`underexposed`, `shift_1..3`), each with its own ground truth. The public test
split is 114 images but only **19 independent scenes**, and four normal images per
lighting condition.

Treating 114 images as 114 observations gives absurdly narrow confidence
intervals. Resampling *scenes* instead gives honest ones — and shows how little
the data supports:

| Contrast (aspect config) | ΔAUROC | 95% CI (scene-level bootstrap) |
|---|---|---|
| Top-1% vs max | +0.212 | [+0.064, +0.465] |
| Global-mean vs max | +0.293 | [+0.136, +0.476] |

The direction is clear; the magnitude is not. A CI of width 0.4 cannot support
"method A beats method B by 0.05".

**Rule.** Resample at the unit that was actually varied (scene / workpiece), not
at the unit that was measured (image, and never pixel). Use the same resampling
indices for every method being compared, so the contrast is paired.

---

## 6. Exploring on a public test split is not validation

Everything in this repository was computed on `test_public` (MVTec AD 2) or the
single official `test` split (MVTec AD), and the results informed which questions
we then asked. That makes them **exploratory**, including the "crop 2% + mean"
number (0.8894) in `docs/reports/Industrial_AD_Final_Audit.md`.

The private splits (`test_private`, `test_private_mixed`) are released without
ground truth and are scored by the official evaluation server. We never touched
them, so we have no independent confirmation of anything here.

**Rule.** Label exploratory results as exploratory in the table itself, not in a
footnote. Freeze the method *before* looking at a held-out split; otherwise the
number you report is the maximum over the choices you made, not an estimate of
generalization.
