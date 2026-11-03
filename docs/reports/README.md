# Archived experiment reports

These are the **original lab-notebook records** from a five-day investigation, kept
verbatim (only one local absolute path was redacted). They are written in Chinese,
including the numbers that were later found to be wrong. They are archived for
traceability, not as documentation of the current state.

Read them in this order if you want the whole story:

| File | Day | Status | What it contains |
|---|---|---|---|
| `Day1_Report.md` | 1 | ⚠️ **Numbers invalid** | environment bring-up, dataset verification, first PaDiM baseline. Its metrics (Image AUROC 0.8445 / Pixel 0.8387) are wrong — see the errata. |
| `Day2_Errata.md` | 2 | ✅ canonical corrections | itemized list of the 7 errors in Day 1, with the numeric evidence for each |
| `Day2_Report.md` | 2 | ✅ valid | corrected protocol, determinism proof, 256 vs 840 comparison, 6 synthetic illumination conditions |
| `Day3_Report.md` | 3 | ⚠️ partially superseded | real-lighting results on MVTec AD 2 / sheet_metal. Per-lighting and pairing numbers are valid; the §"failure is a calibration failure" attribution was corrected in Day 4. |
| `Day3_Novelty_Matrix.md` | 3 | ✅ valid | 12 closest papers, with evidence grades A/B/C |
| `Day3_Decision.md` | 3 | ✅ valid | why the direction was not declared GO, and the gate conditions set |
| `Day4_Report.md` | 4 | ✅ valid (image-level only) | controlled aggregation experiment. **Its pixel-level columns are not comparable to Day 3** — see `Industrial_AD_Final_Audit.md`. |
| `Day4_Aggregation_Novelty.md` | 4 | ✅ valid | prior work on anomaly-score aggregation, Top-K, extreme-value correction |
| `Day4_Decision.md` | 4 | ✅ valid | stop conditions for the aggregation direction |
| `Day5_Research_Directions.md` | 5 | — (different project) | screening of three follow-up directions; not about this codebase |
| `Day5_Novelty_Comparison.md` | 5 | — (different project) | literature comparison for those directions |
| `Day5_Final_Recommendation.md` | 5 | — (different project) | recommendation and 7-day plan for the follow-up |
| `Industrial_AD_Final_Audit.md` | 5 | ✅ **read this before quoting any pixel metric** | resolves the Day 3 / Day 4 pixel discrepancy, and tests whether "crop ROI + mean" alone explains the aggregation gain |
| `Industrial_AD_Research_Summary.md` | 5 | ✅ **best single summary** | verified conclusions / refuted hypotheses / unverified speculations / overlap with published work / reusable assets |
| `dataset_check_tile.md` | — | ✅ valid | dataset integrity report for MVTec AD tile |
| `ad2_check_sheet_metal.md` | — | ✅ valid | dataset integrity report for MVTec AD 2 sheet_metal |

The last two are the archived outputs of `src/dataset_check.py` and
`src/ad2_check.py`. Running those scripts again writes a fresh copy into
`results/metrics/`, so you can diff your local data against these.

## Reading warning

Two pitfalls bite when skimming these files:

1. **Day 1's numbers are wrong** and Day 2's are right. Do not average them, do not
   cite Day 1.
2. **Pixel-level metrics are not comparable across days** when the mask
   rasterization differs. Day 3 (AUPRO 0.1669) and Day 4 (AUPRO 0.2243) measured the
   same heatmaps under different mask protocols; neither is "the" answer, and the
   Day 3 one is the more faithful of the two.

A third, cosmetic one: the reports link to `figures/dayN_*/…` paths from the
original working directory. Those images are not shipped (they contain dataset
pixels). Regenerate them with the scripts in `src/` — the curated, dataset-free
subset is in `assets/figures/`.

`../evaluation_pitfalls.md` explains both substantive pitfalls, plus four more.
