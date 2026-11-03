# Dataset placeholder

Nothing in this directory is redistributed. Both datasets are licensed
**CC BY-NC-SA 4.0** and must be downloaded from MVTec yourself:

- MVTec AD — <https://www.mvtec.com/research-teaching/datasets/mvtec-ad>
- MVTec AD 2 — <https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2>

## Expected layout

The code reads `datasets/raw/` by default. Override with `IDAD_DATA_ROOT` if your
data lives elsewhere (an external drive, for instance).

```
datasets/raw/
├── mvtec_ad/
│   └── tile/                     # any MVTec AD category works for the tile pipeline
│       ├── train/good/*.png      # 230 images for tile
│       ├── test/
│       │   ├── good/*.png        # 33
│       │   ├── crack/*.png       # 84 defect images in total, across defect types
│       │   └── ...
│       ├── ground_truth/crack/*_mask.png
│       ├── license.txt
│       └── readme.txt
└── ad2/
    └── sheet_metal/
        ├── train/good/*.png      # 137, all under "regular" lighting
        ├── validation/good/*.png # 19, all under "regular" lighting
        ├── test_public/
        │   ├── good/*.png        # 24
        │   ├── <defect type>/*.png  # 90, each with a mask
        │   └── ground_truth/...
        ├── test_private/         # released without ground truth - not used here
        └── test_private_mixed/   # released without ground truth - not used here
```

## Verification

After extracting, run the checkers — they validate counts, mask pairing, image
sizes and content-level train/test leakage, and write a report to `results/metrics/`:

```bash
python src/dataset_check.py --category tile
python src/ad2_check.py --archive datasets/raw/ad2/sheet_metal.tar.gz
```

The reports produced by the original runs are archived in
`docs/reports/dataset_check_tile.md` and `docs/reports/ad2_check_sheet_metal.md`
so you can compare your copy against them without running anything.

## Note on MVTec AD 2 file names

The lighting condition is encoded in the file name — `000_regular.png`,
`000_overexposed.png`, `000_shift_1.png` … — so the six variants of one scene
share the numeric prefix and the ground truth is available for each variant.
That is what makes paired analysis possible. Do not assume two files are the
same scene just because they look similar; use the prefix.
