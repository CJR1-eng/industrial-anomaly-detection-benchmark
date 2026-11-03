# Publish checklist

Things to verify before this repository becomes public. Written from the failures
of the first packaging pass, so that they do not happen twice.

## Never commit

The `.gitignore` covers these; this list exists so you can check by eye.

- `datasets/` — both datasets are CC BY-NC-SA 4.0 and **must not be redistributed**.
  Also excluded: every archive type (`.tar`, `.tar.gz`, `.tar.xz`, `.zip`, `.parts/`).
- Model weights and framework caches: `*.ckpt`, `*.pth`, `*.pt`, `*.onnx`,
  `*.safetensors`, `*.bin`, `weights/`, `checkpoints/`, `vendor/`, `lightning_logs/`.
  A single PaDiM checkpoint from these runs is ~40 MB; a map cache is ~50 MB.
- `results/` and `figures/` — regenerated artefacts. The curated tables live in
  `docs/results/`, the curated charts in `assets/figures/`.
- `.venv/`, `__pycache__/`, `.pytest_cache/`, `.idea/`, `.vscode/`.
- `.env`, `*.key`, `*.pem`, `secrets.*`, `*.local.yaml`.

## Checked before publishing

| Check | How | Result |
|---|---|---|
| No dataset pixels in shipped figures | all 11 files in `assets/figures/` are matplotlib output from numbers; the only two that ever touch images (heatmap overlays) were excluded | pass |
| No absolute paths in tracked files | pattern `[A-Za-z]:\\` and `/Users/`, `/home/` over every tracked file | pass (one hit in `docs/reports/Day1_Report.md`, redacted) |
| No credentials or tokens | pattern `proxy`, `token`, `api_key`, `password`, `secret`, IP literals | pass (only `HF_ENDPOINT` and prose about an HTTP 502) |
| No file over 512 KB | largest is `assets/figures/fig03_*.png` at 308 KB | pass |
| No model checkpoints or array caches | `find -name '*.ckpt' -o -name '*.npz'` | pass |
| Total repository size | `du -sh` | ~1.7 MB |
| Tests pass from a clean tree | `python -m pytest tests -q` | 57 passed |
| Runs with no data give a readable error | `python src/dataset_check.py --data-root no-such-dir` | non-zero exit with the category name and download URL |
| Third-party licences documented | `THIRD_PARTY_NOTICES.md` | pass |

## Deliberately not done

- **No `LICENSE` file.** The outbound licence for this code has not been decided.
  Choosing MIT or Apache-2.0 is a decision for the owner, not a default. Until one
  is added, default copyright applies.
- **No CI config.** A GitHub Actions workflow would need to download 335 MB of
  licensed data to mean anything; the unit tests run in 25 s locally.
- **No git history.** The repository was assembled as a fresh tree rather than by
  rewriting the original project's history, so there is nothing in the past to
  scrub. If you decide to import the original history instead, note that it
  contains absolute Windows paths and multi-GB dataset commits — a `.gitignore`
  added later does **not** remove them.

## Before flipping to public

1. `git status` shows only intended files; `git ls-files | wc -l` is around 40.
2. Render `README.md` on GitHub and click every image link — all should resolve
   relative to `assets/figures/`.
3. Decide the licence.
4. Re-read the "Known limitations" section and confirm you are comfortable with
   every statement in it. It is deliberately unflattering.
