# Third-party notices

This repository contains **only original code and original analysis**. It does
not vendor or redistribute any third-party source code, model weights, or
dataset. The dependencies below are installed by the user, not shipped here.

## Python packages

| Package | Version used | License | Used for |
|---|---|---|---|
| [anomalib](https://github.com/open-edge-platform/anomalib) | 2.7.0 | Apache-2.0 | PaDiM implementation, data modules, PreProcessor / PostProcessor |
| [PyTorch](https://pytorch.org) | 2.14.1+cu126 | BSD-3-Clause | tensor runtime |
| [torchvision](https://github.com/pytorch/vision) | 0.29.1+cu126 | BSD-3-Clause | image I/O, resize / normalize |
| [Lightning](https://github.com/Lightning-AI/pytorch-lightning) | 2.6.6 | Apache-2.0 | trainer used by anomalib |
| scikit-learn | 1.9.1 | BSD-3-Clause | ROC AUC / average precision |
| SciPy | 1.18.1 | BSD-3-Clause | connected components for AUPRO |
| NumPy / pandas / Matplotlib / Pillow / PyYAML / psutil | see `requirements.txt` | BSD / PSF / Apache-2.0 | numerics, IO, plotting |

This project calls anomalib's public API. It does not modify or reimplement
PaDiM. No anomalib source file is copied into `src/`.

## Pretrained weights

Anomalib downloads the ImageNet backbone (`timm/resnet18.a1_in1k`) from
Hugging Face at first use. Those weights are **not** included here; they carry
their own licence upstream. Set `HF_ENDPOINT=https://hf-mirror.com` if
huggingface.co is unreachable from your network.

## Datasets

| Dataset | License | Redistributed here? |
|---|---|---|
| MVTec AD (Bergmann et al., CVPR 2019 / IJCV 2021) | CC BY-NC-SA 4.0 | **No** |
| MVTec AD 2 (IJCV 2026) | CC BY-NC-SA 4.0 | **No** |

Both are **non-commercial**. Users must obtain them from MVTec directly.

Because of this, the figures shipped in `assets/figures/` contain **no dataset
pixels**. They are charts rendered from numbers (metrics, score histograms,
aggregation comparisons). The heatmap overlays produced by the original runs are
deliberately excluded; regenerate them locally with the scripts in `src/` after
downloading the data.

## Repository licence

No `LICENSE` file is included yet. The code is the project owner's own work, but
the choice of an outbound licence (MIT / Apache-2.0 / none) has not been
confirmed, so it is intentionally left unset rather than guessed. Add one before
publishing if you want others to reuse the code. Until then, default copyright
applies: all rights reserved.
