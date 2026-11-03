# -*- coding: utf-8 -*-
"""工业异常检测科研项目 —— 环境自检与环境快照存档。

用法:
    python src/env_check.py

输出:
    results/logs/env_report_<timestamp>.json   机器可读的环境快照
    results/logs/env_report.md                 人可读摘要
    docs/env/requirements_export.txt           当前 venv 的 pip freeze
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "results" / "logs"
ENV_DOC_DIR = PROJECT_ROOT / "results" / "env"

# 关键包 -> 用来判断"是否装了正确的东西"
KEY_PACKAGES = [
    "torch",
    "torchvision",
    "anomalib",
    "lightning",
    "numpy",
    "opencv-python-headless",
    "opencv-python",
    "matplotlib",
    "scikit-learn",
    "scikit-image",
    "timm",
    "kornia",
    "torchmetrics",
    "pillow",
    "pandas",
]


def _run(cmd: list[str]) -> str:
    """执行外部命令，失败返回错误字符串而不抛异常。"""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace"
        )
        return (out.stdout or "") + (out.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return f"<failed: {exc}>"


def collect() -> dict:
    info: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }

    # --- NVIDIA 驱动 ---
    nvidia_smi = shutil.which("nvidia-smi")
    info["nvidia_smi_available"] = nvidia_smi is not None
    if nvidia_smi:
        info["nvidia_smi"] = _run([nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv"])

    # --- pip 包版本 ---
    from importlib import metadata

    pkgs = {}
    for name in KEY_PACKAGES:
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    info["packages"] = pkgs

    # --- PyTorch / CUDA ---
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_compiled_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "mps_available": getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available(),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "total_memory_MB": round(props.total_memory / 1024**2, 1),
                "multi_processor_count": props.multi_processor_count,
            }
            # --- 真实算力验证：真跑一次卷积/矩阵乘 + 多卡内同步 ---
            try:
                dev = torch.device("cuda:0")
                torch.cuda.reset_peak_memory_stats(dev)
                a = torch.randn(2048, 2048, device=dev)
                b = torch.randn(2048, 2048, device=dev)
                c = a @ b
                torch.cuda.synchronize(dev)
                info["gpu_compute_test"] = {
                    "matmul_shape": [2048, 2048],
                    "result_sum": float(c.sum().item()),
                    "dtype": str(c.dtype),
                    "peak_mem_MB": round(torch.cuda.max_memory_allocated(dev) / 1024**2, 1),
                    "status": "ok",
                }
                # 半精度也测一下（4GB 卡常用 AMP）
                a16 = a.half()
                b16 = b.half()
                c16 = a16 @ b16
                torch.cuda.synchronize(dev)
                info["gpu_compute_test"]["fp16_matmul_max"] = float(c16.max().item())
                del a, b, c, a16, b16, c16
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                info["gpu_compute_test"] = {"status": f"failed: {exc}"}
        else:
            info["gpu_compute_test"] = {"status": "skipped: cuda not available"}
    except Exception as exc:  # noqa: BLE001
        info["torch"] = {"import_error": str(exc)}

    # --- CPU / 内存 ---
    try:
        import psutil

        info["host"] = {
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "ram_total_GB": round(psutil.virtual_memory().total / 1024**3, 1),
        }
    except Exception:  # noqa: BLE001
        info["host"] = {"cpu_count_logical": os.cpu_count()}

    return info


def write_outputs(info: dict) -> tuple[Path, Path, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ENV_DOC_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = LOG_DIR / f"env_report_{stamp}.json"
    json_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    # pip freeze 全量快照（复现用）
    freeze = _run([sys.executable, "-m", "pip", "freeze"])
    freeze_path = ENV_DOC_DIR / "requirements_export.txt"
    freeze_path.write_text(freeze, encoding="utf-8")

    lines = []
    lines.append(f"# 环境快照 {info['timestamp']}\n")
    lines.append(f"- Python: {info['python']['version'].splitlines()[0]}")
    lines.append(f"- 解释器: `{info['python']['executable']}`")
    lines.append(f"- 平台: {info['platform']['system']} {info['platform']['release']} ({info['platform']['machine']})")

    torch_info = info.get("torch", {})
    if "import_error" in torch_info:
        lines.append(f"- **PyTorch 导入失败**: {torch_info['import_error']}")
    else:
        lines.append(f"- PyTorch: {torch_info.get('version')} (编译 CUDA {torch_info.get('cuda_compiled_version')})")
        lines.append(f"- cuDNN: {torch_info.get('cudnn_version')}")
        lines.append(f"- CUDA 可用: **{torch_info.get('cuda_available')}**, 设备数: {torch_info.get('device_count')}")
    if "gpu" in info:
        g = info["gpu"]
        lines.append(f"- GPU: {g['name']} | sm_{g['capability'].replace('.', '')} | 显存 {g['total_memory_MB']} MB")
    if "gpu_compute_test" in info:
        lines.append(f"- GPU 实测算力: {info['gpu_compute_test']}")
    if "host" in info:
        lines.append(f"- 主机: {info['host']}")
    lines.append("\n## 关键包版本\n")
    for k, v in sorted(info.get("packages", {}).items()):
        lines.append(f"- {k} == {v}")
    lines.append(f"\n完整 pip freeze: `{freeze_path.relative_to(PROJECT_ROOT)}`")

    md_path = LOG_DIR / "env_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path, freeze_path


def main() -> int:
    ap = argparse.ArgumentParser(description="环境自检：版本、GPU 可用性、实测算力，并落盘快照")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="CUDA 不可用时仍以退出码 0 结束（CPU-only 机器上跑冒烟测试用）")
    args = ap.parse_args()

    info = collect()
    jp, mp, fp = write_outputs(info)
    print(mp.read_text(encoding="utf-8"))
    print(f"\n[OK] JSON: {jp}")
    print(f"[OK] MD  : {mp}")
    print(f"[OK] 依赖: {fp}")

    ok = bool(info.get("torch", {}).get("cuda_available"))
    if not ok:
        print("\n[WARN] CUDA 不可用，后续实验会退回 CPU。")
        return 0 if args.allow_cpu else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
