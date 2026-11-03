# -*- coding: utf-8 -*-
"""校验并解压 MVTec AD 单类别压缩包。

用法:
    python scripts/prepare_dataset.py --archive datasets/raw/tile.tar.xz --out datasets/raw/mvtec_ad

行为:
1. 记录文件大小与 SHA256（写入 .sha256 文件，供论文复现留痕）。
2. 列出压缩包成员，检查顶层目录名。
3. 解压到目标目录（若已存在同名类别目录则跳过，避免重复解压）。
4. 打印解压后的顶层结构，交给 src/dataset_check.py 做细粒度校验。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(PROJECT_ROOT / "datasets" / "raw" / "tile.tar.xz"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "datasets" / "raw" / "mvtec_ad"))
    args = ap.parse_args()

    arc = Path(args.archive)
    out = Path(args.out)
    if not arc.exists():
        print(f"[ERR] 找不到压缩包: {arc}")
        return 1

    size_mb = arc.stat().st_size / 1024**2
    print(f"压缩包: {arc}")
    print(f"大小  : {size_mb:.1f} MB")
    digest = sha256_of(arc)
    print(f"SHA256: {digest}")
    (arc.parent / (arc.name + ".sha256")).write_text(f"{digest}  {arc.name}\n", encoding="utf-8")

    if not tarfile.is_tarfile(arc):
        print("[ERR] 不是有效的 tar 归档（可能是 tar 后缀但用了别的压缩，或被截断）")
        return 2

    # 顶层目录
    with tarfile.open(arc, "r:*") as tf:
        names = tf.getnames()
        tops = sorted({n.split("/")[0] for n in names if n and not n.startswith(".")})
        print(f"成员数: {len(names)}")
        print(f"顶层  : {tops}")

        category = tops[0] if len(tops) == 1 else None
        target = out / category if category else out
        if category and target.exists():
            print(f"[SKIP] 已存在 {target}，跳过解压")
        else:
            out.mkdir(parents=True, exist_ok=True)
            print(f"解压到: {out} ...")
            tf.extractall(out, filter="data")
            print("[OK] 解压完成")

    # 结构预览
    print("\n目录结构（前 2 层 + 文件数）:")
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        n_all = sum(1 for _ in d.rglob("*") if _.is_file())
        print(f"  {d.name}/  ({n_all} 个文件)")
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            n = sum(1 for _ in sub.rglob("*") if _.is_file())
            print(f"    {sub.name}/  ({n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
