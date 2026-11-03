# -*- coding: utf-8 -*-
"""多线程分段下载器（用于从慢速但官方的源拉取数据集）。

背景：MVTec 官方分发服务器 mydrive.ch 对单连接限速约 45 KB/s，
分片并发可以显著提升聚合吞吐。本脚本仅做"同一 URL 的 HTTP Range 并发"，
不改变数据来源，下载完成后校验大小并写入 sha256 供后续复现核对。

用法:
    python scripts/download_dataset.py <url> <out_path> --connections 16
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

CHUNK = 256 * 1024


def remote_size(url: str) -> tuple[int, bool]:
    """返回 (content-length, 是否支持 Range)。"""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers.get("Content-Length", 0))
        accept_ranges = r.headers.get("Accept-Ranges", "").lower() == "bytes"
    return size, accept_ranges


def download_range(url: str, start: int, end: int, path: Path, idx: int, retries: int = 8) -> None:
    """下载 [start, end] 闭区间到 path，**断点续传**（重试不清空已写内容）。"""
    expect = end - start + 1
    for attempt in range(1, retries + 1):
        already = path.stat().st_size if path.exists() else 0
        if already >= expect:
            return
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + already}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "ab") as f:
                while True:
                    buf = r.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)
            got = path.stat().st_size
            if got != expect:
                raise IOError(f"分片 {idx} 大小不符: 期望 {expect}, 实际 {got}")
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                raise
            have = path.stat().st_size if path.exists() else 0
            print(f"  [retry {attempt}/{retries}] 分片 {idx} 已有 {have}/{expect}B: {exc}", flush=True)
            time.sleep(min(2 * attempt, 10))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--connections", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part_dir = out.with_suffix(out.suffix + ".parts")
    part_dir.mkdir(exist_ok=True)

    size, accept_ranges = remote_size(args.url)
    if size == 0:
        print("[ERR] 服务端未返回 Content-Length，无法分段下载")
        return 1
    print(f"文件大小: {size/1024**2:.1f} MB | 支持 Range: {accept_ranges} | 并发: {args.connections}")

    if not accept_ranges:
        print("[WARN] 服务端未声明 Accept-Ranges，仍尝试分段（可能失败）")

    n = args.connections
    seg = size // n
    ranges = []
    for i in range(n):
        s = i * seg
        e = size - 1 if i == n - 1 else (s + seg - 1)
        ranges.append((s, e))

    t0 = time.time()
    threads = []
    for i, (s, e) in enumerate(ranges):
        p = part_dir / f"part_{i:03d}"
        if p.exists() and p.stat().st_size >= (e - s + 1):
            continue
        th = threading.Thread(target=download_range, args=(args.url, s, e, p, i), daemon=True)
        th.start()
        threads.append(th)

    # 进度显示
    while any(t.is_alive() for t in threads):
        done = sum(f.stat().st_size for f in part_dir.glob("part_*") if f.exists())
        el = time.time() - t0
        pct = done / size * 100
        print(
            f"\r  {done/1024**2:6.1f}/{size/1024**2:.1f} MB ({pct:5.1f}%) "
            f"{done/1024/el:6.1f} KB/s  {el:5.0f}s",
            end="",
            flush=True,
        )
        time.sleep(2)

    for t in threads:
        t.join()
    print()

    # 拼接
    with open(out, "wb") as fo:
        for i in range(n):
            p = part_dir / f"part_{i:03d}"
            with open(p, "rb") as fi:
                while True:
                    buf = fi.read(1024 * 1024)
                    if not buf:
                        break
                    fo.write(buf)

    got = out.stat().st_size
    print(f"拼接完成: {got/1024**2:.1f} MB (期望 {size/1024**2:.1f} MB)")

    h = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    print(f"SHA256: {digest}")

    sha_file = out.parent / (out.name + ".sha256")
    sha_file.write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    print(f"已写入校验文件: {sha_file}")

    if got == size:
        import shutil

        shutil.rmtree(part_dir, ignore_errors=True)
        print("[OK] 大小校验通过，已清理分片")
        return 0
    print("[ERR] 大小校验失败，保留分片以便续传")
    return 2


if __name__ == "__main__":
    sys.exit(main())
