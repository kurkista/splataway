#!/usr/bin/env python3
"""
train_splatfacto.py — Train 3D Gaussian Splats using Nerfstudio splatfacto.

Replaces OpenSplat with a pure-PyTorch implementation that works reliably
on macOS with Metal (MPS) via PyTorch — no binary compilation required.

Usage:
  python3 train_splatfacto.py <colmap_dir> -n ITERS -o OUTPUT.ply
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run(cmd: list, log_path: Path | None = None, env: dict | None = None) -> None:
    cmd = [str(c) for c in cmd]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,   # feed "n" to any Rich Confirm.ask() prompts
        text=True,
        env=env,
    )
    # Accept any interactive prompts (e.g. "downscale images now?") automatically.
    # Nerfstudio auto-selects a downscale factor for large images; saying "y" creates
    # the downscaled copies once in a sibling folder and reuses them on future runs.
    if proc.stdin:
        proc.stdin.write("y\n" * 10)
        proc.stdin.close()
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if log_path:
            with open(log_path, "a") as log:
                log.write(line)
    proc.wait()
    if proc.returncode != 0:
        print(f"\nERROR: command exited with code {proc.returncode}", flush=True)
        sys.exit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train splatfacto and export .ply")
    parser.add_argument("colmap", help="Path to COLMAP project directory (contains sparse/)")
    parser.add_argument("-n", "--iters", type=int, default=7000)
    parser.add_argument("-o", "--output", required=True, help="Output .ply file path")
    parser.add_argument("--log", help="Append output to this log file")
    args = parser.parse_args()

    colmap_dir = Path(args.colmap).resolve()
    output_ply = Path(args.output).resolve()
    log_path = Path(args.log) if args.log else None
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    if not (colmap_dir / "sparse" / "0").exists():
        print(f"ERROR: No COLMAP reconstruction at {colmap_dir}/sparse/0/")
        sys.exit(1)

    # Nerfstudio writes output to: <ns_dir>/splatfacto/<experiment>/
    ns_dir = SCRIPT_DIR / "projects" / output_ply.parent.name / "nerfstudio"
    ns_dir.mkdir(parents=True, exist_ok=True)

    # ── Train ───────────────────────────────────────────────────────────────
    import os
    train_env = {
        **os.environ,
        "WANDB_MODE":       "disabled",   # no wandb login prompt
        "COMET_MODE":       "disabled",   # no comet login prompt
        "WANDB_SILENT":     "true",
    }
    run([
        SCRIPT_DIR / ".venv" / "bin" / "ns-train",
        "splatfacto",
        "--data",               colmap_dir,
        "--output-dir",         ns_dir,
        "--max-num-iterations", args.iters,
        "--vis",                "viewer_legacy",   # headless-compatible viewer
        "--machine.device-type", "cpu",             # gsplat rasterizer has no MPS backend
        "colmap",                                   # explicit dataparser
        "--colmap-path",        "sparse/0",
        "--images-path",        "images",
    ], log_path, env=train_env)

    # ── Find config.yml ─────────────────────────────────────────────────────
    configs = sorted(ns_dir.glob("splatfacto/*/config.yml"), key=lambda p: p.stat().st_mtime)
    if not configs:
        print(f"ERROR: No config.yml found under {ns_dir}/splatfacto/")
        sys.exit(1)
    config = configs[-1]
    print(f"\nUsing config: {config}")

    # ── Export .ply ─────────────────────────────────────────────────────────
    export_dir = output_ply.parent / "ns_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    run([
        SCRIPT_DIR / ".venv" / "bin" / "ns-export",
        "gaussian-splat",
        "--load-config",  config,
        "--output-dir",   export_dir,
    ], log_path)

    # Move exported .ply to final output path
    exported = next(export_dir.glob("*.ply"), None)
    if exported:
        shutil.move(str(exported), str(output_ply))
        print(f"\nOutput: {output_ply}")
    else:
        print(f"ERROR: No .ply found in {export_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
