#!/usr/bin/env python3
"""
splat.py — Gaussian Splat automation pipeline
Video or image folder → COLMAP SfM → OpenSplat (Metal) → .ply

Usage:
  python3 splat.py <input> [options]

Examples:
  python3 splat.py inbox/cemetery.mp4
  python3 splat.py inbox/building/ --matcher exhaustive --iters 30000
  python3 splat.py footage/fly.mov  --name castle --fps 1
  python3 splat.py footage/fly.mov  --name castle --from-step matching
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
try:
    import tomllib          # Python 3.11+
except ImportError:
    import tomli as tomllib  # Python < 3.11
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "config.toml"
PROJECTS    = SCRIPT_DIR / "projects"
OPENSPLAT   = SCRIPT_DIR / "OpenSplat" / "build" / "opensplat"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".dng"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mts", ".avi", ".mkv", ".m4v"}

STEPS = ["frames", "features", "matching", "mapping", "train"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


def step_header(n: int, total: int, label: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  [{n}/{total}] {label}\n{bar}\n", flush=True)


def run(cmd: list, log_path: Path, dry_run: bool = False, env: dict | None = None) -> None:
    cmd = [str(c) for c in cmd]
    print(f"$ {' '.join(cmd)}\n", flush=True)
    if dry_run:
        return
    with open(log_path, "a") as log:
        log.write(f"\n[{datetime.now().isoformat()}] {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        proc.wait()
    if proc.returncode != 0:
        print(f"\nERROR: command exited with code {proc.returncode}", flush=True)
        print(f"Full log: {log_path}")
        sys.exit(proc.returncode)


def guard_tool(name: str, hint: str = "") -> None:
    if not shutil.which(name):
        print(f"ERROR: '{name}' not found. {hint}")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated Gaussian Splat pipeline (COLMAP + OpenSplat)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input",        help="Video file or folder of images")
    parser.add_argument("--name",       help="Project name (default: input stem)")
    parser.add_argument("--output",     help="Destination path for the final .ply file")
    parser.add_argument("--fps",        type=float, help="Frames/sec to extract from video")
    parser.add_argument("--iters",      type=int,   help="OpenSplat training iterations")
    parser.add_argument(
        "--matcher",
        choices=["sequential", "exhaustive"],
        help="'sequential' for video/ordered frames; 'exhaustive' for photo sets",
    )
    parser.add_argument(
        "--from-step",
        choices=STEPS,
        metavar="STEP",
        help=f"Resume from a specific step: {', '.join(STEPS)}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    # ── Config + arg merging ─────────────────────────────────────────────────
    cfg     = load_config()
    fps     = args.fps     or cfg["ffmpeg"]["fps"]
    iters   = args.iters   or cfg["opensplat"]["iterations"]
    matcher = args.matcher or cfg["colmap"]["matcher"]
    quality = cfg["ffmpeg"]["quality"]
    single_camera = 1 if cfg["colmap"]["single_camera"] else 0
    out_name      = cfg["opensplat"]["output_name"]
    sh_degree     = cfg["opensplat"].get("sh_degree", 1)
    skip_before   = STEPS.index(args.from_step) if args.from_step else 0

    # ── Input validation ─────────────────────────────────────────────────────
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}")
        sys.exit(1)

    is_video  = input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS
    is_images = input_path.is_dir()
    if not is_video and not is_images:
        print(f"ERROR: input must be a video file or an image folder. Got: {input_path}")
        sys.exit(1)

    name = args.name or input_path.stem or "scene"

    # Final .ply destination — caller (watch.sh) can override via --output
    out_ply = Path(args.output) if args.output else SCRIPT_DIR / "output" / name / out_name
    if not args.dry_run:
        out_ply.parent.mkdir(parents=True, exist_ok=True)

    # ── Guard required tools ─────────────────────────────────────────────────
    if is_video:
        guard_tool("ffmpeg", "Run: brew install ffmpeg")
    guard_tool("colmap", "Run: brew install colmap")
    if not OPENSPLAT.exists():
        print(f"ERROR: opensplat binary not found at {OPENSPLAT}\nRun: bash install.sh")
        sys.exit(1)

    # ── Intermediate directory layout ─────────────────────────────────────────
    proj      = PROJECTS / name
    frames    = proj / "frames"
    colmap    = proj / "colmap"
    sparse    = colmap / "sparse"
    db        = colmap / "database.db"
    images_ln = colmap / "images"    # symlink → ../frames (OpenSplat expects this layout)
    log_path  = proj / "run.log"

    if not args.dry_run:
        for d in [colmap, sparse]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'━' * 60}")
    print(f"  Project  : {name}")
    print(f"  Input    : {input_path}")
    print(f"  Output   : {out_ply}")
    print(f"  Matcher  : {matcher}  |  Iterations : {iters}")
    if args.from_step:
        print(f"  Resuming from: {args.from_step}")
    if args.dry_run:
        print(f"  DRY RUN")
    print(f"{'━' * 60}")

    total = len(STEPS)

    # ── Step 1: Frame extraction ──────────────────────────────────────────────
    # colmap_images is the real directory COLMAP and OpenSplat read from.
    # For JPEG folders we pass input_path directly (no symlink indirection that
    # can confuse COLMAP's relative-path bookkeeping).
    # For video / converted images it is the frames/ dir we write to.
    if is_video:
        colmap_images = frames
    elif is_images:
        source_images = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        already_jpeg = all(f.suffix.lower() in {".jpg", ".jpeg"} for f in source_images)
        if already_jpeg:
            colmap_images = input_path.resolve()
        else:
            colmap_images = frames
            if not args.dry_run:
                frames.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        # OpenSplat expects colmap/images/ to exist — symlink it to the real image dir
        images_ln = colmap / "images"
        if images_ln.is_symlink():
            images_ln.unlink()
        if not images_ln.exists():
            images_ln.symlink_to(colmap_images)

    if STEPS.index("frames") >= skip_before:
        step_header(1, total, "Frame extraction")
        if is_video:
            if not args.dry_run:
                frames.mkdir(parents=True, exist_ok=True)
            run([
                "ffmpeg", "-i", input_path,
                "-qscale:v", quality,
                "-vf", f"fps={fps}",
                frames / "%04d.jpg",
            ], log_path, args.dry_run)
        elif already_jpeg:
            print(f"All JPEGs in input folder ({len(source_images)} images) — used directly, no copy.")
        else:
            # Convert PNG/TIFF/DNG → JPEG (3–5× smaller; COLMAP needs no lossless data)
            print(f"Converting {len(source_images)} images to JPEG (saves ~70% disk vs PNG)...")
            for i, img in enumerate(source_images, 1):
                out = frames / f"{i:04d}.jpg"
                if args.dry_run:
                    print(f"  ffmpeg -i {img.name} → {out.name}")
                elif not out.exists():
                    subprocess.run(
                        ["ffmpeg", "-i", str(img), "-qscale:v", str(quality),
                         "-loglevel", "error", str(out)],
                        check=True,
                    )
            if not args.dry_run:
                print(f"Converted {len(source_images)} images.")
    else:
        print(f"\n  Skipping: frames")

    # ── Step 2: COLMAP feature extraction ────────────────────────────────────
    if STEPS.index("features") >= skip_before:
        step_header(2, total, "COLMAP — feature extraction")
        run([
            "colmap", "feature_extractor",
            "--database_path", db,
            "--image_path",    colmap_images,
            "--ImageReader.single_camera", single_camera,
        ], log_path, args.dry_run)
    else:
        print(f"\n  Skipping: features")

    # ── Step 3: COLMAP feature matching ──────────────────────────────────────
    if STEPS.index("matching") >= skip_before:
        step_header(3, total, f"COLMAP — {matcher} matching")
        run([
            "colmap", f"{matcher}_matcher",
            "--database_path", db,
        ], log_path, args.dry_run)
    else:
        print(f"\n  Skipping: matching")

    # ── Step 4: COLMAP SfM mapper ─────────────────────────────────────────────
    if STEPS.index("mapping") >= skip_before:
        step_header(4, total, "COLMAP — SfM mapping (camera poses + sparse cloud)")
        run([
            "colmap", "mapper",
            "--database_path", db,
            "--image_path",    colmap_images,
            "--output_path",   sparse,
        ], log_path, args.dry_run)

        if not args.dry_run:
            reconstruction = sparse / "0"
            if not reconstruction.exists():
                print("\nERROR: COLMAP produced no reconstruction.")
                print("Possible causes:")
                print("  • Not enough image overlap (aim for 70–80% between consecutive frames)")
                print("  • Too few images (minimum ~20 for a meaningful scene)")
                print("  • Motion blur or blown highlights reducing feature matches")
                print(f"  • Check {log_path} for COLMAP output")
                sys.exit(1)
    else:
        print(f"\n  Skipping: mapping")

    # ── Step 5: Nerfstudio Gaussian Splat training ──────────────────────────
    if STEPS.index("train") >= skip_before:
        step_header(5, total, f"OpenSplat — 3DGS training ({iters} iterations, CPU)")
        print("  (Running on CPU — Metal toolchain not available on macOS 26 / Xcode 26.5)")
        # OMP_NUM_THREADS=1 is required even in CPU mode: both PyTorch and OpenSplat
        # bundle libomp, and without this the mutex initialisation races and crashes.
        splat_env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"}
        run([OPENSPLAT, colmap, "-n", iters, "-o", out_ply, "--cpu",
             "--sh-degree", sh_degree],
            log_path, args.dry_run, env=splat_env)

        if not args.dry_run:
            print(f"\n{'━' * 60}")
            print(f"  Done.")
            print(f"  Output : {out_ply}")
            print(f"  Log    : {log_path}")
            print(f"  Viewer : https://superspl.at/editor  (drag & drop the .ply)")
            print(f"{'━' * 60}\n", flush=True)
    else:
        print(f"\n  Skipping: train")


if __name__ == "__main__":
    main()
