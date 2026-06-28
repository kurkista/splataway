#!/usr/bin/env python3
"""
splat.py — Gaussian Splat automation pipeline
Video or image folder → COLMAP SfM → OpenSplat → .ply

Usage:
  python3 splat.py <input> [options]

Examples:
  python3 splat.py inbox/cemetery.mp4
  python3 splat.py inbox/building/ --matcher exhaustive --iters 30000
  python3 splat.py footage/fly.mov  --name castle --fps 1
  python3 splat.py footage/fly.mov  --name castle --from-step matching
  python3 splat.py inbox/scene/     --cloud runpod
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


_pipeline_start: float = 0.0

def step_header(n: int, total: int, label: str) -> None:
    import time
    elapsed = time.time() - _pipeline_start
    bar = "=" * 60
    elapsed_str = f"  (+{elapsed/60:.1f} min elapsed)" if n > 1 else ""
    print(f"\n{bar}\n  [{n}/{total}] {label}{elapsed_str}\n{bar}\n", flush=True)


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


def ensure_vocab_tree(db_path: Path, colmap_dir: Path) -> Path:
    """Build a vocab tree from extracted features if one doesn't exist yet."""
    tree_path = colmap_dir / "vocab_tree.bin"
    if tree_path.exists():
        return tree_path
    print("  Building vocab tree from extracted features (one-time, ~5–15 min)…")
    import subprocess as _sp
    result = _sp.run(
        ["colmap", "vocab_tree_builder",
         "--database_path", str(db_path),
         "--vocab_tree_path", str(tree_path),
         "--num_visual_words", "1024",
         "--max_num_descriptors", "500000"],
        check=True,
    )
    print("  Vocab tree ready.")
    return tree_path


def _colmap4_to_3bin(sparse0_dir: Path, log_path: Path) -> None:
    """Convert COLMAP 4.x binary to 3.x binary in-place so OpenSplat can read it.

    COLMAP 4.0 changed images.bin causing an OpenMP crash in OpenSplat's tensor
    clone step (PyTorch libomp conflict when parallelising large tensor copy).
    The text format is stable; we round-trip through it to produce the old binary.
    Only runs if the text conversion produces a different-size images.bin.
    """
    import struct

    txt_dir = sparse0_dir.parent / (sparse0_dir.name + "_txt")
    if txt_dir.exists() and (txt_dir / "cameras.txt").exists():
        pass  # already converted by a previous step
    else:
        txt_dir.mkdir(exist_ok=True)
        subprocess.run(
            ["colmap", "model_converter",
             "--input_path", str(sparse0_dir),
             "--output_path", str(txt_dir),
             "--output_type", "TXT"],
            check=True, capture_output=True,
        )

    # Write COLMAP 3.x-compatible cameras.bin, images.bin, points3D.bin
    MODEL_IDS = {
        "SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2, "RADIAL": 3,
        "OPENCV": 4, "OPENCV_FISHEYE": 5, "FULL_OPENCV": 6, "FOV": 7,
        "SIMPLE_RADIAL_FISHEYE": 8, "RADIAL_FISHEYE": 9, "THIN_PRISM_FISHEYE": 10,
    }

    def _write_cameras(src: Path, dst: Path) -> None:
        rows = [l for l in src.read_text().splitlines() if l and not l.startswith("#")]
        cams = []
        for l in rows:
            p = l.split(); cams.append((int(p[0]), MODEL_IDS[p[1]], int(p[2]), int(p[3]), [float(x) for x in p[4:]]))
        with open(dst, "wb") as f:
            f.write(struct.pack("<Q", len(cams)))
            for cid, mid, w, h, params in cams:
                f.write(struct.pack("<IiQQ", cid, mid, w, h))
                for v in params: f.write(struct.pack("<d", v))

    def _write_images(src: Path, dst: Path) -> None:
        raw = [l for l in src.read_text().splitlines() if l and not l.startswith("#")]
        images = []
        i = 0
        while i < len(raw):
            p = raw[i].split(); i += 1
            img_id, qw, qx, qy, qz = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
            tx, ty, tz, cam_id, name = float(p[5]), float(p[6]), float(p[7]), int(p[8]), p[9]
            pts = []
            if i < len(raw):
                toks = raw[i].split(); i += 1
                for j in range(0, len(toks), 3):
                    pts.append((float(toks[j]), float(toks[j+1]), int(toks[j+2])))
            images.append((img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name, pts))
        with open(dst, "wb") as f:
            f.write(struct.pack("<Q", len(images)))
            for img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name, pts in images:
                f.write(struct.pack("<I", img_id))
                f.write(struct.pack("<ddddddd", qw, qx, qy, qz, tx, ty, tz))
                f.write(struct.pack("<I", cam_id))
                f.write(name.encode() + b"\x00")
                f.write(struct.pack("<Q", len(pts)))
                for x, y, pid in pts: f.write(struct.pack("<ddq", x, y, pid))

    def _write_points3d(src: Path, dst: Path) -> None:
        rows = [l for l in src.read_text().splitlines() if l and not l.startswith("#")]
        with open(dst, "wb") as f:
            f.write(struct.pack("<Q", len(rows)))
            for l in rows:
                p = l.split()
                f.write(struct.pack("<Q", int(p[0])))
                f.write(struct.pack("<ddd", float(p[1]), float(p[2]), float(p[3])))
                f.write(struct.pack("<BBB", int(p[4]), int(p[5]), int(p[6])))
                f.write(struct.pack("<d", float(p[7])))
                track = [(int(p[k]), int(p[k+1])) for k in range(8, len(p), 2)]
                f.write(struct.pack("<Q", len(track)))
                for iid, pidx in track: f.write(struct.pack("<II", iid, pidx))

    print("  Converting COLMAP 4.x binary → 3.x binary for OpenSplat compatibility…")
    _write_cameras(txt_dir / "cameras.txt", sparse0_dir / "cameras.bin")
    _write_images(txt_dir / "images.txt", sparse0_dir / "images.bin")
    _write_points3d(txt_dir / "points3D.txt", sparse0_dir / "points3D.bin")


# Images above this count route COLMAP to the cloud pod when --cloud is set
CLOUD_COLMAP_THRESHOLD = 300

# ── Cloud training ─────────────────────────────────────────────────────────────
def _cloud_train(
    colmap_dir: Path,
    images_dir: Path,
    out_ply: Path,
    iters: int,
    matcher: str,
    gpu: str,
    image: str,
    log_path: Path,
    dry_run: bool,
    cloud_colmap: bool = False,
) -> None:
    from cloud.pack       import pack_scene, pack_images_only
    from cloud.runpod_api import create_pod, wait_ready, run_remote, terminate_pod, install_colmap, run_colmap_remote
    from cloud.transfer   import upload, download

    if dry_run:
        if cloud_colmap:
            print(f"  [dry-run] Large dataset — COLMAP + training both on RunPod ({gpu})")
            print(f"  [dry-run] Remote: colmap {matcher}_matcher → opensplat -n {iters}")
        else:
            print(f"  [dry-run] Would pack scene and train on RunPod ({gpu})")
            print(f"  [dry-run] Remote: opensplat /workspace/scene/colmap -n {iters} --sh-degree 3")
        return

    tmp_dir = SCRIPT_DIR / "projects" / out_ply.parent.name / "cloud_tmp"
    pod_id = None
    try:
        # Pack
        if cloud_colmap:
            tar_path = pack_images_only(images_dir, tmp_dir)
        else:
            tar_path = pack_scene(colmap_dir, images_dir, tmp_dir)

        # Spin up pod
        pod = create_pod(gpu=gpu, image=image, name=f"opensplat-{out_ply.parent.name}")
        pod_id = pod["id"]
        pod = wait_ready(pod_id)

        # Upload
        print("  Uploading…")
        remote_tar = upload(pod, tar_path)

        # Extract on pod
        print("  Extracting…")
        run_remote(pod, f"tar -xzf {remote_tar} -C /workspace/")

        # COLMAP on pod (large dataset path) — COLMAP baked into Docker image
        if cloud_colmap:
            print(f"  Running COLMAP ({matcher} matcher) on pod…")
            with open(log_path, "a") as log:
                run_colmap_remote(pod, matcher, log_file=log)

        # Train
        remote_cmd = (
            f"OMP_NUM_THREADS=4 /opensplat/build/opensplat /workspace/scene/colmap"
            f" -n {iters}"
            f" -o /workspace/splat.ply"
            f" --sh-degree 3"
        )
        print(f"  Training: {remote_cmd}\n")
        with open(log_path, "a") as log:
            log.write(f"\n[{datetime.now().isoformat()}] [runpod:{pod_id}] {remote_cmd}\n")
            rc = run_remote(pod, remote_cmd, log_file=log)
        if rc != 0:
            print(f"\nERROR: remote opensplat exited with code {rc}")
            sys.exit(rc)

        # Download
        print("  Downloading splat.ply…")
        download(pod, "/workspace/splat.ply", out_ply)

    finally:
        if pod_id:
            terminate_pod(pod_id)
        # Clean up local tar
        if "tar_path" in dir() and tar_path.exists():  # type: ignore[possibly-undefined]
            tar_path.unlink()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    import time
    global _pipeline_start
    _pipeline_start = time.time()

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
        choices=["sequential", "exhaustive", "vocab_tree"],
        help="'sequential' for video; 'vocab_tree' for multi-mission/unordered sets; 'exhaustive' for small sets",
    )
    parser.add_argument(
        "--from-step",
        choices=STEPS,
        metavar="STEP",
        help=f"Resume from a specific step: {', '.join(STEPS)}",
    )
    parser.add_argument(
        "--cloud",
        choices=["runpod"],
        metavar="PROVIDER",
        help="Train on cloud GPU instead of local CPU (requires RUNPOD_API_KEY env var)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    # ── Config + arg merging ─────────────────────────────────────────────────
    cfg     = load_config()
    fps     = args.fps     or cfg["ffmpeg"]["fps"]
    matcher = args.matcher or cfg["colmap"]["matcher"]
    quality = cfg["ffmpeg"]["quality"]
    single_camera = 1 if cfg["colmap"]["single_camera"] else 0
    out_name      = cfg["opensplat"]["output_name"]
    sh_degree         = cfg["opensplat"].get("sh_degree", 1)
    num_downscales    = cfg["opensplat"].get("num_downscales", 2)
    reset_alpha_every = cfg["opensplat"].get("reset_alpha_every", 30)
    skip_before       = STEPS.index(args.from_step) if args.from_step else 0

    # Cloud config (used only with --cloud)
    cloud_cfg   = cfg.get("cloud", {})
    cloud_iters = args.iters or cloud_cfg.get("iterations", 30000)
    local_iters = args.iters or cfg["opensplat"]["iterations"]
    cloud_gpu   = cloud_cfg.get("gpu", "NVIDIA GeForce RTX 4090")
    cloud_image = cloud_cfg.get("image", "kurkista/opensplat-cuda:latest")

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
    if not args.cloud and not OPENSPLAT.exists():
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

    # ── Routing: cloud COLMAP for large image sets ───────────────────────────
    # Count source images now so we can decide before the summary is printed.
    if is_images:
        _src_count = sum(1 for f in input_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        _src_count = 0  # video: frame count unknown until extraction, always local COLMAP

    cloud_colmap = bool(args.cloud) and is_images and _src_count > CLOUD_COLMAP_THRESHOLD

    # ── Summary ──────────────────────────────────────────────────────────────
    iters_display = cloud_iters if args.cloud else local_iters
    print(f"\n{'━' * 60}")
    print(f"  Project  : {name}")
    print(f"  Input    : {input_path}")
    print(f"  Output   : {out_ply}")
    print(f"  Matcher  : {matcher}  |  Iterations : {iters_display}")
    if args.cloud:
        colmap_loc = f"cloud pod ({_src_count} images)" if cloud_colmap else "local"
        print(f"  COLMAP   : {colmap_loc}")
        print(f"  Training : cloud GPU ({args.cloud}, {cloud_gpu})")
    if args.from_step:
        print(f"  Resuming from: {args.from_step}")
    if args.dry_run:
        print(f"  DRY RUN")
    print(f"{'━' * 60}")

    total = len(STEPS)

    # ── Step 1: Frame extraction ──────────────────────────────────────────────
    if is_video:
        colmap_images = frames
    elif is_images:
        source_images = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not source_images:
            # Input dir has no images at its top level — it's a COLMAP directory.
            # The colmap/images symlink was established by a prior run; don't touch it.
            colmap_images = None
            already_jpeg = True
        else:
            already_jpeg = all(f.suffix.lower() in {".jpg", ".jpeg"} for f in source_images)
            if already_jpeg:
                colmap_images = input_path.resolve()
            else:
                colmap_images = frames
                if not args.dry_run:
                    frames.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and colmap_images is not None:
        images_ln = colmap / "images"
        new_target = Path(colmap_images).resolve()
        current_target = images_ln.resolve() if images_ln.exists() else None
        if new_target != current_target:
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

    # ── Steps 2–4: COLMAP (skipped when cloud_colmap — runs on pod instead) ──
    if cloud_colmap:
        print("\n  Steps 2–4 (COLMAP) will run on cloud pod — skipping locally.")
    else:
        # ── Step 2: COLMAP feature extraction ────────────────────────────────
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

        # ── Step 3: COLMAP feature matching ──────────────────────────────────
        if STEPS.index("matching") >= skip_before:
            step_header(3, total, f"COLMAP — {matcher} matching")
            match_cmd = ["colmap", f"{matcher}_matcher", "--database_path", db]
            if matcher == "vocab_tree":
                match_cmd += ["--VocabTreeMatching.vocab_tree_path", ensure_vocab_tree(db, colmap)]
            run(match_cmd, log_path, args.dry_run)
        else:
            print(f"\n  Skipping: matching")

        # ── Step 4: COLMAP SfM mapper ─────────────────────────────────────────
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

                # COLMAP 4.x outputs images.bin in a format incompatible with OpenSplat.
                # Convert via stable text format to COLMAP 3.x binary that OpenSplat can read.
                _colmap4_to_3bin(reconstruction, log_path)
        else:
            print(f"\n  Skipping: mapping")

    # ── Step 5: 3DGS training ─────────────────────────────────────────────────
    if STEPS.index("train") >= skip_before:
        if args.cloud:
            step_header(5, total, f"OpenSplat — cloud GPU training ({cloud_iters} iters, {cloud_gpu})")
            _cloud_train(
                colmap_dir   = colmap,
                images_dir   = input_path if cloud_colmap else colmap / "images",
                out_ply      = out_ply,
                iters        = cloud_iters,
                matcher      = matcher,
                gpu          = cloud_gpu,
                image        = cloud_image,
                log_path     = log_path,
                dry_run      = args.dry_run,
                cloud_colmap = cloud_colmap,
            )
        else:
            step_header(5, total, f"OpenSplat — 3DGS training ({local_iters} iterations, Metal GPU)")
            splat_env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"}
            run([OPENSPLAT, colmap, "-n", local_iters, "-o", out_ply,
                 "--sh-degree", sh_degree,
                 "--num-downscales", num_downscales,
                 "--reset-alpha-every", reset_alpha_every],
                log_path, args.dry_run, env=splat_env)

        if not args.dry_run:
            import time
            total_min = (time.time() - _pipeline_start) / 60
            print(f"\n{'━' * 60}")
            print(f"  Done.  Total time: {total_min:.1f} min")
            print(f"  Output : {out_ply}")
            print(f"  Log    : {log_path}")
            print(f"  Viewer : https://superspl.at/editor  (drag & drop the .ply)")
            print(f"{'━' * 60}\n", flush=True)
    else:
        print(f"\n  Skipping: train")


if __name__ == "__main__":
    main()
