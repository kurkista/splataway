"""
cloud/pack.py — pack a COLMAP scene into a self-contained tar for upload.

Two modes:
  pack_scene()      — local COLMAP already done; packs images + sparse/0/
  pack_images_only() — cloud COLMAP path; packs raw images only, no sparse yet
"""

from __future__ import annotations

import tarfile
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _collect_images(images_dir: Path) -> list[Path]:
    real = images_dir.resolve()
    if not real.exists():
        raise FileNotFoundError(f"Images directory not found: {real}")
    files = sorted(f for f in real.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No image files found in {real}")
    return files


def pack_images_only(images_dir: Path, dest_dir: Path) -> Path:
    """
    Pack raw images into dest_dir/scene.tar.gz for cloud-COLMAP path.
    Archive layout: scene/images/<filename>
    COLMAP will run on the pod and produce scene/colmap/sparse/0/ there.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "scene.tar.gz"
    image_files = _collect_images(images_dir)
    print(f"  Packing {len(image_files)} images for cloud COLMAP + training…")
    tar_path.unlink(missing_ok=True)
    with open(tar_path, "wb") as raw_fh:
        with tarfile.open(fileobj=raw_fh, mode="w:gz") as tar:
            for img in image_files:
                tar.add(img, arcname=f"scene/images/{img.name}")
    size_mb = tar_path.stat().st_size / 1_048_576
    print(f"  Packed: {tar_path.name}  ({size_mb:.1f} MB)")
    return tar_path


def pack_scene(colmap_dir: Path, images_dir: Path, dest_dir: Path) -> Path:
    """
    Pack colmap/sparse/0/ + all images into dest_dir/scene.tar.gz.
    Resolves symlinks so the tar is fully self-contained.
    Returns the path to the created tar.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "scene.tar.gz"

    sparse_dir = colmap_dir / "sparse" / "0"
    if not sparse_dir.exists():
        raise FileNotFoundError(f"COLMAP sparse reconstruction not found: {sparse_dir}")

    image_files = _collect_images(images_dir)

    print(f"  Packing {len(image_files)} images + COLMAP sparse reconstruction…")

    # Open the file explicitly so creation errors are visible immediately,
    # then pass fileobj= to tarfile to avoid Python 3.9's gzopen silently
    # deleting the file on internal init errors.
    tar_path.unlink(missing_ok=True)  # remove any stale partial file
    with open(tar_path, "wb") as raw_fh:
        with tarfile.open(fileobj=raw_fh, mode="w:gz") as tar:
            # Images → scene/images/
            for img in image_files:
                tar.add(img, arcname=f"scene/colmap/images/{img.name}")

            # COLMAP sparse → scene/colmap/sparse/0/
            for f in sparse_dir.iterdir():
                if f.is_file():
                    tar.add(f, arcname=f"scene/colmap/sparse/0/{f.name}")

    size_mb = tar_path.stat().st_size / 1_048_576
    print(f"  Packed: {tar_path.name}  ({size_mb:.1f} MB)")
    return tar_path
