"""
cloud/pack.py — pack a COLMAP scene into a self-contained tar for upload.

OpenSplat on the remote pod expects:
  scene/
    images/           ← actual image files (symlink resolved)
    colmap/
      sparse/0/       ← cameras.bin, images.bin, points3D.bin
"""

from __future__ import annotations

import tarfile
from pathlib import Path


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

    # Resolve the images directory (colmap/images is usually a symlink)
    real_images = images_dir.resolve()
    if not real_images.exists():
        raise FileNotFoundError(f"Images directory not found: {real_images}")

    image_files = sorted(
        f for f in real_images.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    if not image_files:
        raise FileNotFoundError(f"No image files found in {real_images}")

    print(f"  Packing {len(image_files)} images + COLMAP sparse reconstruction…")

    with tarfile.open(tar_path, "w:gz") as tar:
        # Images → scene/images/
        for img in image_files:
            tar.add(img, arcname=f"scene/images/{img.name}")

        # COLMAP sparse → scene/colmap/sparse/0/
        for f in sparse_dir.iterdir():
            if f.is_file():
                tar.add(f, arcname=f"scene/colmap/sparse/0/{f.name}")

    size_mb = tar_path.stat().st_size / 1_048_576
    print(f"  Packed: {tar_path.name}  ({size_mb:.1f} MB)")
    return tar_path
