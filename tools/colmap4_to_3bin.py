#!/usr/bin/env python3
"""
Convert COLMAP 4.x text output to COLMAP 3.x binary format for OpenSplat.

Usage:
  python3 tools/colmap4_to_3bin.py <sparse_txt_dir> <out_bin_dir>

Reads: cameras.txt, images.txt, points3D.txt
Writes: cameras.bin, images.bin, points3D.bin (old 3.x format OpenSplat expects)
"""

import struct
import sys
from pathlib import Path


def write_cameras_bin(txt_path: Path, out_path: Path) -> None:
    MODEL_IDS = {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 1,
        "SIMPLE_RADIAL": 2,
        "RADIAL": 3,
        "OPENCV": 4,
        "OPENCV_FISHEYE": 5,
        "FULL_OPENCV": 6,
        "FOV": 7,
        "SIMPLE_RADIAL_FISHEYE": 8,
        "RADIAL_FISHEYE": 9,
        "THIN_PRISM_FISHEYE": 10,
    }
    cameras = []
    for line in txt_path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        cam_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(p) for p in parts[4:]]
        cameras.append((cam_id, MODEL_IDS[model], width, height, params))

    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam_id, model_id, width, height, params in cameras:
            f.write(struct.pack("<I", cam_id))
            f.write(struct.pack("<i", model_id))
            f.write(struct.pack("<Q", width))
            f.write(struct.pack("<Q", height))
            for p in params:
                f.write(struct.pack("<d", p))


def write_images_bin(txt_path: Path, out_path: Path) -> None:
    images = []
    lines = [l for l in txt_path.read_text().splitlines() if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]
        i += 1
        pts2d_raw = []
        if i < len(lines):
            pts2d_raw = lines[i].split()
            i += 1
        # flat: x y point3d_id  x y point3d_id ...
        assert len(pts2d_raw) % 3 == 0
        pts2d = []
        for j in range(0, len(pts2d_raw), 3):
            pts2d.append((float(pts2d_raw[j]), float(pts2d_raw[j+1]), int(pts2d_raw[j+2])))
        images.append((img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name, pts2d))

    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name, pts2d in images:
            f.write(struct.pack("<I", img_id))
            f.write(struct.pack("<dddd", qw, qx, qy, qz))
            f.write(struct.pack("<ddd", tx, ty, tz))
            f.write(struct.pack("<I", cam_id))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", len(pts2d)))
            for x, y, pt_id in pts2d:
                f.write(struct.pack("<ddq", x, y, pt_id))


def write_points3d_bin(txt_path: Path, out_path: Path) -> None:
    points = []
    for line in txt_path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        pt_id = int(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
        error = float(parts[7])
        track = []
        for k in range(8, len(parts), 2):
            track.append((int(parts[k]), int(parts[k + 1])))
        points.append((pt_id, x, y, z, r, g, b, error, track))

    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pt_id, x, y, z, r, g, b, error, track in points:
            f.write(struct.pack("<Q", pt_id))
            f.write(struct.pack("<ddd", x, y, z))
            f.write(struct.pack("<BBB", r, g, b))
            f.write(struct.pack("<d", error))
            f.write(struct.pack("<Q", len(track)))
            for img_id, pt2d_idx in track:
                f.write(struct.pack("<II", img_id, pt2d_idx))


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)

    print(f"cameras.txt → cameras.bin", flush=True)
    write_cameras_bin(src / "cameras.txt", dst / "cameras.bin")

    print(f"images.txt → images.bin", flush=True)
    write_images_bin(src / "images.txt", dst / "images.bin")

    print(f"points3D.txt → points3D.bin", flush=True)
    write_points3d_bin(src / "points3D.txt", dst / "points3D.bin")

    print(f"Done → {dst}")


if __name__ == "__main__":
    main()
