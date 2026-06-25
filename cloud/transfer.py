"""
cloud/transfer.py — upload/download files to/from a RunPod pod via SFTP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import paramiko

from cloud.runpod_api import _ssh_connect


def _progress(label: str):
    def _cb(transferred: int, total: int) -> None:
        pct = int(transferred / total * 100)
        mb = transferred / 1_048_576
        total_mb = total / 1_048_576
        sys.stdout.write(f"\r  {label}: {mb:.1f}/{total_mb:.1f} MB  ({pct}%)")
        sys.stdout.flush()
        if transferred >= total:
            sys.stdout.write("\n")
    return _cb


def upload(pod: dict, local_path: Path) -> str:
    """Upload local_path to /workspace/ on the pod. Returns remote path."""
    remote_path = f"/workspace/{local_path.name}"
    ssh = _ssh_connect(pod)
    try:
        sftp = ssh.open_sftp()
        sftp.put(str(local_path), remote_path, callback=_progress("Uploading"))
        sftp.close()
    finally:
        ssh.close()
    return remote_path


def download(pod: dict, remote_path: str, local_path: Path) -> None:
    """Download remote_path from pod to local_path."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    ssh = _ssh_connect(pod)
    try:
        sftp = ssh.open_sftp()
        sftp.get(remote_path, str(local_path), callback=_progress("Downloading"))
        sftp.close()
    finally:
        ssh.close()
