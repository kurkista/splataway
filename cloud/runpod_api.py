"""
cloud/runpod_api.py — RunPod pod lifecycle management.

Requires: pip install runpod paramiko
Requires: RUNPOD_API_KEY environment variable
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import paramiko
import runpod as _runpod


def _client() -> None:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("ERROR: RUNPOD_API_KEY environment variable not set.")
        print("  Get your key at: https://runpod.io/console/user/settings")
        print("  Then: export RUNPOD_API_KEY=your_key_here")
        sys.exit(1)
    _runpod.api_key = key


def create_pod(gpu: str, image: str, name: str) -> dict:
    """Spin up a pod. Returns the pod dict (id, ssh details populated after ready)."""
    _client()
    print(f"  Creating RunPod pod ({gpu})…")
    pod = _runpod.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=gpu,
        cloud_type="SECURE",
        ports="22/tcp",
        container_disk_in_gb=20,
    )
    print(f"  Pod created: {pod['id']}")
    return pod


def wait_ready(pod_id: str, timeout: int = 600) -> dict:
    """Poll until pod is RUNNING and has SSH host info. Returns updated pod dict."""
    _client()
    print(f"  Waiting for pod {pod_id} to start…", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = _runpod.get_pod(pod_id)
        if pod is None:
            time.sleep(5)
            continue
        status = pod.get("desiredStatus") or pod.get("status", "")
        if status == "RUNNING" and (pod.get("runtime") or {}).get("ports"):
            print(" ready.")
            return pod
        print(".", end="", flush=True)
        time.sleep(5)
    print()
    raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout}s")


def _ssh_connect(pod: dict) -> paramiko.SSHClient:
    """Return an open SSH connection to the pod."""
    ports = pod["runtime"]["ports"]
    ssh_port_info = next((p for p in ports if p["privatePort"] == 22), None)
    if ssh_port_info is None:
        raise RuntimeError("No SSH port found on pod")

    host = ssh_port_info["ip"]
    port = int(ssh_port_info["publicPort"])

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username="root", timeout=30)
    return client


def run_remote(pod: dict, cmd: str, log_file=None) -> int:
    """
    SSH into pod, run cmd, stream stdout/stderr line-by-line.
    Writes to log_file if provided (same file object used by splat.py's run()).
    Returns exit code.
    """
    ssh = _ssh_connect(pod)
    try:
        transport = ssh.get_transport()
        channel = transport.open_session()
        channel.set_combine_stderr(True)
        channel.exec_command(cmd)

        while True:
            if channel.recv_ready():
                data = channel.recv(4096).decode(errors="replace")
                for line in data.splitlines(keepends=True):
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    if log_file:
                        log_file.write(line)
                        log_file.flush()
            elif channel.exit_status_ready():
                # Drain any remaining output
                remaining = b""
                while channel.recv_ready():
                    remaining += channel.recv(4096)
                if remaining:
                    text = remaining.decode(errors="replace")
                    sys.stdout.write(text)
                    if log_file:
                        log_file.write(text)
                        log_file.flush()
                break
            else:
                time.sleep(0.1)

        return channel.recv_exit_status()
    finally:
        ssh.close()


def install_colmap(pod: dict) -> None:
    """Install COLMAP on the pod via apt (CPU build, much faster than Apple Silicon)."""
    print("  Installing COLMAP on pod…")
    rc = run_remote(pod, "apt-get update -qq && apt-get install -y -qq colmap 2>&1 | tail -3")
    if rc != 0:
        raise RuntimeError(f"COLMAP install failed (exit {rc})")
    print("  COLMAP ready.")


def run_colmap_remote(pod: dict, matcher: str, log_file=None) -> None:
    """
    Run COLMAP feature extraction, matching, and mapping on the pod.
    Expects images at /workspace/scene/images/.
    Produces sparse reconstruction at /workspace/scene/colmap/sparse/0/.
    """
    db   = "/workspace/scene/colmap/database.db"
    imgs = "/workspace/scene/images"
    sparse = "/workspace/scene/colmap/sparse"

    env = "QT_QPA_PLATFORM=offscreen"

    cmds = [
        f"mkdir -p {sparse}",
        (
            f"{env} colmap feature_extractor"
            f" --database_path {db}"
            f" --image_path {imgs}"
            f" --ImageReader.single_camera 1"
            f" --SiftExtraction.use_gpu 0"
        ),
    ]

    if matcher == "vocab_tree":
        vtree = "/workspace/vocab_tree.bin"
        cmds += [
            (
                f"{env} colmap vocab_tree_builder"
                f" --database_path {db}"
                f" --vocab_tree_path {vtree}"
                f" --num_visual_words 1024"
                f" --max_num_descriptors 500000"
            ),
            (
                f"{env} colmap vocab_tree_matcher"
                f" --database_path {db}"
                f" --VocabTreeMatching.vocab_tree_path {vtree}"
            ),
        ]
    else:
        cmds.append(
            f"{env} colmap {matcher}_matcher --database_path {db}"
        )

    cmds.append(
        f"{env} colmap mapper"
        f" --database_path {db}"
        f" --image_path {imgs}"
        f" --output_path {sparse}"
    )

    for cmd in cmds:
        print(f"  $ {cmd}")
        rc = run_remote(pod, cmd, log_file=log_file)
        if rc != 0:
            raise RuntimeError(f"Remote COLMAP step failed (exit {rc}): {cmd}")

    # Verify reconstruction was produced
    rc = run_remote(pod, f"test -d {sparse}/0 && echo OK || echo MISSING")
    if rc != 0:
        raise RuntimeError("COLMAP produced no reconstruction on pod (sparse/0 missing)")


def terminate_pod(pod_id: str) -> None:
    """Terminate and delete the pod."""
    _client()
    _runpod.terminate_pod(pod_id)
    print(f"  Pod {pod_id} terminated.")
