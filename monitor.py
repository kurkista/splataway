#!/usr/bin/env python3
"""
monitor.py — live terminal dashboard for the splataway pipeline.

Usage:
  python3 monitor.py              # auto-discovers most recent project
  python3 monitor.py my_scene     # watch a specific project by name

Press Ctrl+C to quit.
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from rich.columns import Columns
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("Run:  /Users/scan/Claude/Gaussian_Splat/.venv/bin/pip install rich")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECTS    = SCRIPT_DIR / "projects"
LOGS_DIR    = SCRIPT_DIR / "logs"
CONFIG_FILE = SCRIPT_DIR / "config.toml"


def _load_iters() -> int:
    try:
        import tomllib  # type: ignore
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return 7000
    try:
        with open(CONFIG_FILE, "rb") as f:
            return int(tomllib.load(f)["opensplat"]["iterations"])
    except Exception:
        return 7000

# ── Pipeline metadata ─────────────────────────────────────────────────────────
STEPS = ["frames", "features", "matching", "mapping", "train"]

STEP_LABELS = {
    "frames":   "Frame extraction",
    "features": "COLMAP  feature extraction",
    "matching": "COLMAP  feature matching",
    "mapping":  "COLMAP  SfM mapping",
    "train":    "OpenSplat  3DGS training",
}

# Substrings that appear in the timestamped command lines written by run()
STEP_CMD_MARKERS = {
    "frames":   ["ffmpeg"],
    "features": ["colmap feature_extractor"],
    "matching": ["colmap sequential_matcher", "colmap exhaustive_matcher"],
    "mapping":  ["colmap mapper"],
    "train":    ["opensplat"],
}

# ── Log discovery ─────────────────────────────────────────────────────────────

def find_active_project() -> Optional[str]:
    """Return the name of the most recently active project."""
    candidates = []
    for run_log in PROJECTS.glob("*/run.log"):
        candidates.append((run_log.stat().st_mtime, run_log.parent.name))
    if not candidates:
        return None
    return max(candidates)[1]


def project_log_path(name: str) -> Optional[Path]:
    p = PROJECTS / name / "run.log"
    return p if p.exists() else None

# ── Log parsing ───────────────────────────────────────────────────────────────

_CMD_TS_RE  = re.compile(r'^\[(\d{4}-\d{2}-\d{2}T[\d:.]+)\] (.+)')
_FEAT_RE    = re.compile(r'Processed file \[(\d+)/(\d+)\]')
_MAP_REG_RE = re.compile(r'num_reg_frames=(\d+)')
# OpenSplat iteration formats (in priority order):
#   "Step 100/7000"  or  "Iter 100/7000"  (explicit total)
#   "Step 100: 0.123 (45%)"               (percentage, no total — common OpenSplat CPU output)
#   "[100/7000]"                           (bracket form)
_TRAIN_RE      = re.compile(r'(?:[Ss]tep[:\s]+(\d+)[/\s]+(\d+)|[Ii]ter[:\s]+(\d+)[/\s]+(\d+)|\[(\d+)/(\d+)\])')
_TRAIN_PCT_RE  = re.compile(r'^Step (\d+):\s+[\d.]+\s+\((\d+)%\)')
_COLMAP_PREFIX = re.compile(r'^[IWE]\d{8}\s+\S+\s+\S+\]\s*')


def parse_log(log_path: Path) -> dict:
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return {}

    lines = text.splitlines()
    step_start: dict[str, datetime] = {}
    step_latest_start: dict[str, datetime] = {}   # most recent restart of each step
    ordered_seen: list[str] = []
    features_progress: Optional[tuple[int, int]] = None
    train_progress:    Optional[tuple[int, int]] = None
    mapping_count:     int = 0
    done = False
    failed = False

    active_step = None
    last_cmd_line_idx = 0

    for idx, line in enumerate(lines):
        # ── Timestamped command lines ────────────────────────────────────────
        m = _CMD_TS_RE.match(line)
        if m:
            ts_raw, cmd = m.group(1), m.group(2)
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                ts = None
            for step, markers in STEP_CMD_MARKERS.items():
                if any(mk in cmd for mk in markers):
                    if step not in step_start:
                        step_start[step] = ts
                        ordered_seen.append(step)
                    # Always track the most recent invocation (handles restarts)
                    step_latest_start[step] = ts
                    # Reset progress counters on restart
                    if step == "train":
                        train_progress = None
                    elif step == "features":
                        features_progress = None
                    active_step = step
                    last_cmd_line_idx = idx
                    break
            continue

        # ── Progress signals — only within the correct step ──────────────────
        if active_step == "features":
            mf = _FEAT_RE.search(line)
            if mf:
                features_progress = (int(mf.group(1)), int(mf.group(2)))

        if active_step == "mapping":
            mm = _MAP_REG_RE.search(line)
            if mm:
                mapping_count = int(mm.group(1))

        if active_step == "train":
            # Prefer explicit-total formats first
            mt = _TRAIN_RE.search(line)
            if mt:
                g = mt.groups()
                pairs = [(g[0], g[1]), (g[2], g[3]), (g[4], g[5])]
                for a, b in pairs:
                    if a is not None and b is not None:
                        train_progress = (int(a), int(b))
                        break
            else:
                # "Step N: loss (P%)" — derive absolute step from config total
                mp = _TRAIN_PCT_RE.match(line)
                if mp:
                    step_n = int(mp.group(1))
                    total_iters = _load_iters()
                    train_progress = (step_n, total_iters)

        # ── Completion signals ───────────────────────────────────────────────
        if "Done." in line and "Output" in line:
            done = True
        if "FAILED:" in line or ("ERROR:" in line and "command exited" in line):
            failed = True

    # Determine current step and which are complete
    current_step = ordered_seen[-1] if ordered_seen else None
    if current_step:
        current_idx = STEPS.index(current_step)
        steps_done = [s for s in STEPS[:current_idx] if s in ordered_seen]
    else:
        steps_done = []

    if done:
        steps_done = list(STEPS)
        current_step = None

    # Compute step end times (= next step's start time)
    step_end: dict[str, Optional[datetime]] = {}
    for i, step in enumerate(ordered_seen[:-1]):
        step_end[step] = step_start.get(ordered_seen[i + 1])

    # Show output lines from the current (most recent) command onward
    recent_output = lines[last_cmd_line_idx + 1:] if last_cmd_line_idx else lines[-10:]

    return {
        "step_start":        step_start,
        "step_latest_start": step_latest_start,   # for ETA on restarted steps
        "step_end":          step_end,
        "steps_done":        steps_done,
        "current_step":      current_step,
        "features_progress": features_progress,
        "train_progress":    train_progress,
        "mapping_count":     mapping_count,
        "last_lines":        recent_output,
        "done":              done,
        "failed":            failed,
    }

# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_elapsed(start: Optional[datetime], end: Optional[datetime] = None) -> str:
    if start is None:
        return ""
    delta = (end or datetime.now()) - start
    s = int(delta.total_seconds())
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def progress_bar(current: int, total: int, width: int = 24) -> str:
    if total == 0:
        return ""
    pct = min(current / total, 1.0)
    filled = int(width * pct)
    return "█" * filled + "░" * (width - filled)


def eta_str(start: Optional[datetime], current: int, total: int) -> str:
    if start is None or current == 0:
        return ""
    elapsed = (datetime.now() - start).total_seconds()
    rate = current / elapsed
    remaining = (total - current) / rate
    m, s = divmod(int(remaining), 60)
    return f"~{m}m{s:02d}s left"

# ── Display builder ───────────────────────────────────────────────────────────

def build_panel(project: str, state: dict) -> Panel:
    now = datetime.now()
    step_start        = state.get("step_start", {})
    step_latest_start = state.get("step_latest_start", {})
    step_end          = state.get("step_end", {})
    steps_done        = state.get("steps_done", [])
    current           = state.get("current_step")

    # ── Overall status ────────────────────────────────────────────────────────
    # Determine run_start for the header clock.
    # If there's a large gap between the earliest and latest step starts (e.g.
    # pipeline resumed overnight), the min would show yesterday's timestamp.
    # Instead: if max − min > 3 h, treat the latest step start as the run start
    # (this invocation). For a fresh end-to-end run the gap is small so min
    # gives the correct total elapsed.
    if step_latest_start:
        ts_vals  = list(step_latest_start.values())
        ts_min   = min(ts_vals)
        ts_max   = max(ts_vals)
        gap_h    = (ts_max - ts_min).total_seconds() / 3600
        run_start = ts_max if gap_h > 3 else ts_min
    elif step_start:
        run_start = min(step_start.values())
    else:
        run_start = None
    total_elapsed = fmt_elapsed(run_start)

    if state.get("done"):
        status_text = Text("  ✓  Complete", style="bold green")
        border_style = "green"
    elif state.get("failed"):
        status_text = Text("  ✗  Stopped", style="bold white")
        border_style = "white"
    elif current:
        status_text = Text("  ⟳  Running", style="bold yellow")
        border_style = "yellow"
    else:
        status_text = Text("  ◌  Waiting", style="dim")
        border_style = "blue"

    # ── Step table ────────────────────────────────────────────────────────────
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(width=3, no_wrap=True)   # icon
    tbl.add_column(width=35, no_wrap=True)  # label + index
    tbl.add_column(width=8,  no_wrap=True)  # elapsed
    tbl.add_column()                         # progress / status

    for i, step in enumerate(STEPS, 1):
        label   = f"[{i}/5]  {STEP_LABELS[step]}"
        start   = step_start.get(step)
        end     = step_end.get(step)

        if step in steps_done:
            icon    = Text("✓", style="bold green")
            elapsed = Text(fmt_elapsed(start, end), style="dim green")
            detail  = Text("done", style="dim green")

        elif step == current:
            icon    = Text("●", style="bold yellow blink")
            elapsed = Text(fmt_elapsed(start), style="yellow")

            # Step-specific live progress
            if step == "features":
                prog = state.get("features_progress")
                if prog:
                    cur, tot = prog
                    bar = progress_bar(cur, tot)
                    detail = Text(f"{bar}  {cur}/{tot}", style="cyan")
                else:
                    detail = Text("extracting features…", style="yellow")

            elif step == "train":
                prog = state.get("train_progress")
                if prog:
                    cur, tot = prog
                    bar = progress_bar(cur, tot)
                    # Use most recent restart time so ETA reflects current run
                    train_start = step_latest_start.get("train") or start
                    eta = eta_str(train_start, cur, tot)
                    detail = Text(f"{bar}  {cur}/{tot}  {eta}", style="cyan")
                else:
                    detail = Text("initialising…", style="yellow")

            elif step == "mapping":
                n = state.get("mapping_count", 0)
                detail = Text(
                    f"{'▓' * min(n, 24)}  {n} registered" if n else "running…",
                    style="cyan"
                )

            else:
                detail = Text("running…", style="yellow")

        elif state.get("failed") and start is not None and step not in steps_done:
            # Step was attempted but didn't complete
            icon    = Text("✗", style="bold white")
            elapsed = Text("")
            detail  = Text("—", style="dim")

        else:
            icon    = Text("○", style="dim")
            elapsed = Text("")
            detail  = Text("", style="dim")

        tbl.add_row(icon, Text(label, style="bold" if step == current else ""), elapsed, detail)

    # ── Log tail ──────────────────────────────────────────────────────────────
    raw_lines = state.get("last_lines", [])
    clean = []
    for ln in raw_lines:
        ln = _COLMAP_PREFIX.sub("", ln).strip()
        if ln:
            clean.append(ln)
    log_text = "\n".join(clean[-6:])

    # ── Compose ──────────────────────────────────────────────────────────────
    body = Table.grid(padding=(0, 0))
    body.add_row(tbl)
    body.add_row(Text(""))
    body.add_row(Text("─" * 70, style="dim"))
    body.add_row(Text(log_text, style="dim", overflow="fold"))

    title = f"  splataway  ·  {project}  ·  {total_elapsed}  " if total_elapsed else f"  splataway  ·  {project}  "

    return Panel(
        body,
        title=Text(title, style="bold white"),
        subtitle=status_text,
        border_style=border_style,
        padding=(1, 2),
    )


def idle_panel() -> Panel:
    return Panel(
        Text(
            "\n  No active project found.\n\n"
            "  Drop a video or image folder into inbox/\n"
            "  or run:  python3 splat.py <input>\n",
            style="dim",
        ),
        title="  splataway monitor  ",
        border_style="dim",
        padding=(1, 3),
    )

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Optional: explicit project name as CLI arg
    target_project = sys.argv[1] if len(sys.argv) > 1 else None

    console = Console()

    with Live(console=console, refresh_per_second=0.5, screen=False) as live:
        while True:
            project = target_project or find_active_project()

            if project is None:
                live.update(idle_panel())
            else:
                log_path = project_log_path(project)
                if log_path is None:
                    live.update(idle_panel())
                else:
                    state = parse_log(log_path)
                    live.update(build_panel(project, state))

            time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
