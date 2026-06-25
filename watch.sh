#!/usr/bin/env bash
# watch.sh — triggered by launchd WatchPaths whenever inbox/ changes.
# Processes all pending items: video files and image folders.
# Input is archived on success; moved to archive/failed/ on error.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Ensure Homebrew tools (ffmpeg, colmap) are on PATH in non-interactive shells
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:$PATH"

INBOX="$ROOT/inbox"
ARCHIVE="$ROOT/archive"
OUTPUT="$ROOT/output"
LOGS="$ROOT/logs"
LOCK="$ROOT/.watch.lock"
PYTHON="$ROOT/.venv/bin/python3"

# Fall back to system python if venv isn't built yet (shouldn't happen post-install)
[ -x "$PYTHON" ] || PYTHON="python3"

VIDEO_EXTS_RE='^(mp4|mov|mts|avi|mkv|m4v)$'

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p "$LOGS"
WATCHER_LOG="$LOGS/watcher.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHER_LOG"
}

# ── Lock — prevent overlapping runs ──────────────────────────────────────────
if [ -f "$LOCK" ]; then
    log "Already running (lock exists). Exiting."
    exit 0
fi
touch "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# ── Wait for a file or folder to finish copying (size stability check) ────────
wait_stable() {
    local path="$1"
    local prev=-1
    local curr
    log "Waiting for copy of $(basename "$path") to stabilise..."
    while true; do
        curr=$(du -sk "$path" 2>/dev/null | cut -f1 || echo 0)
        if [ "$curr" -eq "$prev" ] && [ "$curr" -gt 0 ]; then
            break
        fi
        prev=$curr
        sleep 3
    done
    log "Copy stable (${curr}KB)."
}

# ── Convert filename stem to safe slug ────────────────────────────────────────
slugify() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr -cd '[:alnum:]_-'
}

# ── Auto-group loose image files ──────────────────────────────────────────────
# If images are dropped directly into inbox/ (not wrapped in a folder), gather
# them into a subfolder so the pipeline treats them as a single project.
IMAGE_EXTS_RE='^(jpg|jpeg|png|tif|tiff|webp|dng)$'
loose_images=()
while IFS= read -r -d '' f; do
    ext="${f##*.}"
    if echo "$ext" | grep -qiE "$IMAGE_EXTS_RE"; then
        loose_images+=("$f")
    fi
done < <(find "$INBOX" -maxdepth 1 -type f -print0 2>/dev/null | sort -z)

if [ ${#loose_images[@]} -gt 0 ]; then
    # Derive folder name from first filename: strip trailing _NNNN and date sequences
    first_stem="$(basename "${loose_images[0]}" | sed 's/\.[^.]*$//')"
    folder_name="$(echo "$first_stem" | sed -E 's/[_-]*[0-9]+([_-][0-9]+)*$//')"
    [ -z "$folder_name" ] && folder_name="images"
    folder_name="$(slugify "$folder_name")_$(date +%Y%m%d)"
    dest="$INBOX/$folder_name"
    mkdir -p "$dest"
    log "Auto-grouping ${#loose_images[@]} loose image files → inbox/$folder_name/"
    for f in "${loose_images[@]}"; do
        mv "$f" "$dest/"
    done
fi

# ── Scan inbox ────────────────────────────────────────────────────────────────
shopt -s nullglob
items=("$INBOX"/*)

if [ ${#items[@]} -eq 0 ]; then
    log "Inbox is empty — nothing to do."
    exit 0
fi

log "Found ${#items[@]} item(s) in inbox."

for item in "${items[@]}"; do
    stem="$(basename "$item")"
    stem_noext="${stem%.*}"
    name="$(slugify "$stem_noext")"
    [ -z "$name" ] && name="scene_$(date +%s)"

    # ── Classify input ────────────────────────────────────────────────────────
    if [ -f "$item" ]; then
        ext="${stem##*.}"
        if ! echo "$ext" | grep -qiE "$VIDEO_EXTS_RE"; then
            log "SKIP: $stem (not a recognised video format)"
            continue
        fi
        input_type="video"
    elif [ -d "$item" ]; then
        input_type="images"
    else
        log "SKIP: $stem (not a file or directory)"
        continue
    fi

    log "──────────────────────────────────────────────"
    log "Processing: $stem  →  project: $name  (${input_type})"

    wait_stable "$item"

    # ── Ensure unique output dir ──────────────────────────────────────────────
    out_dir="$OUTPUT/$name"
    # If a previous successful run exists, suffix with timestamp
    if [ -d "$out_dir" ] && [ -f "$out_dir/splat.ply" ]; then
        out_dir="${out_dir}_$(date +%Y%m%d_%H%M%S)"
        name="$(basename "$out_dir")"
        log "Previous output found — using: $out_dir"
    fi
    mkdir -p "$out_dir"

    project_log="$LOGS/${name}.log"

    # ── Run the pipeline ──────────────────────────────────────────────────────
    osascript -e "display notification \"Starting: $name\" with title \"splataway\" subtitle \"Pipeline started\"" 2>/dev/null || true

    # Use cloud GPU automatically if RUNPOD_API_KEY is set; fall back to local CPU
    CLOUD_FLAG=""
    [ -n "${RUNPOD_API_KEY:-}" ] && CLOUD_FLAG="--cloud runpod"

    if "$PYTHON" "$ROOT/splat.py" "$item" \
        --name "$name" \
        --output "$out_dir/splat.ply" \
        ${CLOUD_FLAG} \
        2>&1 | tee "$project_log"; then

        osascript -e "display notification \"Splat ready — output/$name/\" with title \"splataway\" subtitle \"✓ Complete\"" 2>/dev/null || true
        log "SUCCESS: output → $out_dir/splat.ply"

        # Copy the pipeline log into the output folder for reference
        cp "$project_log" "$out_dir/run.log" 2>/dev/null || true

        # Archive the input
        mkdir -p "$ARCHIVE/$name"
        mv "$item" "$ARCHIVE/$name/"
        log "Input archived → archive/$name/"

    else
        osascript -e "display notification \"Check logs/$name.log\" with title \"splataway\" subtitle \"✗ Pipeline failed\"" 2>/dev/null || true
        log "FAILED: $name — see $project_log"
        # Move failed input out of inbox so it doesn't re-trigger on next drop
        mkdir -p "$ARCHIVE/failed/$name"
        mv "$item" "$ARCHIVE/failed/$name/"
        log "Failed input moved → archive/failed/$name/"
    fi
done

log "Batch complete."
