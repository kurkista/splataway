#!/usr/bin/env bash
# install.sh — one-time setup for the splataway pipeline.
# Run from the project root: bash install.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

header() {
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $*"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ── 1. Homebrew dependencies ──────────────────────────────────────────────────
header "Installing Homebrew packages"
brew install ffmpeg colmap cmake opencv pkg-config

# ── 2. Python venv + PyTorch (needed only for the OpenSplat CMake build) ──────
header "Setting up Python venv with PyTorch (MPS/Metal)"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip --quiet
# macOS ARM build of PyTorch includes the MPS (Metal Performance Shaders) backend
.venv/bin/pip install torch torchvision tomli runpod paramiko --quiet

TORCH_CMAKE=$(.venv/bin/python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
echo "LibTorch cmake path: $TORCH_CMAKE"

# ── 3. Clone and build OpenSplat ─────────────────────────────────────────────
header "Cloning OpenSplat"
if [ -d OpenSplat ]; then
    echo "OpenSplat already cloned — pulling latest"
    git -C OpenSplat pull --ff-only
else
    git clone https://github.com/pierotofy/OpenSplat.git
fi

header "Building OpenSplat (takes a few minutes)"

# Metal GPU requires the full Xcode app (not just Command Line Tools).
# Without it, we build CPU-only and add --cpu automatically at runtime.
if xcrun --find metal &>/dev/null; then
    GPU_FLAG="-DGPU_RUNTIME=MPS"
    echo "  Metal compiler found — building with GPU (MPS) support."
else
    GPU_FLAG=""
    echo "  ⚠  Metal compiler not found. Building CPU-only."
    echo "     Install Xcode from the App Store, then re-run install.sh for GPU support."
fi

mkdir -p OpenSplat/build
cmake \
    -S OpenSplat \
    -B OpenSplat/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
    $GPU_FLAG \
    -DOPENSPLAT_BUILD_SIMPLE_TRAINER=ON
cmake --build OpenSplat/build --parallel "$(sysctl -n hw.logicalcpu)"

# ── 4. Directory structure ────────────────────────────────────────────────────
header "Creating directory structure"
mkdir -p inbox archive/failed output projects logs
touch projects/.gitkeep

# ── 5. Make scripts executable ────────────────────────────────────────────────
chmod +x watch.sh

# ── 6. Install launchd plist (folder watcher) ─────────────────────────────────
header "Installing launchd watcher"

PLIST_LABEL="com.splataway.watcher"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$PLIST_DIR/$PLIST_LABEL.plist"

mkdir -p "$PLIST_DIR"

cat > "$PLIST_PATH" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${ROOT}/watch.sh</string>
    </array>

    <!-- Fire whenever inbox/ changes -->
    <key>WatchPaths</key>
    <array>
        <string>${ROOT}/inbox</string>
    </array>

    <!-- Don't run at login — only fire on inbox changes -->
    <key>RunAtLoad</key>
    <false/>

    <!-- Log watcher activity -->
    <key>StandardOutPath</key>
    <string>${ROOT}/logs/watcher.log</string>
    <key>StandardErrorPath</key>
    <string>${ROOT}/logs/watcher.log</string>
</dict>
</plist>
PLIST

# Unload first if already loaded (idempotent re-install)
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"
echo "Launchd watcher registered: $PLIST_LABEL"

# ── 7. Smoke test ─────────────────────────────────────────────────────────────
header "Verifying tools"
echo -n "  ffmpeg    : "; ffmpeg -version 2>&1 | head -1
echo -n "  colmap    : "; colmap help 2>&1 | head -1
echo -n "  opensplat : "
if ./OpenSplat/build/opensplat --help 2>&1 | grep -iqE 'usage|opensplat|options'; then
    echo "OK"
else
    echo "WARNING — check build output above"
fi
echo -n "  watcher   : "; launchctl list | grep "$PLIST_LABEL" | awk '{print "loaded (PID "$1")"}' || echo "loaded (idle)"

header "Setup complete"
cat << 'EOF'

  Drop a video file or image folder into:
    inbox/

  The watcher fires automatically. When done:
    output/{name}/splat.ply   ← your Gaussian Splat
    archive/{name}/           ← your original input
    logs/{name}.log           ← full pipeline log

  Manual run (bypass watcher):
    python3 splat.py <video_or_folder> [--name X] [--iters 7000]

  Quick preview:       --iters 3000
  Maximum quality:     --iters 30000
  Unordered photos:   --matcher exhaustive
  Resume after crash: --from-step matching

  View result at:  https://superspl.at/editor
                   (drag & drop the .ply)

EOF
