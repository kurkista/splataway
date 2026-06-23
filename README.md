# splataway

Automated 3D Gaussian Splat pipeline for Apple Silicon (M2 Max / M1+).  
Drop a video or image folder into `inbox/` — get a `.ply` Gaussian Splat out.

## Pipeline

```
inbox/  (drop file here)
  ↓  launchd WatchPaths — fires automatically
  ↓  FFmpeg — extract frames from video
  ↓  COLMAP — Structure from Motion (camera poses + sparse point cloud)
  ↓  OpenSplat — 3DGS training on Metal GPU
output/{name}/splat.ply
archive/{name}/  (original input moved here)
```

## One-time setup

```bash
bash install.sh
```

Installs: `ffmpeg`, `colmap` (Homebrew), PyTorch (Metal/MPS), OpenSplat (built from source).  
Registers a launchd agent that watches `inbox/` and fires automatically on file drop.

## Usage

**Automatic** — just drop a file or folder into `inbox/`:
- Video file (`.mp4`, `.mov`, `.mts`, …) → frames extracted at `fps` from `config.toml`
- Image folder → images used directly

**Manual**:
```bash
python3 splat.py <video_or_folder> [options]

Options:
  --name NAME         project name (default: input filename stem)
  --fps N             frames/sec to extract from video (overrides config)
  --iters N           training iterations: 3000 preview / 7000 standard / 30000 max
  --matcher TYPE      sequential (video, default) or exhaustive (unordered photos)
  --from-step STEP    resume from: frames, features, matching, mapping, train
  --dry-run           print commands without executing
```

## Output structure

```
output/{name}/
├── splat.ply    ← Gaussian Splat (open in https://superspl.at/editor)
└── run.log      ← full pipeline log

archive/{name}/  ← original input preserved here
logs/{name}.log  ← watcher log
```

## Config

Edit `config.toml` to change defaults:

| Setting | Default | Notes |
|---|---|---|
| `ffmpeg.fps` | `2` | 1 for slow drone, 3 for fast handheld |
| `colmap.matcher` | `sequential` | Use `exhaustive` for unordered photo sets |
| `colmap.single_camera` | `true` | Set `false` for mixed-camera shoots |
| `opensplat.iterations` | `7000` | 3000 quick / 30000 max quality |

## Toolchain

| Tool | Source | License |
|---|---|---|
| FFmpeg | ffmpeg.org | LGPL |
| COLMAP | ETH Zurich | BSD |
| OpenSplat | DroneDB / Lugano | LGPL |
| SuperSplat | PlayCanvas | MIT (browser, no install) |
