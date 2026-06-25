# splataway

Automated 3D Gaussian Splat pipeline for Apple Silicon Macs.  
Drop a video or image folder into `inbox/` — get a `.ply` you can fly through in a browser.

```
37 images  →  COLMAP (5 min)  →  OpenSplat (3–8 h CPU / 20 min GPU)  →  splat.ply
```

> **Status:** CPU training works end-to-end. Metal/GPU training is blocked by a bug in
> Xcode 26.5 / macOS 26 — the Metal compiler fails to link. Cloud GPU workaround is
> on the roadmap (`feature/cloud-gpu`).

---

## What you get

A `.ply` Gaussian Splat file you can:

- **View instantly** — drag onto [superspl.at](https://superspl.at) (free, browser, no install)
- **Import into Unity** — via [aras-p/UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting)
- **Embed on the web** — Three.js has native Gaussian Splat support
- **Share** — superspl.at generates a public link

---

## Requirements

- Apple Silicon Mac (M1 or later), macOS 13+
- Xcode Command Line Tools: `xcode-select --install`
- Homebrew: [brew.sh](https://brew.sh)
- ~10 GB free disk space for the build

---

## One-time setup

```bash
git clone https://github.com/kurkista/splataway
cd splataway
bash install.sh
```

Installs: `ffmpeg`, `colmap` (Homebrew), Python venv with PyTorch, OpenSplat built from source.  
Registers a launchd agent that watches `inbox/` and fires automatically on file drop.

---

## Usage

### Automatic (hot folder)

Drop a file or folder into `inbox/`:

```
inbox/my_scene.mp4       ← video
inbox/my_scene/          ← folder of JPEGs/PNGs
```

The launchd agent triggers, runs the full pipeline, and puts the result in `output/my_scene/splat.ply`.  
You get a macOS notification at start, success, and failure.

### Manual

```bash
python3 splat.py <video_or_folder> [options]
```

```
Options:
  --name NAME         Project name (default: input filename stem)
  --iters N           Training iterations: 3000 preview / 7000 standard / 30000 max quality
  --fps N             Frames/sec to extract from video (overrides config.toml)
  --matcher TYPE      sequential (video/ordered) or exhaustive (unordered photos)
  --from-step STEP    Resume from: frames, features, matching, mapping, train
  --dry-run           Print commands without running anything
```

**Examples:**

```bash
# Full run from video
python3 splat.py inbox/castle.mp4

# Unordered photo set, higher quality
python3 splat.py inbox/sculpture/ --matcher exhaustive --iters 30000

# Resume training after a crash
python3 splat.py output/castle/ --from-step train
```

### Live monitor

Open a second terminal while a pipeline is running:

```bash
.venv/bin/python3 monitor.py
```

Shows step progress, elapsed time, ETA, and a live log tail. Auto-discovers the active project.

---

## Shooting guide

Results depend heavily on capture quality.

**For orbital video (drone or handheld walk-around):**
- Orbit at constant speed (1–2 m/s). Avoid fast pans — motion blur destroys features.
- Cover multiple heights: low pass, mid, high/bird's-eye. One orbit gives mediocre results. Three is good.
- Aim for ~70% overlap between consecutive frames. `fps = 2` is right for normal orbit speed.

**For photo sets:**
- 80–150 images for a mid-size scene, 40+ for a small object.
- Use `--matcher exhaustive` for unordered photos.
- Shoot in overcast light — hard shadows confuse depth estimation.

**Avoid:** shiny surfaces, glass, water, transparent objects — Gaussians can't reconstruct specular reflections reliably.

---

## Configuration

Edit `config.toml` to change defaults:

```toml
[ffmpeg]
fps     = 2      # frames/sec from video; 1 for slow drone, 3 for fast handheld
quality = 1      # JPEG quality (1 = best)

[colmap]
matcher       = "sequential"   # or "exhaustive" for unordered photos
single_camera = true           # false for mixed-camera shoots

[opensplat]
iterations        = 7000    # 3000 quick preview / 7000 standard / 30000 max
sh_degree         = 1       # see CPU stability note below
reset_alpha_every = 300     # see CPU stability note below
num_downscales    = 2       # progressive resolution schedule
```

---

## CPU stability notes

On CPU, OpenSplat's default settings cause NaN loss divergence mid-training. Three root causes, all fixed in this repo:

| Problem | Default | This repo | Why |
|---|---|---|---|
| SH degree | 3 | 1 | Degree 3 causes numerical explosion on CPU above ~3000 steps |
| Opacity reset | every 30 refinements (~step 3400) | every 300 | Reset cascades to NaN on CPU |
| libomp collision | — | `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` | PyTorch and OpenSplat both bundle libomp; without this, mutex init crashes |

These settings cost nothing in quality for runs under 30k iterations with good scene coverage.  
When Metal GPU support is restored, `sh_degree = 3` and default `reset_alpha_every` will give richer colour.

---

## Project layout

```
splataway/
├── splat.py          Pipeline driver
├── monitor.py        Live terminal dashboard
├── watch.sh          launchd-triggered inbox processor
├── install.sh        One-time setup
├── config.toml       Default settings
│
├── inbox/            Drop files here (gitignored)
├── output/           splat.ply output per project (gitignored)
├── archive/          Original input preserved after success (gitignored)
└── projects/         Intermediate COLMAP/frame data (gitignored)
```

---

## Toolchain

| Tool | Source | Role |
|---|---|---|
| FFmpeg | ffmpeg.org | Video → frames |
| COLMAP | ETH Zurich | Structure from Motion (camera poses + sparse point cloud) |
| OpenSplat | DroneDB / Lugano | 3D Gaussian Splatting training |
| SuperSplat | PlayCanvas | Browser viewer — drag & drop, free |

---

## Roadmap

- [ ] `--cloud runpod` — pack COLMAP output, spin up RTX 4090, train, download `.ply` (~$0.50/run)
- [ ] Scene orientation — align world "up" with gravity before training (fixes tilted scenes)
- [ ] Floater pruning — remove out-of-bounds Gaussians post-training
- [ ] Metal GPU support — pending Apple fixing the Xcode 26.5 Metal toolchain bug

---

## License

MIT
