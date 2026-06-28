# Lessons Learned — Cloud COLMAP Pipeline

## Session: saunalautta full run (2016 images, 2025-06-28)

### What worked
- **Feature extraction on pod**: 18 min for 2016 images using CPU SIFT on Linux x86 apt COLMAP. Acceptable.
- **nohup detached execution**: Long COLMAP commands must run detached (`nohup ... &`) and be polled via fresh SSH connections. A single long-lived SSH session drops and kills the job.
- **Subsampled run (504 images)**: Produces a usable splat. Good enough for preview quality.

### What failed and why

| Error | Cause | Fix |
|---|---|---|
| Qt platform crash | Headless pod, no X display | `QT_QPA_PLATFORM=offscreen` |
| OpenGL context crash | SIFT GPU extractor needs display | `--SiftExtraction.use_gpu 0` |
| SSH exit -1 mid-run | Long SSH session drops | nohup + poll via fresh connections |
| `--max_num_descriptors` unknown | Flag missing in apt COLMAP version | Removed flag |
| vocab_tree matching too slow | 19M descriptors × 2016 images on CPU | **Unresolved — see next steps** |

### Cost reality check
- Each failed pod boot + short run: ~$0.20–$0.40
- Successful feature extraction run (18 min) before vocab_tree timeout: ~$0.70
- Full CPU vocab_tree match + mapper on 2016 images: 2+ hours → $1.40+ before training even starts
- **CPU COLMAP on pod is viable for feature extraction only. Matching/mapping on large sets requires GPU.**

### Practical limits of apt COLMAP on pod
- CPU-only (no CUDA matching)
- Some newer flags missing (`--max_num_descriptors`)
- Feature extraction: fine (~18 min / 2016 images)
- Vocab_tree matching: too slow for >500 images

---

## Docker image: COLMAP from source (CPU, no GUI)

`kurkista/opensplat-cuda:latest` now includes COLMAP 3.9.1 compiled from source with:
- `-DCUDA_ENABLED=OFF` — GPU COLMAP in Docker requires a GPU-enabled builder; GitHub Actions and local M2 both fail nvcc compiler detection without NVIDIA drivers
- `-DGUI_ENABLED=OFF` — headless-safe, no Qt/OpenGL issues on pod
- All COLMAP flags available (no missing `--max_num_descriptors` etc.)
- No more `apt install colmap` at pod startup — COLMAP is baked in

### Why GPU COLMAP in Docker failed
Building CUDA code in a Docker image requires `nvcc` to compile a test binary during cmake's compiler detection. This fails unless the builder host has NVIDIA GPU drivers installed. Neither GitHub Actions (`ubuntu-latest`) nor local M2 Mac (QEMU emulation) qualify. A GPU-enabled self-hosted runner would solve this.

### Speed workaround for large datasets
For >500 image multi-mission sets, vocab_tree matching on CPU takes 2+ hours. Practical options:
1. Subsample to ~400 images before upload (every Nth per mission)
2. Use sequential matcher per mission + accept weaker cross-mission connections
