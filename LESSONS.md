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

## Next step: GPU COLMAP in Docker

Rebuild `kurkista/opensplat-cuda:latest` to include COLMAP compiled with CUDA.
GPU feature extraction + matching cuts the COLMAP phase from hours to minutes.

See branch: `feat/cuda-colmap`

### Target Dockerfile additions
```dockerfile
# COLMAP dependencies
RUN apt-get update && apt-get install -y \
    libboost-all-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgflags-dev libsqlite3-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev \
    libcgal-dev libceres-dev && \
    rm -rf /var/lib/apt/lists/*

# Build COLMAP from source with CUDA
RUN git clone https://github.com/colmap/colmap /colmap && \
    cd /colmap && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ENABLED=ON && \
    make -j$(nproc) && make install
```

With GPU COLMAP, the routing stays the same — only the execution speed changes.
Estimated matching time with GPU: ~2–5 min for 2016 images vs 2+ hours on CPU.
