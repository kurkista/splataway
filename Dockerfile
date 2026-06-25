FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    cmake libopencv-dev libglm-dev git && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/pierotofy/OpenSplat.git /opensplat
WORKDIR /opensplat
RUN mkdir build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DGPU_RUNTIME=CUDA \
      -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch && \
    make -j$(nproc)

ENV OPENSPLAT=/opensplat/build/opensplat
WORKDIR /workspace
