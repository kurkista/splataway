FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Single source of truth for OpenCV: only the apt package, no pre-existing torch headers
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    cmake libopencv-dev libglm-dev git pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch for CUDA 11.8 (pinned version)
RUN pip3 install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

RUN git clone https://github.com/pierotofy/OpenSplat.git /opensplat
WORKDIR /opensplat
# opencv_imgcodecs is missing from OpenSplat's hardcoded OpenCV_LIBS list
RUN sed -i 's/opencv_calib3d)/opencv_calib3d opencv_imgcodecs)/' CMakeLists.txt
RUN TORCH_CMAKE=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") && \
    mkdir build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DGPU_RUNTIME=CUDA \
      -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" && \
    make -j$(nproc)

ENV OPENSPLAT=/opensplat/build/opensplat
WORKDIR /workspace
