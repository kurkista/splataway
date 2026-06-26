FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    cmake libopencv-dev libglm-dev git pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/pierotofy/OpenSplat.git /opensplat
WORKDIR /opensplat
# OpenSplat's CMakeLists.txt hardcodes OpenCV_LIBS without opencv_imgcodecs.
# Use a full absolute path so the linker gets it regardless of cmake target resolution.
RUN sed -i \
    's|set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_calib3d)|set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_calib3d /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so)|' \
    CMakeLists.txt && grep "OpenCV_LIBS" CMakeLists.txt
RUN mkdir build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DGPU_RUNTIME=CUDA \
      -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch && \
    echo "=== OpenCV in cache ===" && grep -i "OpenCV_DIR\|OpenCV_LIB" CMakeCache.txt | head -10 && \
    echo "=== imgcodecs file ===" && ls -la /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs* && \
    echo "=== link cmd ===" && grep -o 'opencv[^ ]*' CMakeFiles/opensplat.dir/link.txt | sort -u && \
    make -j$(nproc)

ENV OPENSPLAT=/opensplat/build/opensplat
WORKDIR /workspace
