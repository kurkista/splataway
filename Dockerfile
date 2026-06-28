FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# PyTorch's TorchConfig.cmake sets TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
# and propagates it as INTERFACE_COMPILE_OPTIONS to all code linked against torch.
# Ubuntu's apt OpenCV uses new ABI (CXX11_ABI=1), so cv::imwrite mangles differently
# at compile vs link time → undefined reference.
# Fix: build OpenCV from source with the same old ABI so the mangled names match.

RUN apt-get update && apt-get install -y \
    cmake libglm-dev git pkg-config \
    libjpeg-dev libpng-dev libtiff-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch 4.8.0 https://github.com/opencv/opencv.git /opencv && \
    mkdir /opencv/build && cd /opencv/build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
      -DWITH_CUDA=OFF -DWITH_GTK=OFF -DWITH_QT=OFF \
      -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
      -DBUILD_LIST=core,imgproc,highgui,calib3d,imgcodecs,features2d,flann && \
    make -j$(nproc) && make install && ldconfig && \
    rm -rf /opencv

RUN git clone https://github.com/pierotofy/OpenSplat.git /opensplat
WORKDIR /opensplat
RUN sed -i 's/opencv_calib3d)/opencv_calib3d opencv_imgcodecs)/' CMakeLists.txt
RUN mkdir build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DGPU_RUNTIME=CUDA \
      -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch && \
    make -j2

# COLMAP from source — CPU, no GUI (headless-safe, all flags available)
RUN apt-get update && apt-get install -y \
    libboost-program-options-dev libboost-filesystem-dev \
    libboost-graph-dev libboost-system-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgflags-dev libsqlite3-dev \
    libglew-dev libcgal-dev libceres-dev \
    libgmp-dev libmpfr-dev libatlas-base-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch 3.9.1 https://github.com/colmap/colmap.git /colmap
WORKDIR /colmap
RUN mkdir build
WORKDIR /colmap/build
RUN cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCUDA_ENABLED=OFF \
      -DGUI_ENABLED=OFF \
      -DTESTS_ENABLED=OFF
RUN make -j2
RUN make install && ldconfig
WORKDIR /workspace
RUN rm -rf /colmap

ENV OPENSPLAT=/opensplat/build/opensplat
WORKDIR /workspace
