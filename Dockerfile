FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CC=gcc-13
ENV CXX=g++-13
ENV FLOWEDGE_BUILD_DIR=/workspace/build

RUN apt-get update && apt-get install -y \
    build-essential \
    clang-18 \
    clang-format-18 \
    clang-tidy-18 \
    cmake \
    git \
    ninja-build \
    pybind11-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100 \
    && update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-18 100 \
    && update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-18 100 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100

WORKDIR /workspace

COPY . .

RUN cmake -S . -B "$FLOWEDGE_BUILD_DIR" -G Ninja \
      -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
      -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
      -DCMAKE_BUILD_TYPE=Release \
      -DFLOWEDGE_TESTS=ON \
      -DFLOWEDGE_BENCH=ON \
      -DFLOWEDGE_PYTHON=ON \
    && cmake --build "$FLOWEDGE_BUILD_DIR" \
    && ctest --test-dir "$FLOWEDGE_BUILD_DIR" --output-on-failure

RUN python3 -m pip install --break-system-packages .

CMD ["./build/flowedge_tests"]
