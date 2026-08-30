FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CC=clang-23
ENV CXX=clang++-23
ENV FLOWEDGE_BUILD_DIR=/workspace/build

RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    gnupg \
    ninja-build \
    pybind11-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
      | gpg --dearmor -o /usr/share/keyrings/apt.llvm.org.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-23 main" \
      > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y clang-23 clang-format-23 clang-tidy-23 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-23 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-23 100 \
    && update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-23 100 \
    && update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-23 100 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/clang-23 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-23 100

WORKDIR /workspace

COPY . .

RUN cmake -S . -B "$FLOWEDGE_BUILD_DIR" -G Ninja \
      -DCMAKE_C_COMPILER=/usr/bin/clang-23 \
      -DCMAKE_CXX_COMPILER=/usr/bin/clang++-23 \
      -DCMAKE_BUILD_TYPE=Release \
      -DFLOWEDGE_TESTS=ON \
      -DFLOWEDGE_BENCH=ON \
      -DFLOWEDGE_PYTHON=ON \
    && cmake --build "$FLOWEDGE_BUILD_DIR" \
    && ctest --test-dir "$FLOWEDGE_BUILD_DIR" --output-on-failure

RUN python3 -m pip install --break-system-packages .

CMD ["./build/flowedge_tests"]
