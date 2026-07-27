FROM ubuntu:24.04

# Non-interactive apt
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies: Clang 18, CMake, Python 3.12, Git
RUN apt-get update && apt-get install -y \
    build-essential \
    clang-18 \
    cmake \
    git \
    python3.12 \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set clang-18 as default
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100

WORKDIR /workspace

# Copy source
COPY . .

# Build standard release
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_PYTHON=ON \
    && cmake --build build --config Release -j$(nproc)

# Install python bindings system-wide
RUN pip install --break-system-packages .

# Default command: run inference
CMD ["./build/flow_sample", "models/mamba_flow.safetensors", "euler"]
