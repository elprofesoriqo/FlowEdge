#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$ROOT/models/mamba_flow.safetensors}"
BUILD_DIR="${FLOWEDGE_BUILD_DIR:-$ROOT/build-relay}"
ITERS="${FLOWEDGE_RELAY_BENCH_ITERS:-500}"

if [[ ! -f "$MODEL" ]]; then
    echo "Downloading mamba_flow.safetensors from Hugging Face..."
    mkdir -p "$ROOT/models"
    wget -qO "$MODEL" "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors"
fi

case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) export PATH="/c/Strawberry/c/bin:$PATH" ;; esac

FLOWEDGE_BUILD_DIR="$BUILD_DIR" "$ROOT/scripts/build.sh" Release \
    -DFLOWEDGE_RELAY=ON -DFLOWEDGE_BENCH=ON -DFLOWEDGE_TESTS=OFF

EXE="$BUILD_DIR/flowedge_relay_bench"
[[ -f "$EXE.exe" ]] && EXE="$EXE.exe"

if [[ -n "${FLOWEDGE_THREADS:-}" ]]; then
    "$EXE" "$MODEL" "$ITERS" "$FLOWEDGE_THREADS"
else
    "$EXE" "$MODEL" "$ITERS"
fi
