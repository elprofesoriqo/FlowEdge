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

POOL_EXE="$BUILD_DIR/flowedge_relay_pool_bench"
[[ -f "$POOL_EXE.exe" ]] && POOL_EXE="$POOL_EXE.exe"
POOL_WORKERS="${FLOWEDGE_RELAY_POOL_WORKERS:-2}"
POOL_THREADS="${FLOWEDGE_RELAY_POOL_THREADS:-0}"
"$POOL_EXE" "$MODEL" "$ITERS" "$POOL_WORKERS" "$POOL_THREADS"

QUEUE_EXE="$BUILD_DIR/flowedge_job_queue_bench"
[[ -f "$QUEUE_EXE.exe" ]] && QUEUE_EXE="$QUEUE_EXE.exe"
"$QUEUE_EXE" "${FLOWEDGE_RELAY_QUEUE_BENCH_ROUNDS:-3000}"

STREAM_EXE="$BUILD_DIR/flowedge_mamba_stream_bench"
[[ -f "$STREAM_EXE.exe" ]] && STREAM_EXE="$STREAM_EXE.exe"
"$STREAM_EXE" "$MODEL" "$ITERS"

DRAIN_EXE="$BUILD_DIR/flowedge_worker_drain_bench"
[[ -f "$DRAIN_EXE.exe" ]] && DRAIN_EXE="$DRAIN_EXE.exe"
"$DRAIN_EXE" "${FLOWEDGE_RELAY_DRAIN_BENCH_ITERS:-10000}"

ACTION_EXE="$BUILD_DIR/flowedge_action_delivery_bench"
[[ -f "$ACTION_EXE.exe" ]] && ACTION_EXE="$ACTION_EXE.exe"
"$ACTION_EXE" "${FLOWEDGE_RELAY_ACTION_BENCH_ITERS:-1000000}"
