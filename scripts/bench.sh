#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$ROOT/models/mamba_flow.safetensors}"
BENCH="$ROOT/build"

if [[ ! -f "$MODEL" ]]; then
    echo "Downloading mamba_flow.safetensors from Hugging Face..."
    mkdir -p "$ROOT/models"
    wget -qO "$MODEL" "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors"
fi


# MinGW runtime DLLs live in Strawberry
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) export PATH="/c/Strawberry/c/bin:$PATH" ;; esac

"$ROOT/scripts/build.sh" Release -DFLOWEDGE_BENCH=ON -DFLOWEDGE_TESTS=OFF

PY="$(command -v py || command -v python3 || command -v python)"

echo "== Action-head Real-time Latency (us) [Euler, N=10] =="
echo "Engine     | Mean (us) | p50 (us)  | p99 (us)  | p999 (us) | Min (us)  | Max (us)  | Allocs"
echo "----------------------------------------------------------------------------------------"
"$BENCH/flowedge_latency_bench" euler
"$PY" "$ROOT/scripts/torch_ref.py" latency

echo ""
echo "== kernel microbenchmarks =="
"$BENCH/flowedge_kernels_bench" --benchmark_counters_tabular=true --benchmark_color=true --benchmark_out="$ROOT/build/kernels.json" --benchmark_out_format=json

[[ -f "$MODEL" ]] || { echo "note: $MODEL absent — skipping end-to-end vs-PyTorch"; exit 0; }

echo "== FlowEdge (C-ABI) =="
FLOWEDGE_MODEL="$MODEL" "$BENCH/flowedge_engine_bench"

echo "== PyTorch =="
"$PY" "$ROOT/scripts/torch_ref.py" bench "$MODEL" 1
