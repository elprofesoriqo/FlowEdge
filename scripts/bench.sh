#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$ROOT/models/mamba_flow.safetensors}"
BENCH="${FLOWEDGE_BUILD_DIR:-$ROOT/build}"
LATENCY_ITERS="${FLOWEDGE_LATENCY_ITERS:-100000}"
TORCH_LATENCY_ITERS="${FLOWEDGE_TORCH_LATENCY_ITERS:-$LATENCY_ITERS}"

if [[ ! -f "$MODEL" ]]; then
    echo "Downloading mamba_flow.safetensors from Hugging Face..."
    mkdir -p "$ROOT/models"
    wget -qO "$MODEL" "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors"
fi


# MinGW runtime DLLs live in Strawberry
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) export PATH="/c/Strawberry/c/bin:$PATH" ;; esac

"$ROOT/scripts/build.sh" Release -DFLOWEDGE_BENCH=ON -DFLOWEDGE_TESTS=OFF

PY="${PYTHON:-$(command -v python3 || command -v python || command -v py)}"

exe_l="$BENCH/flowedge_latency_bench"
exe_k="$BENCH/flowedge_kernels_bench"
exe_m="$BENCH/flowedge_model_latency_bench"
exe_t="$BENCH/flowedge_threaded_matmul_bench"
if [[ -f "$BENCH/flowedge_latency_bench.exe" ]]; then
    exe_l="$BENCH/flowedge_latency_bench.exe"
    exe_k="$BENCH/flowedge_kernels_bench.exe"
    exe_m="$BENCH/flowedge_model_latency_bench.exe"
    exe_t="$BENCH/flowedge_threaded_matmul_bench.exe"
fi

echo "== Action-head Real-time Latency (us) [Euler, N=10] =="
echo "Engine     | Mean (us) | p50 (us)  | p99 (us)  | p999 (us) | Min (us)  | Max (us)  | Allocs"
echo "----------------------------------------------------------------------------------------"
"$exe_l" euler "$LATENCY_ITERS"
"$PY" "$ROOT/scripts/torch_ref.py" latency euler "$TORCH_LATENCY_ITERS" || echo "PyTorch          | (requires torch/numpy)"

echo ""
echo "== kernel microbenchmarks =="
"$exe_k" --benchmark_counters_tabular=true --benchmark_color=true --benchmark_out="$BENCH/kernels.json" --benchmark_out_format=json

echo ""
echo "== threaded matmul scaling =="
"$exe_t" --benchmark_counters_tabular=true --benchmark_color=true

[[ -f "$MODEL" ]] || { echo "note: $MODEL absent — skipping end-to-end vs-PyTorch"; exit 0; }

echo "== FlowEdge (C-ABI) =="
FLOWEDGE_MODEL="$MODEL" "$BENCH/flowedge_engine_bench"

F32_MODEL="${FLOWEDGE_MODEL_F32:-$ROOT/models/mamba.safetensors}"
BF16_MODEL="${FLOWEDGE_MODEL_BF16:-$ROOT/models/mamba_bf16_bench.safetensors}"
if [[ -f "$F32_MODEL" && ! -f "$BF16_MODEL" ]] && \
   "$PY" -c "import torch, safetensors" >/dev/null 2>&1; then
    "$PY" "$ROOT/convert/convert.py" "$F32_MODEL" "$BF16_MODEL" --arch mamba --dtype bf16
fi
if [[ -f "$F32_MODEL" && -f "$BF16_MODEL" ]]; then
    echo "== Matched FP32/BF16 model latency =="
    "$exe_m" "$F32_MODEL" "$BF16_MODEL" \
      "${FLOWEDGE_MODEL_LATENCY_ITERS:-100}"
else
    echo "note: matched FP32/BF16 latency requires a convertible checkpoint and torch+safetensors"
fi

echo "== PyTorch =="
"$PY" "$ROOT/scripts/torch_ref.py" bench "$MODEL" 1 || echo "PyTorch forward: (requires torch/numpy)"
