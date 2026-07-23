#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$ROOT/models/mamba.safetensors}"
BENCH="$ROOT/build/bench"

# MinGW runtime DLLs live in Strawberry
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) export PATH="/c/Strawberry/c/bin:$PATH" ;; esac

"$ROOT/scripts/build.sh" Release -DFLOWEDGE_BENCH=ON -DFLOWEDGE_VERIFY=OFF

echo "== kernel microbenchmarks =="
"$BENCH/flowedge_kernels_bench" --benchmark_out="$ROOT/build/kernels.json" --benchmark_out_format=json

echo "== flow head + ODE: FlowEdge vs PyTorch =="
PY="$(command -v py || command -v python3 || command -v python)"
"$BENCH/flowedge_flow_bench"
"$PY" "$ROOT/scripts/torch_ref.py" flowbench

[[ -f "$MODEL" ]] || { echo "note: $MODEL absent — skipping end-to-end vs-PyTorch"; exit 0; }

echo "== FlowEdge (C-ABI) =="
FLOWEDGE_MODEL="$MODEL" "$BENCH/flowedge_engine_bench"

echo "== PyTorch =="
PY="$(command -v py || command -v python3 || command -v python)"
"$PY" "$ROOT/scripts/torch_ref.py" bench "$MODEL" 1
