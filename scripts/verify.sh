#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$ROOT/models/mamba.safetensors}"
BASE="$ROOT/models/baseline"

# MinGW runtime DLLs live in Strawberry
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) export PATH="/c/Strawberry/c/bin:$PATH" ;; esac
PY="$(command -v py || command -v python3 || command -v python)"

"$ROOT/scripts/build.sh" Release -DFLOWEDGE_VERIFY=ON -DFLOWEDGE_BENCH=OFF

if [[ ! -f "$MODEL" ]]; then
  echo "no checkpoint — generating a tiny synthetic one (CI path)"
  MODEL="$ROOT/models/synthetic.safetensors"
  BASE="$ROOT/models/baseline_synth"
  "$PY" "$ROOT/scripts/gen_synthetic_model.py" "$MODEL"
fi

"$PY" "$ROOT/scripts/torch_ref.py" dump "$MODEL" "$BASE" # torch reference
"$ROOT/build/verify/verify" "$MODEL" "$BASE.bin"
