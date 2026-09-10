#!/usr/bin/env bash
# Complete local release verification. Set FLOWEDGE_VERIFY_BENCH_ITERS to shorten/extend benches.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$ROOT/models/mamba_flow.safetensors}"
BUILD_DIR="${FLOWEDGE_BUILD_DIR:-$ROOT/build-all}"
mkdir -p "$BUILD_DIR"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
INSTALL_DIR="$BUILD_DIR/install-check"
CONSUMER_SUFFIX=native
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) CONSUMER_SUFFIX=mingw ;; esac
CONSUMER_DIR="${FLOWEDGE_INSTALL_CONSUMER_DIR:-$BUILD_DIR/install-consumer-$CONSUMER_SUFFIX}"
BENCH_ITERS="${FLOWEDGE_VERIFY_BENCH_ITERS:-500}"

case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) export PATH="/c/Strawberry/c/bin:$PATH" ;; esac

resolve_executable() {
  local name=$1
  local candidate
  for candidate in "$BUILD_DIR/$name" "$BUILD_DIR/$name.exe" \
                   "$BUILD_DIR/src/relay/$name" "$BUILD_DIR/src/relay/$name.exe"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if [[ ! -f "$MODEL" ]]; then
  mkdir -p "$(dirname "$MODEL")"
  URL="https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors"
  if command -v wget >/dev/null; then
    wget -qO "$MODEL" "$URL"
  elif command -v curl >/dev/null; then
    curl -fsSL "$URL" -o "$MODEL"
  else
    echo "wget or curl is required to download the sample checkpoint" >&2
    exit 1
  fi
fi

PYTHON="${PYTHON:-$(command -v python3 || command -v python || command -v py || true)}"
PYTHON_OPTION=OFF
if [[ "${FLOWEDGE_VERIFY_PYTHON:-auto}" != 0 && -n "$PYTHON" ]] &&
   "$PYTHON" -c 'import sys' >/dev/null 2>&1; then
  PYTHON_OPTION=ON
fi

FLOWEDGE_BUILD_DIR="$BUILD_DIR" "$ROOT/scripts/build.sh" Release \
  -DFLOWEDGE_RELAY=ON -DFLOWEDGE_TESTS=ON -DFLOWEDGE_BENCH=ON \
  -DFLOWEDGE_PYTHON="$PYTHON_OPTION"
ctest --test-dir "$BUILD_DIR" --output-on-failure

if [[ -n "$PYTHON" ]]; then
  (cd "$ROOT" && "$PYTHON" scripts/test_benchmark_artifact.py)
  "$PYTHON" "$ROOT/scripts/report_budgets.py" \
    --build-dir "$BUILD_DIR" --model "$MODEL" \
    --budget "$ROOT/bench/budgets.json" --report "$BUILD_DIR/budget-report.md"
else
  echo "note: Python unavailable; skipping size/setup budget report"
fi

"$(resolve_executable flow_sample)" "$MODEL" euler 6
"$(resolve_executable flowedge-inspect)" "$MODEL" --json > "$BUILD_DIR/checkpoint-inspect.json"
grep -q '"deployment_profile":' "$BUILD_DIR/checkpoint-inspect.json"
"$(resolve_executable flow_sample)" "$MODEL" heun 6
"$(resolve_executable flow_sample)" "$MODEL" rk4 6
"$(resolve_executable mamba_forward)" "$MODEL"
"$(resolve_executable external_flow_sample)" "$MODEL"
"$(resolve_executable streaming_snapshot)" "$MODEL"
"$(resolve_executable cooperative_job_sample)"
"$(resolve_executable mamba_relay_stream)" "$MODEL"
"$(resolve_executable action_delivery_sample)"
JOB_TRACE="$BUILD_DIR/generic-job.trace"
rm -f "$JOB_TRACE"
"$(resolve_executable routed_job_sample)" "$JOB_TRACE"
"$(resolve_executable flowedge-relay-trace)" inspect "$JOB_TRACE" --jsonl

"$(resolve_executable flowedge_relay_bench)" "$MODEL" "$BENCH_ITERS" 0
"$(resolve_executable flowedge_relay_pool_bench)" "$MODEL" "$BENCH_ITERS" \
  "${FLOWEDGE_RELAY_POOL_WORKERS:-2}" "${FLOWEDGE_RELAY_POOL_THREADS:-0}"
"$(resolve_executable flowedge_cooperative_job_bench)" "$((BENCH_ITERS * 10))"
"$(resolve_executable flowedge_job_queue_bench)" "$BENCH_ITERS"
"$(resolve_executable flowedge_job_qos_bench)" "$((BENCH_ITERS * 1000))"
"$(resolve_executable flowedge_mamba_stream_bench)" "$MODEL" "$BENCH_ITERS"
"$(resolve_executable flowedge_worker_drain_bench)" "$((BENCH_ITERS * 10))"
"$(resolve_executable flowedge_action_delivery_bench)" "$((BENCH_ITERS * 1000))"
FLOWEDGE_BUILD_DIR="$BUILD_DIR" "$ROOT/scripts/relay_demo.sh" "$MODEL"
FLOWEDGE_BUILD_DIR="$BUILD_DIR" "$ROOT/scripts/job_demo.sh" "$MODEL"

cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR"
CONSUMER_CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_PREFIX_PATH="$INSTALL_DIR"
)
case "$(uname -s)" in
MINGW* | MSYS* | CYGWIN*)
  CONSUMER_CMAKE_ARGS+=(
    -DCMAKE_CXX_COMPILER=clang++
    -DCMAKE_CXX_COMPILER_TARGET=x86_64-w64-mingw32
  )
  ;;
esac
cmake -S "$ROOT/test/install_consumer" -B "$CONSUMER_DIR" "${CONSUMER_CMAKE_ARGS[@]}"
cmake --build "$CONSUMER_DIR" --config Release -j
CONSUMER="$CONSUMER_DIR/flowedge_install_consumer"
[[ -f "$CONSUMER.exe" ]] && CONSUMER="$CONSUMER.exe"
"$CONSUMER"

if [[ "$PYTHON_OPTION" == ON ]]; then
  export PYTHONPATH="$BUILD_DIR${PYTHONPATH:+:$PYTHONPATH}"
  if "$PYTHON" -c 'import numpy, safetensors' >/dev/null 2>&1; then
    "$PYTHON" "$ROOT/scripts/verify_external_head.py" "$BUILD_DIR"
  else
    echo "note: Python extension built; numpy+safetensors verification dependencies unavailable"
  fi
  if "$PYTHON" -c 'import numpy, torch, safetensors' >/dev/null 2>&1; then
    (cd "$ROOT" && "$PYTHON" scripts/verify_ulp.py "$MODEL")
    (cd "$ROOT" && "$PYTHON" scripts/verify_diffusion.py "$BUILD_DIR")
  else
    echo "note: PyTorch ULP verification dependencies unavailable"
  fi
else
  echo "note: no runnable native Python; C++ release surface fully verified"
fi

echo "FlowEdge complete verification passed"
