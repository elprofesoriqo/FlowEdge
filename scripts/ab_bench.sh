#!/usr/bin/env bash
# Compare kernel performance against a Git ref without touching the caller's checkout.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE_REF="main"
FILTER=""
RUNS=9
THRESHOLD=5
RENAMES=()
ALLOWED_REMOVED=()
BACKEND="${FLOWEDGE_BACKEND:-cpu}"
RESULT_ROOT="${FLOWEDGE_BENCH_RESULTS_DIR:-$ROOT/bench-results}"

usage() {
  echo "usage: $0 [--baseline-ref REF] [--filter REGEX] [--runs N] [--threshold PERCENT] [--rename OLD=NEW] [--allow-removed REGEX]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline-ref) BASELINE_REF="$2"; shift 2 ;;
    --filter) FILTER="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --rename) RENAMES+=(--rename "$2"); shift 2 ;;
    --allow-removed) ALLOWED_REMOVED+=(--allow-removed "$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
[[ "$RUNS" =~ ^[0-9]+$ ]] && ((10#$RUNS >= 3)) || {
  echo "--runs must be an integer >= 3" >&2
  exit 2
}
[[ "$THRESHOLD" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "--threshold must be non-negative" >&2; exit 2; }

cd "$ROOT"
BASELINE_COMMIT="$(git rev-parse --verify "${BASELINE_REF}^{commit}")"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$RESULT_ROOT/$STAMP"
CURRENT_BUILD="$OUT/current-build"
BASELINE_BUILD="$OUT/baseline-build"
BASELINE_SOURCE="$(mktemp -d "${TMPDIR:-/tmp}/flowedge-benchmark.XXXXXX")"
mkdir -p "$OUT"

cleanup() {
  git worktree remove --force "$BASELINE_SOURCE" >/dev/null 2>&1 || rm -rf "$BASELINE_SOURCE"
}
trap cleanup EXIT
git worktree add --detach "$BASELINE_SOURCE" "$BASELINE_COMMIT" >/dev/null

capture() {
  local source=$1 build=$2 output=$3
  local cmake_args=()
  if command -v clang++ >/dev/null; then
    cmake_args+=(-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++)
    case "$(uname -s)" in
      MINGW* | MSYS* | CYGWIN*)
        cmake_args+=(-DCMAKE_C_COMPILER_TARGET=x86_64-w64-mingw32
                     -DCMAKE_CXX_COMPILER_TARGET=x86_64-w64-mingw32) ;;
    esac
  fi
  cmake -S "$source" -B "$build" -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_BENCH=ON \
    -DFLOWEDGE_BACKEND="$BACKEND" "${cmake_args[@]}"
  cmake --build "$build" --config Release -j
  local executable="$build/flowedge_kernels_bench"
  [[ -x "$build/flowedge_kernels_bench.exe" ]] && executable="$build/flowedge_kernels_bench.exe"
  [[ -x "$build/bench/flowedge_kernels_bench" ]] && executable="$build/bench/flowedge_kernels_bench"
  [[ -x "$build/bench/flowedge_kernels_bench.exe" ]] && executable="$build/bench/flowedge_kernels_bench.exe"
  [[ -x "$executable" ]] || { echo "kernel benchmark executable not found in $build" >&2; exit 1; }
  local args=(--benchmark_repetitions="$RUNS" --benchmark_report_aggregates_only=true
              --benchmark_out="$output" --benchmark_out_format=json)
  [[ -n "$FILTER" ]] && args+=(--benchmark_filter="$FILTER")
  "$executable" "${args[@]}"
}

capture "$BASELINE_SOURCE" "$BASELINE_BUILD" "$OUT/baseline-kernels.json"
capture "$ROOT" "$CURRENT_BUILD" "$OUT/candidate-kernels.json"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || command -v py)}"
"$PYTHON" "$ROOT/scripts/benchmark_regression.py" \
  --baseline "$OUT/baseline-kernels.json" \
  --candidate "$OUT/candidate-kernels.json" \
  --threshold "$THRESHOLD" \
  "${RENAMES[@]}" \
  "${ALLOWED_REMOVED[@]}" \
  --report "$OUT/report.md"

echo "Artifacts: $OUT"
