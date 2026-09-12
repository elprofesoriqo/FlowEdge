#!/usr/bin/env bash
# Compare kernel performance against a Git ref without touching the caller's checkout.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASELINE_REF="main"
FILTER=""
RUNS=9
THRESHOLD=5
RENAMES=()
ALLOWED_REMOVED=()
BACKEND="${FLOWEDGE_BACKEND:-cpu}"
RESULT_ROOT="${FLOWEDGE_BENCH_RESULTS_DIR:-$ROOT/bench-results}"
FETCHCONTENT_SOURCE_ROOT="${FLOWEDGE_FETCHCONTENT_SOURCE_ROOT:-}"

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
  if [[ -n "$FETCHCONTENT_SOURCE_ROOT" ]]; then
    [[ -d "$FETCHCONTENT_SOURCE_ROOT/mdspan-src" &&
       -d "$FETCHCONTENT_SOURCE_ROOT/googlebenchmark-src" ]] || {
      echo "FLOWEDGE_FETCHCONTENT_SOURCE_ROOT must contain mdspan-src and googlebenchmark-src" >&2
      exit 2
    }
    cmake_args+=("-DFETCHCONTENT_SOURCE_DIR_MDSPAN=$FETCHCONTENT_SOURCE_ROOT/mdspan-src"
                "-DFETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK=$FETCHCONTENT_SOURCE_ROOT/googlebenchmark-src")
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
  case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*)
      local powershell_args=(
        -NoProfile -NonInteractive -File "$(cygpath -m "$ROOT/tools/benchmark/run_windows_benchmark.ps1")"
        -Executable "$(cygpath -m "$executable")"
        -Output "$(cygpath -m "$output")"
        -Runs "$RUNS"
        -RuntimeDirectory "$(cygpath -m "$(dirname "$(command -v g++)")")"
      )
      [[ -n "$FILTER" ]] && powershell_args+=(-Filter "$FILTER")
      local windows_root="$(cygpath -u "${SYSTEMROOT:-C:\\Windows}")"
      local windows_drive="${windows_root:1:1}"
      local wsl_powershell="/mnt/${windows_drive,,}${windows_root:2}/System32/WindowsPowerShell/v1.0/powershell.exe"
      if command -v wsl.exe >/dev/null &&
         MSYS_NO_PATHCONV=1 wsl.exe test -x "$wsl_powershell" >/dev/null 2>&1; then
        MSYS_NO_PATHCONV=1 wsl.exe "$wsl_powershell" "${powershell_args[@]}"
      else
        powershell.exe "${powershell_args[@]}"
      fi
      ;;
    *) "$executable" "${args[@]}" ;;
  esac
}

capture "$BASELINE_SOURCE" "$BASELINE_BUILD" "$OUT/baseline-kernels.json"
capture "$ROOT" "$CURRENT_BUILD" "$OUT/candidate-kernels.json"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || command -v py || true)}"
if [[ -n "$PYTHON" ]] && "$PYTHON" -c 'import sys' >/dev/null 2>&1; then
  "$PYTHON" "$ROOT/tools/benchmark/benchmark_regression.py" \
    --baseline "$OUT/baseline-kernels.json" \
    --candidate "$OUT/candidate-kernels.json" \
    --threshold "$THRESHOLD" \
    "${RENAMES[@]}" \
    "${ALLOWED_REMOVED[@]}" \
    --report "$OUT/report.md"
elif [[ "$(uname -s)" =~ ^(MINGW|MSYS|CYGWIN) && "$ROOT" =~ ^/[A-Za-z]/ &&
        -x "$(command -v wsl.exe)" ]]; then
  DRIVE="${ROOT:1:1}"
  WSL_ROOT="/mnt/${DRIVE,,}${ROOT:2}"
  WSL_OUT="/mnt/${DRIVE,,}${OUT:2}"
  MSYS_NO_PATHCONV=1 wsl.exe python3 "$WSL_ROOT/tools/benchmark/benchmark_regression.py" \
    --baseline "$WSL_OUT/baseline-kernels.json" \
    --candidate "$WSL_OUT/candidate-kernels.json" \
    --threshold "$THRESHOLD" \
    "${RENAMES[@]}" \
    "${ALLOWED_REMOVED[@]}" \
    --report "$WSL_OUT/report.md"
else
  echo "python3, python, or py is required to compare benchmark JSON" >&2
  exit 1
fi

echo "Artifacts: $OUT"
