#!/usr/bin/env bash
# LLVM/clang build. Usage: ./scripts/build.sh [Debug|Release] [extra -D cmake args...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_TYPE="${1:-Release}"

command -v cmake >/dev/null || { echo "cmake not found"; exit 1; }

GEN="Unix Makefiles"
command -v ninja >/dev/null && GEN="Ninja"

ARGS=(
  -S "$ROOT" -B "$ROOT/build" -G "$GEN"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

if command -v clang++ >/dev/null; then
  ARGS+=(-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++)
  case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*)
      ARGS+=(-DCMAKE_C_COMPILER_TARGET=x86_64-w64-mingw32
             -DCMAKE_CXX_COMPILER_TARGET=x86_64-w64-mingw32) ;;
  esac
fi

shift || true # remove build type

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      ARGS+=("-DFLOWEDGE_BACKEND=$2")
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done
cmake --log-level=WARNING "${ARGS[@]}"
cmake --build "$ROOT/build" -j "$(nproc 2>/dev/null || echo 4)"
