#!/usr/bin/env bash
# Build and run the GoogleTest suite via CTest.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) export PATH="/c/Strawberry/c/bin:$PATH" ;; esac

"$ROOT/scripts/build.sh" Release -DFLOWEDGE_TESTS=ON -DFLOWEDGE_BENCH=OFF
ctest --test-dir "$ROOT/build" --output-on-failure
