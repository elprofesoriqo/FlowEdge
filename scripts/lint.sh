#!/usr/bin/env bash
# Usage: ./scripts/lint.sh [--fix]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${FLOWEDGE_BUILD_DIR:-$ROOT/build}"
FIX="${1:-}"
FORMAT_FLAG=$([[ "$FIX" == "--fix" ]] && echo "-i" || echo "--dry-run")

find "$ROOT/src" "$ROOT/bench" "$ROOT/test" "$ROOT/examples" "$ROOT/convert" \
  -type f \( -name "*.h" -o -name "*.cc" -o -name "*.cpp" \) -print0 \
  | xargs -0 clang-format --style=file --Werror "$FORMAT_FLAG"

PY="$(command -v py || command -v python3 || command -v python)"
if [ -f "$BUILD_DIR/compile_commands.json" ]; then
  if grep -qi microsoft /proc/version 2>/dev/null &&
     grep -q -- '--target=x86_64-w64-mingw32' "$BUILD_DIR/compile_commands.json"; then
    echo "note: skipping clang-tidy because WSL clang-tidy cannot consume the Windows/MinGW compile database"
    exit 0
  fi
  if [[ "$(uname -s)" == "Linux" ]]; then
    PROBE="$(mktemp --suffix=.cc)"
    trap 'rm -f "$PROBE"' EXIT
    printf '#include <expected>\nstd::expected<int, int> f();\n' > "$PROBE"
    if ! clang-tidy "$PROBE" -- -std=c++23 >/dev/null 2>&1; then
      echo "note: skipping clang-tidy because this Linux clang-tidy cannot parse C++23 std::expected"
      exit 0
    fi
  fi
  "$PY" -c "import json, sys
print('\0'.join(e['file'] for e in json.load(sys.stdin)
                if not any(x in e['file'].replace(chr(92), '/') for x in ['/bench/', '/test/', '/python/', '/_deps/'])), end='')" \
    < "$BUILD_DIR/compile_commands.json" \
    | xargs -0 clang-tidy -p "$BUILD_DIR" ${FIX:+--fix}
else
  echo "note: skipping clang-tidy because $BUILD_DIR/compile_commands.json is missing"
fi
