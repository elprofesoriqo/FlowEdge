#!/usr/bin/env bash
# Usage: ./scripts/lint.sh [--fix]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIX="${1:-}"
FORMAT_FLAG=$([[ "$FIX" == "--fix" ]] && echo "-i" || echo "--dry-run")

find "$ROOT" \( -name "*.h" -o -name "*.cc" -o -name "*.cpp" \) \
  ! -path "*/build/*" -print0 \
  | xargs -0 clang-format --style=file --Werror "$FORMAT_FLAG"

if [ -f "$ROOT/build/compile_commands.json" ]; then
  python3 -c "import json, sys; print('\0'.join(e['file'] for e in json.load(sys.stdin) if not any(x in e['file'] for x in ['/bench/', '/test/', '/python/'])), end='')" < "$ROOT/build/compile_commands.json" \
    | xargs -0 clang-tidy -p "$ROOT/build" ${FIX:+--fix}
else
  echo "note: skipping clang-tidy because build/compile_commands.json is missing"
fi
