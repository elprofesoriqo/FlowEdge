#!/usr/bin/env bash
# Usage: ./scripts/lint.sh [--fix]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIX="${1:-}"
FORMAT_FLAG=$([[ "$FIX" == "--fix" ]] && echo "-i" || echo "--dry-run")

find "$ROOT" \( -name "*.h" -o -name "*.cc" -o -name "*.cpp" \) \
  ! -path "*/build/*" -print0 \
  | xargs -0 clang-format --style=file --Werror "$FORMAT_FLAG"

find "$ROOT" \( -name "*.cc" -o -name "*.cpp" \) \
  ! -path "*/build/*" -print0 \
  | xargs -0 clang-tidy -p "$ROOT/build" ${FIX:+--fix}
