#!/usr/bin/env bash
# Jetson / ARM helper for the Diffusion Policy simulator JSON.
set -euo pipefail

checkpoint=${1:?usage: $0 <converted.safetensors> [extra args]}
shift || true
python -m flowedge_dev pipeline rollout "$checkpoint" \
  --steps 20 --threads 4 --period-ms 10 --on-miss hold "$@"
