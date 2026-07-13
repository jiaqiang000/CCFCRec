#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EPOCH="${EPOCH:-1}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-m11_target_competitor_pair_seed43_smoke}"
exec bash "${SCRIPT_DIR}/run_m11_target_competitor_pair_seed43_fast_uniform_mps_100epoch.sh" "$@"
