#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_task4_acat_v3_weight_controls_m1_m2_m6_m3_seed43_fast_uniform_mps_100epoch.sh" "$@"
