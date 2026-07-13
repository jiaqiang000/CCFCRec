#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper for the post path-audit q-only rescue smoke run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EPOCH="${EPOCH:-1}"
export ALLOW_SMOKE_EPOCH=1
export EXPERIMENT_ID="${EXPERIMENT_ID:-task4_post_path_audit_qonly_rescue_smoke_1epoch}"
exec bash "${SCRIPT_DIR}/run_task4_post_path_audit_qonly_rescue_seed43_fast_uniform_mps_100epoch.sh" "$@"
