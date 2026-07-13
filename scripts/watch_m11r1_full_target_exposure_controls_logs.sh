#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LATEST_FILE="${LATEST_FILE:-${REPO_ROOT}/m11r1_full_target_exposure_controls_latest_result_root.txt}"
TAIL_LINES="${TAIL_LINES:-80}"
FOLLOW="${FOLLOW:-0}"

if [[ ! -f "${LATEST_FILE}" ]]; then
  echo "latest result root file not found: ${LATEST_FILE}" >&2
  exit 1
fi

RESULT_ROOT="$(sed -n '1p' "${LATEST_FILE}")"
if [[ -z "${RESULT_ROOT}" || ! -d "${RESULT_ROOT}" ]]; then
  echo "result root not found: ${RESULT_ROOT}" >&2
  exit 1
fi

echo "RESULT_ROOT=${RESULT_ROOT}"
echo

if [[ -f "${RESULT_ROOT}/launcher_manifest.env" ]]; then
  echo "== launcher_manifest.env =="
  sed -n '1,180p' "${RESULT_ROOT}/launcher_manifest.env"
  echo
fi

if [[ -f "${RESULT_ROOT}/logs/master.log" ]]; then
  echo "== master.log =="
  tail -n "${TAIL_LINES}" "${RESULT_ROOT}/logs/master.log"
  echo
fi

if compgen -G "${RESULT_ROOT}/logs/*.log" >/dev/null; then
  for log_file in "${RESULT_ROOT}"/logs/*.log; do
    [[ "$(basename "${log_file}")" == "master.log" ]] && continue
    echo "== $(basename "${log_file}") =="
    tail -n "${TAIL_LINES}" "${log_file}"
    echo
  done
fi

if [[ "${FOLLOW}" == "1" ]]; then
  echo "== follow logs =="
  tail -n "${TAIL_LINES}" -f "${RESULT_ROOT}"/logs/*.log
fi
