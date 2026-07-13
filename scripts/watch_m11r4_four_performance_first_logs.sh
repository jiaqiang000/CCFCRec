#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-}"

if [[ -z "${RESULT_ROOT}" ]]; then
  POINTER="${REPO_ROOT}/m11r4_four_performance_first_latest_result_root.txt"
  if [[ ! -f "${POINTER}" ]]; then
    echo "M11-R4 result pointer not found: ${POINTER}" >&2
    exit 1
  fi
  RESULT_ROOT="$(sed -n '1p' "${POINTER}")"
fi

if [[ ! -d "${RESULT_ROOT}" ]]; then
  echo "M11-R4 result root not found: ${RESULT_ROOT}" >&2
  exit 1
fi

echo "RESULT_ROOT=${RESULT_ROOT}"
if [[ -f "${RESULT_ROOT}/status.tsv" ]]; then
  printf '\n== status ==\n'
  column -t -s $'\t' "${RESULT_ROOT}/status.tsv" 2>/dev/null || cat "${RESULT_ROOT}/status.tsv"
fi
if [[ -f "${RESULT_ROOT}/logs/master.log" ]]; then
  printf '\n== master tail ==\n'
  tail -n 20 "${RESULT_ROOT}/logs/master.log"
fi
printf '\n== branch tails ==\n'
for log_file in "${RESULT_ROOT}"/logs/[1-4]_*.log; do
  [[ -e "${log_file}" ]] || continue
  echo "-- $(basename "${log_file}")"
  tail -n 8 "${log_file}"
done
