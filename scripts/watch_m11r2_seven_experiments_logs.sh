#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LATEST_FILE="${LATEST_FILE:-${REPO_ROOT}/m11r2_seven_experiments_latest_result_root.txt}"
TAIL_LINES="${TAIL_LINES:-30}"
FOLLOW="${FOLLOW:-0}"
REFRESH_SECONDS="${REFRESH_SECONDS:-20}"

if [[ ! -f "${LATEST_FILE}" ]]; then
  echo "latest result root file not found: ${LATEST_FILE}" >&2
  exit 1
fi

RESULT_ROOT="$(sed -n '1p' "${LATEST_FILE}")"
if [[ -z "${RESULT_ROOT}" || ! -d "${RESULT_ROOT}" ]]; then
  echo "result root not found: ${RESULT_ROOT}" >&2
  exit 1
fi

show_snapshot() {
  echo "RESULT_ROOT=${RESULT_ROOT}"
  echo
  if [[ -f "${RESULT_ROOT}/status.tsv" ]]; then
    echo "== branch status history =="
    column -t -s $'\t' "${RESULT_ROOT}/status.tsv" 2>/dev/null || cat "${RESULT_ROOT}/status.tsv"
    echo
  fi

  echo "== per-branch latest progress =="
  if compgen -G "${RESULT_ROOT}/logs/*.log" >/dev/null; then
    for log_file in "${RESULT_ROOT}"/logs/*.log; do
      [[ "$(basename "${log_file}")" == "master.log" ]] && continue
      progress_line="$(grep -E '\[epoch [0-9]+/100\]' "${log_file}" | tail -n 1 || true)"
      if [[ -z "${progress_line}" ]]; then
        progress_line="$(tail -n 1 "${log_file}" 2>/dev/null || true)"
      fi
      printf '%-72s %s\n' "$(basename "${log_file}")" "${progress_line:-waiting}"
    done
  else
    echo "no branch logs yet"
  fi
  echo

  if [[ -f "${RESULT_ROOT}/logs/master.log" ]]; then
    echo "== master.log tail =="
    tail -n "${TAIL_LINES}" "${RESULT_ROOT}/logs/master.log"
  fi
}

if [[ "${FOLLOW}" != "1" ]]; then
  show_snapshot
  exit 0
fi

while true; do
  printf '\033[2J\033[H'
  show_snapshot
  if grep -q '^ALL_DONE' "${RESULT_ROOT}/logs/master.log" 2>/dev/null; then
    exit 0
  fi
  sleep "${REFRESH_SECONDS}"
done
