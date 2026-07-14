#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-}"
FOLLOW=0
INTERVAL=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --follow)
      FOLLOW=1
      shift
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${RESULT_ROOT}" ]]; then
  POINTER="${CICPR2_LATEST_POINTER:-${REPO_ROOT}/cicpr2_six_embedding_generation_latest_result_root.txt}"
  [[ -f "${POINTER}" ]] || { echo "CICP-R2 result pointer not found: ${POINTER}" >&2; exit 1; }
  RESULT_ROOT="$(sed -n '1p' "${POINTER}")"
fi
[[ -d "${RESULT_ROOT}" ]] || { echo "CICP-R2 result root not found: ${RESULT_ROOT}" >&2; exit 1; }

render_once() {
  echo "RESULT_ROOT=${RESULT_ROOT}"
  printf '\n== CICP-R2 latest branch states ==\n'
  if [[ -f "${RESULT_ROOT}/status.tsv" ]]; then
    awk -F '\t' 'NR==1 {next} {line[$1]=$0} END {for (i=1; i<=6; i++) if (i in line) print line[i]}' \
      "${RESULT_ROOT}/status.tsv" | column -t -s $'\t' 2>/dev/null \
      || awk -F '\t' 'NR==1 {next} {line[$1]=$0} END {for (i=1; i<=6; i++) if (i in line) print line[i]}' \
        "${RESULT_ROOT}/status.tsv"
  else
    echo "status.tsv not created yet"
  fi

  local completed_count
  completed_count="$(find "${RESULT_ROOT}/status" -maxdepth 1 -name '*.done' -type f 2>/dev/null | wc -l | tr -d ' ')"
  printf '\ncompleted=%s/6\n' "${completed_count}"

  if [[ -f "${RESULT_ROOT}/logs/master.log" ]]; then
    printf '\n== master tail ==\n'
    tail -n 12 "${RESULT_ROOT}/logs/master.log"
  fi

  printf '\n== CICP-R2 branch progress ==\n'
  local log_file progress_line
  for log_file in "${RESULT_ROOT}"/logs/[1-6]_CICP-R2-*.log; do
    [[ -e "${log_file}" ]] || continue
    progress_line="$(grep -E '\[epoch [0-9]+/100\]\[batch [0-9]+/[0-9]+\]' "${log_file}" | tail -n 1 || true)"
    echo "-- $(basename "${log_file}")"
    if [[ -n "${progress_line}" ]]; then
      echo "${progress_line}"
    else
      tail -n 3 "${log_file}"
    fi
  done

  [[ "${completed_count}" == "6" ]]
}

while true; do
  if [[ "${FOLLOW}" == "1" ]] && [[ -t 1 ]]; then
    clear
  fi
  if render_once; then
    exit 0
  fi
  [[ "${FOLLOW}" == "1" ]] || exit 0
  sleep "${INTERVAL}"
done
