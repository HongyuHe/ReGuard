#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
GRACE_SECONDS=3
DRY_RUN=0

usage() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [--dry-run] [--grace-seconds N]

Kill offline Sage fine-tuning processes:
  - sage_rl/offline/offline_agent.py
  - scripts/finetune_sage_on_adv_traces.py

Options:
  --dry-run          Show matching processes without killing them.
  --grace-seconds N  Wait time between TERM and KILL (default: ${GRACE_SECONDS}).
  -h, --help         Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --grace-seconds)
      GRACE_SECONDS="${2:-}"
      if [[ -z "${GRACE_SECONDS}" || ! "${GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
        echo "invalid --grace-seconds value: ${2:-<missing>}" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v ps >/dev/null 2>&1; then
  echo "ps is required but not found" >&2
  exit 1
fi

MATCHES="$(
  ps -eo pid=,etimes=,pcpu=,pmem=,args= | awk '
    /sage_rl\/offline\/offline_agent\.py/ || /scripts\/finetune_sage_on_adv_traces\.py/ {
      print
    }
  '
)"

if [[ -z "${MATCHES}" ]]; then
  echo "No offline fine-tuning jobs found."
  exit 0
fi

echo "Matching offline fine-tuning jobs:"
printf '%s\n' "${MATCHES}"

PIDS=()
while read -r pid _rest; do
  [[ -z "${pid}" ]] && continue
  [[ "${pid}" =~ ^[0-9]+$ ]] || continue
  PIDS+=("${pid}")
done <<< "${MATCHES}"

if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "No valid PIDs found." >&2
  exit 1
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run only; no processes were killed."
  exit 0
fi

echo
echo "Sending TERM to: ${PIDS[*]}"
kill "${PIDS[@]}" 2>/dev/null || true

if [[ "${GRACE_SECONDS}" -gt 0 ]]; then
  sleep "${GRACE_SECONDS}"
fi

REMAINING=()
for pid in "${PIDS[@]}"; do
  if kill -0 "${pid}" 2>/dev/null; then
    REMAINING+=("${pid}")
  fi
done

if [[ ${#REMAINING[@]} -gt 0 ]]; then
  echo "Sending KILL to stubborn processes: ${REMAINING[*]}"
  kill -9 "${REMAINING[@]}" 2>/dev/null || true
fi

echo
echo "Remaining matches after cleanup:"
POST_MATCHES="$(
  ps -eo pid=,etimes=,pcpu=,pmem=,args= | awk '
    /sage_rl\/offline\/offline_agent\.py/ || /scripts\/finetune_sage_on_adv_traces\.py/ {
      print
    }
  '
)"

if [[ -z "${POST_MATCHES}" ]]; then
  echo "None."
else
  printf '%s\n' "${POST_MATCHES}"
fi
