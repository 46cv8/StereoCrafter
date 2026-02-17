#!/usr/bin/env bash
set -u

# Monitor GPU VRAM usage via nvidia-smi and report peak usage on exit.
#
# Usage:
#   ./scripts/monitor_vram_peak.sh [duration_sec] [interval_sec]
#
# Examples:
#   ./scripts/monitor_vram_peak.sh
#   ./scripts/monitor_vram_peak.sh 120
#   ./scripts/monitor_vram_peak.sh 0 0.1
#
# Notes:
# - duration_sec=0 (default) means run until Ctrl+C.
# - interval_sec defaults to 0.1 (100 ms).

DURATION_SEC="${1:-0}"
INTERVAL_SEC="${2:-0.1}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found in PATH." >&2
  exit 1
fi

trim() {
  local v="$1"
  v="${v#"${v%%[![:space:]]*}"}"
  v="${v%"${v##*[![:space:]]}"}"
  printf '%s' "$v"
}

declare -A PEAK_USED_MIB
declare -A TOTAL_MIB
declare -A GPU_NAME

SAMPLE_COUNT=0
START_NS="$(date +%s%N)"
PRINTED=0

print_summary() {
  if [[ "${PRINTED}" -ne 0 ]]; then
    return
  fi
  PRINTED=1

  local end_ns elapsed_sec
  end_ns="$(date +%s%N)"
  elapsed_sec="$(awk -v s="${START_NS}" -v e="${end_ns}" 'BEGIN { printf "%.3f", (e-s)/1e9 }')"

  echo
  echo "=== VRAM Peak Summary ==="
  echo "Samples: ${SAMPLE_COUNT}"
  echo "Elapsed: ${elapsed_sec}s"

  if [[ "${#PEAK_USED_MIB[@]}" -eq 0 ]]; then
    echo "No GPU samples captured."
    return
  fi

  local sum_used=0
  local sum_total=0
  local idx used total pct name

  while IFS= read -r idx; do
    used="${PEAK_USED_MIB[$idx]}"
    total="${TOTAL_MIB[$idx]:-0}"
    name="${GPU_NAME[$idx]:-unknown}"
    pct="$(awk -v u="${used}" -v t="${total}" 'BEGIN { if (t > 0) printf "%.2f", (u/t)*100; else printf "0.00" }')"
    printf "GPU%s (%s): peak %s MiB / %s MiB (%s%%)\n" "${idx}" "${name}" "${used}" "${total}" "${pct}"
    sum_used=$((sum_used + used))
    sum_total=$((sum_total + total))
  done < <(printf '%s\n' "${!PEAK_USED_MIB[@]}" | sort -n)

  local sum_pct
  sum_pct="$(awk -v u="${sum_used}" -v t="${sum_total}" 'BEGIN { if (t > 0) printf "%.2f", (u/t)*100; else printf "0.00" }')"
  printf "All GPUs combined peak: %s MiB / %s MiB (%s%%)\n" "${sum_used}" "${sum_total}" "${sum_pct}"
}

trap 'print_summary; exit 130' INT TERM
trap 'print_summary' EXIT

while true; do
  mapfile -t rows < <(nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)

  if [[ "${#rows[@]}" -eq 0 ]]; then
    echo "WARNING: nvidia-smi returned no data; retrying..." >&2
    sleep "${INTERVAL_SEC}"
    continue
  fi

  for row in "${rows[@]}"; do
    IFS=',' read -r raw_idx raw_name raw_used raw_total <<<"${row}"
    idx="$(trim "${raw_idx}")"
    name="$(trim "${raw_name}")"
    used="$(trim "${raw_used}")"
    total="$(trim "${raw_total}")"

    [[ -z "${idx}" || -z "${used}" || -z "${total}" ]] && continue

    if [[ -z "${PEAK_USED_MIB[$idx]+x}" || "${used}" -gt "${PEAK_USED_MIB[$idx]}" ]]; then
      PEAK_USED_MIB[$idx]="${used}"
      TOTAL_MIB[$idx]="${total}"
      GPU_NAME[$idx]="${name}"
    fi
  done

  SAMPLE_COUNT=$((SAMPLE_COUNT + 1))

  if awk -v d="${DURATION_SEC}" 'BEGIN { exit !(d > 0) }'; then
    now_ns="$(date +%s%N)"
    if awk -v s="${START_NS}" -v n="${now_ns}" -v d="${DURATION_SEC}" 'BEGIN { exit !(((n-s)/1e9) >= d) }'; then
      break
    fi
  fi

  sleep "${INTERVAL_SEC}"
done
