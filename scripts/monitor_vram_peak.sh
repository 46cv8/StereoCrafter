#!/usr/bin/env bash
set -u

# Real-time GPU monitor (nvidia-smi) with:
# - current memory/utilization
# - EMA over the last ~5 seconds
# - peak memory/utilization
#
# Usage:
#   ./scripts/monitor_vram_peak.sh [duration_sec] [interval_sec] [print_every_sec]
#
# Examples:
#   ./scripts/monitor_vram_peak.sh            # until Ctrl+C, sample 100 ms, print every 1s
#   ./scripts/monitor_vram_peak.sh 120 0.1 1  # 120s run
#
# Notes:
# - duration_sec=0 means run until Ctrl+C.
# - EMA time constant is fixed at 5 seconds.

DURATION_SEC="${1:-0}"
INTERVAL_SEC="${2:-0.1}"
PRINT_EVERY_SEC="${3:-1.0}"
EMA_TAU_SEC="5.0"

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

is_number() {
  [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

to_sec_from_ns() {
  awk -v ns="$1" 'BEGIN { printf "%.6f", ns/1e9 }'
}

calc_alpha() {
  # alpha = 1 - exp(-dt / tau)
  awk -v dt="$1" -v tau="$2" 'BEGIN {
    if (tau <= 0) { printf "%.10f", 1.0; exit }
    if (dt <= 0) { printf "%.10f", 0.0; exit }
    printf "%.10f", 1.0 - exp(-dt / tau)
  }'
}

ema_update() {
  awk -v prev="$1" -v cur="$2" -v a="$3" 'BEGIN {
    printf "%.6f", prev + a * (cur - prev)
  }'
}

float_gt() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'
}

pct_of() {
  awk -v x="$1" -v y="$2" 'BEGIN { if (y > 0) printf "%.2f", (x / y) * 100.0; else printf "0.00" }'
}

declare -A GPU_NAME
declare -A TOTAL_MIB

declare -A CURR_USED_MIB
declare -A CURR_USED_PCT
declare -A CURR_UTIL_PCT

declare -A EMA_USED_MIB
declare -A EMA_USED_PCT
declare -A EMA_UTIL_PCT

declare -A PEAK_USED_MIB
declare -A PEAK_UTIL_PCT

SAMPLE_COUNT=0
START_NS="$(date +%s%N)"
LAST_SAMPLE_NS="${START_NS}"
LAST_PRINT_NS=0
PRINTED=0

CURR_TOTAL_USED_MIB=0
CURR_TOTAL_MIB=0
CURR_TOTAL_USED_PCT="0.00"
CURR_UTIL_MEAN_PCT="0.00"

EMA_TOTAL_USED_MIB="0.000000"
EMA_TOTAL_USED_PCT="0.000000"
EMA_UTIL_MEAN_PCT="0.000000"

PEAK_TOTAL_USED_MIB=0
PEAK_TOTAL_USED_PCT="0.00"
PEAK_UTIL_MEAN_PCT="0.00"

print_live() {
  local now_ns elapsed_ns elapsed_sec
  now_ns="$(date +%s%N)"
  elapsed_ns=$((now_ns - START_NS))
  elapsed_sec="$(to_sec_from_ns "${elapsed_ns}")"

  echo "[live] t=${elapsed_sec}s samples=${SAMPLE_COUNT} interval=${INTERVAL_SEC}s ema_tau=${EMA_TAU_SEC}s"

  if [[ "${#CURR_USED_MIB[@]}" -eq 0 ]]; then
    echo "  No GPU data yet."
    return
  fi

  local idx
  while IFS= read -r idx; do
    printf "  GPU%s (%s)\n" "${idx}" "${GPU_NAME[$idx]:-unknown}"
    printf "    mem current=%s/%s MiB (%s%%) | ema5s=%s MiB (%s%%) | peak=%s MiB (%s%%)\n" \
      "${CURR_USED_MIB[$idx]:-0}" \
      "${TOTAL_MIB[$idx]:-0}" \
      "${CURR_USED_PCT[$idx]:-0.00}" \
      "$(printf "%.1f" "${EMA_USED_MIB[$idx]:-0}")" \
      "$(printf "%.2f" "${EMA_USED_PCT[$idx]:-0}")" \
      "${PEAK_USED_MIB[$idx]:-0}" \
      "$(pct_of "${PEAK_USED_MIB[$idx]:-0}" "${TOTAL_MIB[$idx]:-0}")"
    printf "    util current=%s%% | ema5s=%s%% | peak=%s%%\n" \
      "${CURR_UTIL_PCT[$idx]:-0.0}" \
      "$(printf "%.2f" "${EMA_UTIL_PCT[$idx]:-0}")" \
      "${PEAK_UTIL_PCT[$idx]:-0.0}"
  done < <(printf '%s\n' "${!CURR_USED_MIB[@]}" | sort -n)

  printf "  TOTAL mem current=%s/%s MiB (%s%%) | ema5s=%s MiB (%s%%) | peak=%s MiB (%s%%)\n" \
    "${CURR_TOTAL_USED_MIB}" \
    "${CURR_TOTAL_MIB}" \
    "${CURR_TOTAL_USED_PCT}" \
    "$(printf "%.1f" "${EMA_TOTAL_USED_MIB}")" \
    "$(printf "%.2f" "${EMA_TOTAL_USED_PCT}")" \
    "${PEAK_TOTAL_USED_MIB}" \
    "${PEAK_TOTAL_USED_PCT}"
  printf "  TOTAL util current=%s%% | ema5s=%s%% | peak=%s%%\n" \
    "${CURR_UTIL_MEAN_PCT}" \
    "$(printf "%.2f" "${EMA_UTIL_MEAN_PCT}")" \
    "${PEAK_UTIL_MEAN_PCT}"
}

print_summary() {
  if [[ "${PRINTED}" -ne 0 ]]; then
    return
  fi
  PRINTED=1

  local end_ns elapsed_ns elapsed_sec
  end_ns="$(date +%s%N)"
  elapsed_ns=$((end_ns - START_NS))
  elapsed_sec="$(to_sec_from_ns "${elapsed_ns}")"

  echo
  echo "=== GPU Monitor Summary ==="
  echo "Samples: ${SAMPLE_COUNT}"
  echo "Elapsed: ${elapsed_sec}s"

  if [[ "${#PEAK_USED_MIB[@]}" -eq 0 ]]; then
    echo "No GPU samples captured."
    return
  fi

  local idx
  while IFS= read -r idx; do
    printf "GPU%s (%s): peak_mem=%s/%s MiB (%s%%), peak_util=%s%%\n" \
      "${idx}" \
      "${GPU_NAME[$idx]:-unknown}" \
      "${PEAK_USED_MIB[$idx]:-0}" \
      "${TOTAL_MIB[$idx]:-0}" \
      "$(pct_of "${PEAK_USED_MIB[$idx]:-0}" "${TOTAL_MIB[$idx]:-0}")" \
      "${PEAK_UTIL_PCT[$idx]:-0.0}"
  done < <(printf '%s\n' "${!PEAK_USED_MIB[@]}" | sort -n)

  printf "All GPUs combined: peak_mem=%s/%s MiB (%s%%), peak_util_mean=%s%%\n" \
    "${PEAK_TOTAL_USED_MIB}" "${CURR_TOTAL_MIB}" "${PEAK_TOTAL_USED_PCT}" "${PEAK_UTIL_MEAN_PCT}"
}

trap 'print_summary; exit 130' INT TERM
trap 'print_summary' EXIT

echo "Starting GPU monitor: duration=${DURATION_SEC}s interval=${INTERVAL_SEC}s print_every=${PRINT_EVERY_SEC}s ema_tau=${EMA_TAU_SEC}s"

duration_reached() {
  if ! awk -v d="${DURATION_SEC}" 'BEGIN { exit !(d > 0) }'; then
    return 1
  fi
  local now_ns elapsed_sec
  now_ns="$(date +%s%N)"
  elapsed_sec="$(awk -v s="${START_NS}" -v n="${now_ns}" 'BEGIN { printf "%.6f", (n-s)/1e9 }')"
  if awk -v e="${elapsed_sec}" -v d="${DURATION_SEC}" 'BEGIN { exit !(e >= d) }'; then
    return 0
  fi
  return 1
}

while true; do
  local_now_ns="$(date +%s%N)"
  dt_sec="$(awk -v prev="${LAST_SAMPLE_NS}" -v now="${local_now_ns}" 'BEGIN { printf "%.6f", (now-prev)/1e9 }')"
  if [[ "${SAMPLE_COUNT}" -eq 0 ]]; then
    dt_sec="${INTERVAL_SEC}"
  fi
  alpha="$(calc_alpha "${dt_sec}" "${EMA_TAU_SEC}")"
  LAST_SAMPLE_NS="${local_now_ns}"

  mapfile -t rows < <(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  if [[ "${#rows[@]}" -eq 0 ]]; then
    echo "WARNING: nvidia-smi returned no data; retrying..." >&2
    if duration_reached; then
      break
    fi
    sleep "${INTERVAL_SEC}"
    continue
  fi

  CURR_TOTAL_USED_MIB=0
  CURR_TOTAL_MIB=0
  util_sum="0.0"
  gpu_count=0

  for row in "${rows[@]}"; do
    IFS=',' read -r raw_idx raw_name raw_used raw_total raw_util <<<"${row}"
    idx="$(trim "${raw_idx}")"
    name="$(trim "${raw_name}")"
    used="$(trim "${raw_used}")"
    total="$(trim "${raw_total}")"
    util="$(trim "${raw_util}")"

    is_number "${used}" || continue
    is_number "${total}" || continue
    if ! is_number "${util}"; then
      util="0"
    fi

    GPU_NAME["${idx}"]="${name}"
    TOTAL_MIB["${idx}"]="${total}"

    CURR_USED_MIB["${idx}"]="${used}"
    CURR_USED_PCT["${idx}"]="$(pct_of "${used}" "${total}")"
    CURR_UTIL_PCT["${idx}"]="$(awk -v u="${util}" 'BEGIN { printf "%.1f", u }')"

    if [[ -z "${PEAK_USED_MIB[$idx]+x}" || "${used}" -gt "${PEAK_USED_MIB[$idx]}" ]]; then
      PEAK_USED_MIB["${idx}"]="${used}"
    fi
    if [[ -z "${PEAK_UTIL_PCT[$idx]+x}" ]]; then
      PEAK_UTIL_PCT["${idx}"]="$(awk -v u="${util}" 'BEGIN { printf "%.1f", u }')"
    else
      if float_gt "${util}" "${PEAK_UTIL_PCT[$idx]}"; then
        PEAK_UTIL_PCT["${idx}"]="$(awk -v u="${util}" 'BEGIN { printf "%.1f", u }')"
      fi
    fi

    if [[ -z "${EMA_USED_MIB[$idx]+x}" ]]; then
      EMA_USED_MIB["${idx}"]="$(awk -v x="${used}" 'BEGIN { printf "%.6f", x }')"
      EMA_USED_PCT["${idx}"]="$(pct_of "${used}" "${total}")"
      EMA_UTIL_PCT["${idx}"]="$(awk -v u="${util}" 'BEGIN { printf "%.6f", u }')"
    else
      EMA_USED_MIB["${idx}"]="$(ema_update "${EMA_USED_MIB[$idx]}" "${used}" "${alpha}")"
      current_pct="$(pct_of "${used}" "${total}")"
      EMA_USED_PCT["${idx}"]="$(ema_update "${EMA_USED_PCT[$idx]}" "${current_pct}" "${alpha}")"
      EMA_UTIL_PCT["${idx}"]="$(ema_update "${EMA_UTIL_PCT[$idx]}" "${util}" "${alpha}")"
    fi

    CURR_TOTAL_USED_MIB=$((CURR_TOTAL_USED_MIB + used))
    CURR_TOTAL_MIB=$((CURR_TOTAL_MIB + total))
    util_sum="$(awk -v a="${util_sum}" -v b="${util}" 'BEGIN { printf "%.6f", a + b }')"
    gpu_count=$((gpu_count + 1))
  done

  CURR_TOTAL_USED_PCT="$(pct_of "${CURR_TOTAL_USED_MIB}" "${CURR_TOTAL_MIB}")"
  if [[ "${gpu_count}" -gt 0 ]]; then
    CURR_UTIL_MEAN_PCT="$(awk -v s="${util_sum}" -v n="${gpu_count}" 'BEGIN { printf "%.2f", s / n }')"
  else
    CURR_UTIL_MEAN_PCT="0.00"
  fi

  if [[ "${CURR_TOTAL_USED_MIB}" -gt "${PEAK_TOTAL_USED_MIB}" ]]; then
    PEAK_TOTAL_USED_MIB="${CURR_TOTAL_USED_MIB}"
    PEAK_TOTAL_USED_PCT="${CURR_TOTAL_USED_PCT}"
  fi
  if float_gt "${CURR_UTIL_MEAN_PCT}" "${PEAK_UTIL_MEAN_PCT}"; then
    PEAK_UTIL_MEAN_PCT="${CURR_UTIL_MEAN_PCT}"
  fi

  if [[ "${SAMPLE_COUNT}" -eq 0 ]]; then
    EMA_TOTAL_USED_MIB="$(awk -v x="${CURR_TOTAL_USED_MIB}" 'BEGIN { printf "%.6f", x }')"
    EMA_TOTAL_USED_PCT="$(awk -v x="${CURR_TOTAL_USED_PCT}" 'BEGIN { printf "%.6f", x }')"
    EMA_UTIL_MEAN_PCT="$(awk -v x="${CURR_UTIL_MEAN_PCT}" 'BEGIN { printf "%.6f", x }')"
  else
    EMA_TOTAL_USED_MIB="$(ema_update "${EMA_TOTAL_USED_MIB}" "${CURR_TOTAL_USED_MIB}" "${alpha}")"
    EMA_TOTAL_USED_PCT="$(ema_update "${EMA_TOTAL_USED_PCT}" "${CURR_TOTAL_USED_PCT}" "${alpha}")"
    EMA_UTIL_MEAN_PCT="$(ema_update "${EMA_UTIL_MEAN_PCT}" "${CURR_UTIL_MEAN_PCT}" "${alpha}")"
  fi

  SAMPLE_COUNT=$((SAMPLE_COUNT + 1))

  if [[ "${LAST_PRINT_NS}" -eq 0 ]]; then
    print_live
    LAST_PRINT_NS="${local_now_ns}"
  else
    should_print="$(awk -v last="${LAST_PRINT_NS}" -v now="${local_now_ns}" -v pe="${PRINT_EVERY_SEC}" 'BEGIN { if (((now-last)/1e9) >= pe) print 1; else print 0 }')"
    if [[ "${should_print}" -eq 1 ]]; then
      print_live
      LAST_PRINT_NS="${local_now_ns}"
    fi
  fi

  if duration_reached; then
    break
  fi

  sleep "${INTERVAL_SEC}"
done
