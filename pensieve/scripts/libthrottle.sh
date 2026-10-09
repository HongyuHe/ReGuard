#!/usr/bin/env bash
set -euo pipefail

if [ "${SDR_LIB_THROTTLE_LOADED:-0}" = "1" ]; then
    return 0
fi
SDR_LIB_THROTTLE_LOADED=1

throttle_get_ncpu() {
    if command -v getconf >/dev/null 2>&1; then
        getconf _NPROCESSORS_ONLN
        return
    fi
    if command -v nproc >/dev/null 2>&1; then
        nproc
        return
    fi
    sysctl -n hw.ncpu 2>/dev/null || echo 1
}

throttle_get_load1() {
    if [ -r /proc/loadavg ]; then
        awk '{print $1}' /proc/loadavg
        return
    fi
    sysctl -n vm.loadavg 2>/dev/null | awk '{gsub(/[{}]/,""); print $1}'
}

throttle_get_total_cpu_pct() {
    ps -Ao %cpu= 2>/dev/null | awk '{s+=$1} END{printf "%.2f\n", s+0}'
}

throttle_init() {
    THROTTLE_PIDS=()

    THROTTLE_NCPU="$(throttle_get_ncpu)"
    THROTTLE_MAX_PARALLEL="${THROTTLE_MAX_PARALLEL:-$THROTTLE_NCPU}"
    THROTTLE_MAX_LOAD_1M="${THROTTLE_MAX_LOAD_1M:-$THROTTLE_NCPU}"
    THROTTLE_MAX_CPU_UTIL_PCT="${THROTTLE_MAX_CPU_UTIL_PCT:-95}"
    THROTTLE_SLEEP_S="${THROTTLE_SLEEP_S:-2}"
    THROTTLE_LAUNCH_SLEEP_S="${THROTTLE_LAUNCH_SLEEP_S:-0}"
}

throttle_prune_pids() {
    local alive=()
    local pid
    for pid in "${THROTTLE_PIDS[@]:-}"; do
        if kill -0 "$pid" >/dev/null 2>&1; then
            alive+=("$pid")
        fi
    done
    if [ "${#alive[@]}" -eq 0 ]; then
        THROTTLE_PIDS=()
    else
        THROTTLE_PIDS=("${alive[@]}")
    fi
}

throttle_wait_for_resources() {
    while :; do
        throttle_prune_pids

        local cpu_pct
        cpu_pct="$(throttle_get_total_cpu_pct || echo "")"
        if [ -n "$cpu_pct" ]; then
            local cpu_limit
            cpu_limit="$(awk -v n="$THROTTLE_NCPU" -v pct="$THROTTLE_MAX_CPU_UTIL_PCT" 'BEGIN{printf "%.2f\n", (n*pct)}')"
            if awk -v u="$cpu_pct" -v lim="$cpu_limit" 'BEGIN{exit !(u >= lim)}'; then
                sleep "$THROTTLE_SLEEP_S"
                continue
            fi
        fi

        if [ "${#THROTTLE_PIDS[@]}" -lt "$THROTTLE_MAX_PARALLEL" ]; then
            local load1
            load1="$(throttle_get_load1 || echo "")"
            if [ -z "$load1" ]; then
                break
            fi
            if awk -v l="$load1" -v m="$THROTTLE_MAX_LOAD_1M" 'BEGIN{exit !(l >= m)}'; then
                sleep "$THROTTLE_SLEEP_S"
                continue
            fi
            break
        fi
        sleep "$THROTTLE_SLEEP_S"
    done
}

throttle_run() {
    throttle_wait_for_resources
    if [ "${THROTTLE_LAUNCH_SLEEP_S}" != "0" ]; then
        sleep "$THROTTLE_LAUNCH_SLEEP_S"
    fi
    "$@" &
    THROTTLE_PIDS+=("$!")
}

throttle_wait_all() {
    local pid
    for pid in "${THROTTLE_PIDS[@]:-}"; do
        wait "$pid"
    done
}

export WANDB_API_KEY=249d18df5664b74e2d6e647dc26a0ec062d126c2