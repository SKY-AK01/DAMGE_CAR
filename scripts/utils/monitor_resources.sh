#!/bin/bash
# ==============================================================================
# monitor_resources.sh — Live GPU/CPU/RAM utilization logger
#
# Run this in a SEPARATE terminal/tmux pane while training is happening in
# another one. It samples GPU util/VRAM (via nvidia-smi), overall CPU %
# (via /proc/stat, no extra packages needed), and RAM usage (via free) every
# N seconds and appends each row to a CSV log file — so you have a full
# timeline afterward instead of just whatever you happened to glance at.
#
# Usage:
#   ./scripts/utils/monitor_resources.sh                # default: 2s interval
#   ./scripts/utils/monitor_resources.sh 5               # 5s interval
#   ./scripts/utils/monitor_resources.sh 2 my_run.csv    # custom log filename
#
# Stop with Ctrl+C — it exits cleanly and tells you where the log is.
# ==============================================================================
set -u

INTERVAL="${1:-2}"
LOG_DIR="logs/resource_monitor"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${2:-${LOG_DIR}/resources_${TIMESTAMP}.csv}"

if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found. Are you on the GPU server?"
    exit 1
fi

# CSV header
echo "timestamp,gpu_util_pct,gpu_mem_util_pct,gpu_mem_used_mib,gpu_mem_total_mib,cpu_util_pct,ram_used_gb,ram_total_gb,ram_used_pct" > "$LOG_FILE"

echo "=============================================="
echo " Resource Monitor"
echo " Interval : ${INTERVAL}s"
echo " Log file : ${LOG_FILE}"
echo " Stop with Ctrl+C"
echo "=============================================="

# Reads two /proc/stat snapshots INTERVAL seconds apart to compute a real
# overall CPU utilization percentage (top -bn1's first sample is often
# inaccurate/misleading, this way is more reliable).
read_cpu_snapshot() {
    read -r _ user nice system idle iowait irq softirq steal _ < /proc/stat
    echo "$((user+nice+system+idle+iowait+irq+softirq+steal)) $((user+nice+system+irq+softirq+steal))"
}

cleanup() {
    echo ""
    echo "[OK] Stopped. Full log saved at: ${LOG_FILE}"
    echo "     Quick summary: awk -F, 'NR>1{g+=\$2;c+=\$6;n++} END{printf \"avg GPU %.1f%%, avg CPU %.1f%%\\n\", g/n, c/n}' ${LOG_FILE}"
    exit 0
}
trap cleanup INT TERM

prev_total=0
prev_active=0
read prev_total prev_active < <(read_cpu_snapshot)

while true; do
    sleep "$INTERVAL"

    # --- GPU ---
    gpu_line=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv,noheader,nounits | head -n1)
    gpu_util=$(echo "$gpu_line" | awk -F',' '{gsub(/ /,"",$1); print $1}')
    gpu_mem_util=$(echo "$gpu_line" | awk -F',' '{gsub(/ /,"",$2); print $2}')
    gpu_mem_used=$(echo "$gpu_line" | awk -F',' '{gsub(/ /,"",$3); print $3}')
    gpu_mem_total=$(echo "$gpu_line" | awk -F',' '{gsub(/ /,"",$4); print $4}')

    # --- CPU (overall %, across all cores, since last sample) ---
    read curr_total curr_active < <(read_cpu_snapshot)
    total_delta=$((curr_total - prev_total))
    active_delta=$((curr_active - prev_active))
    if [ "$total_delta" -gt 0 ]; then
        cpu_util=$(awk -v a="$active_delta" -v t="$total_delta" 'BEGIN{printf "%.1f", (a/t)*100}')
    else
        cpu_util="0.0"
    fi
    prev_total=$curr_total
    prev_active=$curr_active

    # --- RAM ---
    ram_line=$(free -g | awk '/^Mem:/{print $3","$2}')
    ram_used=$(echo "$ram_line" | cut -d',' -f1)
    ram_total=$(echo "$ram_line" | cut -d',' -f2)
    ram_pct=$(awk -v u="$ram_used" -v t="$ram_total" 'BEGIN{ if (t>0) printf "%.1f", (u/t)*100; else print "0.0" }')

    now=$(date +"%Y-%m-%d %H:%M:%S")
    row="${now},${gpu_util},${gpu_mem_util},${gpu_mem_used},${gpu_mem_total},${cpu_util},${ram_used},${ram_total},${ram_pct}"
    echo "$row" >> "$LOG_FILE"

    # Live view in the terminal too, not just the file
    printf "\r[%s] GPU: %3s%% (mem %3s%%, %s/%sMiB)  CPU: %5s%%  RAM: %sG/%sG (%s%%)   " \
        "$now" "$gpu_util" "$gpu_mem_util" "$gpu_mem_used" "$gpu_mem_total" "$cpu_util" "$ram_used" "$ram_total" "$ram_pct"
done
