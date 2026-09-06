#!/bin/bash
#
# RUN:
#   chmod +x training_monitor.sh
#   ./training_monitor.sh
#
#   Optional flags (all have sane defaults):
#     ./training_monitor.sh --interval 5 --window 6 --log /tmp/my_run.csv
#
#   Run it in a separate terminal/tmux pane alongside your training job.
#   Press Ctrl+C to stop and print a session summary (peak VRAM, peak GPU util, etc.)
#
# training_monitor.sh -- Live GPU/CPU/RAM monitor with batch/worker tuning advice.
# Improvements over the original:
#   - Persistent full-history CSV log (nothing is overwritten/discarded)
#   - Correct rolling average (no more re-reading a truncated file each loop)
#   - No dependency on `bc` (pure awk/bash arithmetic)
#   - Peak VRAM / peak GPU util tracked across the whole session
#   - Multi-GPU aware (averages across all GPUs, shows per-GPU breakdown)
#   - Graceful Ctrl+C exit with a summary instead of an abrupt kill
#   - Config via env vars or flags, no hardcoded magic numbers buried in logic
#   - Clear thresholds you can tune without touching the recommendation logic

set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override via environment or flags: --interval N --window N --log PATH)
# ---------------------------------------------------------------------------
INTERVAL="${INTERVAL:-5}"          # seconds between samples
WINDOW="${WINDOW:-6}"              # samples used for the rolling average
LOG_FILE="${LOG_FILE:-/tmp/training_monitor_$(date +%Y%m%d_%H%M%S).csv}"
VRAM_CRITICAL_PCT="${VRAM_CRITICAL_PCT:-95}"
VRAM_WARN_PCT="${VRAM_WARN_PCT:-88}"
GPU_LOW_UTIL="${GPU_LOW_UTIL:-60}"
GPU_HIGH_UTIL="${GPU_HIGH_UTIL:-90}"
CPU_HIGH_PCT="${CPU_HIGH_PCT:-80}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval) INTERVAL="$2"; shift 2 ;;
        --window) WINDOW="$2"; shift 2 ;;
        --log) LOG_FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found -- no GPU visible."; exit 1; }

echo "timestamp,gpu_util_avg,gpu_mem_used_mb,gpu_mem_total_mb,cpu_pct,ram_pct" > "$LOG_FILE"

# Rolling window kept in memory (bash arrays), not by re-parsing the file.
declare -a hist_gpu_util hist_gpu_mem hist_cpu hist_ram
peak_gpu_mem=0
peak_gpu_util=0
gpu_mem_total=0
sample_count=0
start_time=$(date +%s)

cleanup() {
    echo
    echo "======================================================"
    echo " MONITOR STOPPED -- SESSION SUMMARY"
    echo "======================================================"
    local elapsed=$(( $(date +%s) - start_time ))
    printf "Duration        : %02d:%02d:%02d\n" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
    printf "Samples taken   : %d\n" "$sample_count"
    printf "Peak GPU VRAM   : %d / %d MB (%.1f%%)\n" "$peak_gpu_mem" "$gpu_mem_total" \
        "$(awk -v m="$peak_gpu_mem" -v t="$gpu_mem_total" 'BEGIN{print (t>0)? m/t*100 : 0}')"
    printf "Peak GPU Util   : %d%%\n" "$peak_gpu_util"
    echo "Full log saved  : $LOG_FILE"
    echo "======================================================"
    exit 0
}
trap cleanup INT TERM

avg() {
    # avg "${arr[@]}" -> prints mean, or 0 if empty
    local sum=0 n=0
    for v in "$@"; do sum=$(awk -v s="$sum" -v x="$v" 'BEGIN{print s+x}'); n=$((n+1)); done
    if [ "$n" -eq 0 ]; then echo 0; else awk -v s="$sum" -v n="$n" 'BEGIN{printf "%.1f", s/n}'; fi
}

while true; do
    # --- GPU (averaged across all GPUs if more than one) ---
    mapfile -t gpu_lines < <(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
    gpu_util_sum=0; gpu_mem_sum=0; gpu_total_sum=0; gpu_count=${#gpu_lines[@]}
    per_gpu_report=""
    idx=0
    for line in "${gpu_lines[@]}"; do
        u=$(echo "$line" | awk -F', ' '{print $1}')
        m=$(echo "$line" | awk -F', ' '{print $2}')
        t=$(echo "$line" | awk -F', ' '{print $3}')
        gpu_util_sum=$((gpu_util_sum + u))
        gpu_mem_sum=$((gpu_mem_sum + m))
        gpu_total_sum=$((gpu_total_sum + t))
        per_gpu_report+="  GPU${idx}: ${u}% util, ${m}/${t} MB\n"
        idx=$((idx+1))
    done
    gpu_util=$((gpu_util_sum / gpu_count))
    gpu_mem=$gpu_mem_sum
    gpu_mem_total=$gpu_total_sum

    # --- CPU ---
    cpu_line=$(top -bn1 | grep "Cpu(s)")
    cpu_idle=$(echo "$cpu_line" | grep -oP '[\d.]+(?=\s*id)' || echo "$cpu_line" | awk -F',' '{for(i=1;i<=NF;i++) if($i ~ /id/) print $i}' | grep -oP '[\d.]+')
    cpu_used=$(awk -v idle="${cpu_idle:-0}" 'BEGIN{printf "%.1f", 100-idle}')

    # --- RAM ---
    ram_used=$(free | awk '/Mem:/ {printf "%.1f", $3/$2*100}')

    # --- track peaks ---
    [ "$gpu_mem" -gt "$peak_gpu_mem" ] && peak_gpu_mem=$gpu_mem
    [ "$gpu_util" -gt "$peak_gpu_util" ] && peak_gpu_util=$gpu_util

    # --- append to persistent log (never truncated) ---
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$ts,$gpu_util,$gpu_mem,$gpu_mem_total,$cpu_used,$ram_used" >> "$LOG_FILE"
    sample_count=$((sample_count + 1))

    # --- maintain in-memory rolling window ---
    hist_gpu_util+=("$gpu_util"); hist_gpu_mem+=("$gpu_mem"); hist_cpu+=("$cpu_used"); hist_ram+=("$ram_used")
    while [ "${#hist_gpu_util[@]}" -gt "$WINDOW" ]; do
        hist_gpu_util=("${hist_gpu_util[@]:1}")
        hist_gpu_mem=("${hist_gpu_mem[@]:1}")
        hist_cpu=("${hist_cpu[@]:1}")
        hist_ram=("${hist_ram[@]:1}")
    done

    avg_gpu_util=$(avg "${hist_gpu_util[@]}")
    avg_gpu_mem=$(avg "${hist_gpu_mem[@]}")
    avg_cpu=$(avg "${hist_cpu[@]}")
    avg_ram=$(avg "${hist_ram[@]}")
    mem_pct=$(awk -v m="$avg_gpu_mem" -v t="$gpu_mem_total" 'BEGIN{printf "%.1f", (t>0)? m/t*100 : 0}')

    clear
    echo "======================================================"
    echo "             TRAINING RESOURCE MONITOR"
    echo "======================================================"
    echo
    printf "Rolling avg over last %d samples (%ds interval)\n" "${#hist_gpu_util[@]}" "$INTERVAL"
    echo
    printf "GPU UTILIZATION : %s%%   (peak: %d%%)\n" "$avg_gpu_util" "$peak_gpu_util"
    printf "GPU VRAM        : %s / %s MB (%s%%)   (peak: %d MB)\n" "$avg_gpu_mem" "$gpu_mem_total" "$mem_pct" "$peak_gpu_mem"
    printf "CPU             : %s%%\n" "$avg_cpu"
    printf "SYSTEM RAM      : %s%%\n" "$avg_ram"
    if [ "$gpu_count" -gt 1 ]; then
        echo
        echo "Per-GPU breakdown:"
        echo -e "$per_gpu_report"
    fi
    echo "------------------------------------------------------"

    # --- recommendation logic (order matters: most urgent first) ---
    mem_pct_int=${mem_pct%.*}
    avg_gpu_util_int=${avg_gpu_util%.*}
    avg_cpu_int=${avg_cpu%.*}

    if [ "$mem_pct_int" -ge "$VRAM_CRITICAL_PCT" ]; then
        echo "!!! GPU VRAM CRITICAL (>=${VRAM_CRITICAL_PCT}%)"
        echo "ACTION: REDUCE BATCH SIZE NOW -- OOM risk is high"
    elif [ "$mem_pct_int" -ge "$VRAM_WARN_PCT" ]; then
        echo "WARNING: GPU VRAM HIGH (>=${VRAM_WARN_PCT}%)"
        echo "ACTION: Do not increase batch size further; this is close to your ceiling"
    elif [ "$avg_gpu_util_int" -lt "$GPU_LOW_UTIL" ] && [ "$avg_cpu_int" -ge "$CPU_HIGH_PCT" ]; then
        echo "GPU UNDERFED, CPU SATURATED"
        echo "ACTION: Data loading is the bottleneck -- REDUCE workers won't help;"
        echo "        check augmentation cost / disk I/O, or move preprocessing off-CPU"
    elif [ "$avg_gpu_util_int" -lt "$GPU_LOW_UTIL" ]; then
        echo "GPU UTILIZATION LOW (<${GPU_LOW_UTIL}%)"
        echo "ACTION: CPU has headroom -- CONSIDER INCREASING WORKERS"
    elif [ "$avg_gpu_util_int" -ge "$GPU_HIGH_UTIL" ] && [ "$avg_cpu_int" -lt 50 ]; then
        echo "GPU UTILIZATION HIGH, CPU LIGHT"
        echo "ACTION: GPU IS WELL FED -- KEEP CONFIGURATION"
    elif [ "$avg_gpu_util_int" -ge "$GPU_HIGH_UTIL" ] && [ "$avg_cpu_int" -ge "$CPU_HIGH_PCT" ]; then
        echo "GPU + CPU BOTH HIGH"
        echo "ACTION: SYSTEM IS FULLY UTILIZED -- KEEP CONFIGURATION"
    else
        echo "RESOURCE USAGE: NORMAL"
        echo "ACTION: KEEP CONFIGURATION"
    fi
    echo "------------------------------------------------------"
    echo "Full log (never truncated): $LOG_FILE"
    echo "Press Ctrl+C for a session summary. Updating every ${INTERVAL}s..."
    echo "======================================================"
    sleep "$INTERVAL"
done
