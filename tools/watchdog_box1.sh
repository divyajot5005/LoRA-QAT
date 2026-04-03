#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/LoRA-QAT}"
LOG_ROOT="${LOG_ROOT:-/opt/dlami/nvme/loraqat_logs}"
SLEEP_SECONDS="${SLEEP_SECONDS:-1800}"
MAIN_LOG="$LOG_ROOT/watchdog_box1.log"

mkdir -p "$LOG_ROOT"

is_running() {
  pgrep -af "$1" >/dev/null
}

start_if_missing() {
  local pattern="$1"
  local cmd="$2"
  local tag="$3"
  if ! is_running "$pattern"; then
    echo "$(date -Is) RESTART $tag" >> "$MAIN_LOG"
    bash -lc "$cmd" >> "$MAIN_LOG" 2>&1 || true
  fi
}

while true; do
  start_if_missing "queue_l40_lora_box1.sh" \
    "nohup bash $REPO_DIR/queue_l40_lora_box1.sh > $LOG_ROOT/launcher_box1.log 2>&1 < /dev/null &" \
    "queue_l40_lora_box1"
  sleep "$SLEEP_SECONDS"
done
