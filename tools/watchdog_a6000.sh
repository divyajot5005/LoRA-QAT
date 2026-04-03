#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/LoRA-QAT}"
LOG_ROOT="${LOG_ROOT:-/mnt/work/loraqat_logs}"
SLEEP_SECONDS="${SLEEP_SECONDS:-1800}"
MAIN_LOG="$LOG_ROOT/watchdog_a6000.log"

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
  start_if_missing "queue_a6000_1b_fullft.sh" \
    "nohup bash $REPO_DIR/queue_a6000_1b_fullft.sh > $LOG_ROOT/launcher_a6000_1b.log 2>&1 < /dev/null &" \
    "queue_a6000_1b_fullft"
  start_if_missing "queue_robustness_a6000_llama1b.sh" \
    "nohup bash $REPO_DIR/queue_robustness_a6000_llama1b.sh > $LOG_ROOT/launcher_robustness_a6000.log 2>&1 < /dev/null &" \
    "queue_robustness_a6000_llama1b"
  sleep "$SLEEP_SECONDS"
done
