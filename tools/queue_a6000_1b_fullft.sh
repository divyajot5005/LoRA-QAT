#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/LoRA-QAT}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda/envs/ndna/bin/python}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-/root/outputs}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/work/hf_cache}"
LOG_ROOT="${LOG_ROOT:-/mnt/work/loraqat_logs}"

mkdir -p "$CACHE_ROOT" "$LOG_ROOT"
export HF_HOME="$CACHE_ROOT"
export HF_HUB_CACHE="$CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"

wait_for_gemma() {
  while pgrep -af "run_full_ft_quant_sweep.py.*fullft_gemma1b" >/dev/null; do
    echo "WAIT_GEMMA $(date -Is)"
    sleep 60
  done
}

run_fullft_model() {
  local family="$1"
  local out_root="$OUTPUT_ROOT_BASE/fullft_${family}"
  local log_path="$LOG_ROOT/fullft_${family}.log"

  echo "START ${family} $(date -Is)"
  cd "$REPO_DIR"
  "$PYTHON_BIN" run_full_ft_quant_sweep.py \
    --output-root "$out_root" \
    --cache-root "$CACHE_ROOT" \
    --models "$family" \
    --device cuda \
    --proxy-device cuda \
    --skip-proxy-quant-stats \
    | tee "$log_path"
  echo "DONE ${family} $(date -Is)"
}

wait_for_gemma
run_fullft_model "qwen3_0p6b"
