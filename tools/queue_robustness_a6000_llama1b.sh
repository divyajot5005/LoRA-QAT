#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/LoRA-QAT}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda/envs/ndna/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/work/hf_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/work/loraqat_outputs/robustness_llama1b_code}"
LOG_ROOT="${LOG_ROOT:-/mnt/work/loraqat_logs}"

mkdir -p "$CACHE_ROOT" "$LOG_ROOT" "$OUTPUT_ROOT"
export HF_HOME="$CACHE_ROOT"
export HF_HUB_CACHE="$CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"

wait_for_current_chain() {
  while pgrep -af "queue_a6000_1b_fullft.sh" >/dev/null; do
    echo "WAIT_A6000_CHAIN $(date -Is)"
    sleep 60
  done
}

cd "$REPO_DIR"
wait_for_current_chain
"$PYTHON_BIN" run_h100_quant_lora_suite.py \
  --output-root "$OUTPUT_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --models llama32_1b \
  --tasks domain_adaptation_code \
  --phases before_ft no_quant_ft quant_int4_ft quant_fp8_ft quant_int4_act_ft quant_fp8_act_ft \
  | tee "$LOG_ROOT/robustness_llama1b_code.log"
