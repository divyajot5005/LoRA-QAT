#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/LoRA-QAT}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/venvs/loraqat/bin/python}"
NVME_ROOT="${NVME_ROOT:-/opt/dlami/nvme}"
LORA_OUTPUT_ROOT="${LORA_OUTPUT_ROOT:-$NVME_ROOT/loraqat_outputs/lora_box2}"
FOLLOWUP_OUTPUT_ROOT="${FOLLOWUP_OUTPUT_ROOT:-$NVME_ROOT/loraqat_outputs/lora_qwen3_0p6b}"
LOG_ROOT="${LOG_ROOT:-$NVME_ROOT/loraqat_logs}"

mkdir -p "$LOG_ROOT" "$NVME_ROOT/hf_cache" "$NVME_ROOT/tmp"
export HF_HOME="$NVME_ROOT/hf_cache"
export HF_HUB_CACHE="$NVME_ROOT/hf_cache/hub"
export HF_DATASETS_CACHE="$NVME_ROOT/hf_cache/datasets"
export TRANSFORMERS_CACHE="$NVME_ROOT/hf_cache/transformers"
export TMPDIR="$NVME_ROOT/tmp"

wait_for_lora_box2() {
  while pgrep -af "run_h100_quant_lora_suite.py --output-root ${LORA_OUTPUT_ROOT}" >/dev/null; do
    echo "WAIT_LORA_BOX2 $(date -Is)"
    sleep 60
  done
}

cd "$REPO_DIR"
wait_for_lora_box2
"$PYTHON_BIN" run_h100_quant_lora_suite.py \
  --output-root "$FOLLOWUP_OUTPUT_ROOT" \
  --cache-root "$NVME_ROOT/hf_cache" \
  --models qwen3_0p6b \
  --tasks instruction_tuning \
  --phases before_ft no_quant_ft quant_int4_ft quant_fp8_ft quant_int4_act_ft quant_fp8_act_ft \
  | tee "$LOG_ROOT/lora_qwen3_0p6b_followup.log"
