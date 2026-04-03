#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/LoRA-QAT}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/venvs/loraqat/bin/python}"
NVME_ROOT="${NVME_ROOT:-/opt/dlami/nvme}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$NVME_ROOT/loraqat_outputs/robustness_llama8b_code}"
LOG_ROOT="${LOG_ROOT:-$NVME_ROOT/loraqat_logs}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$NVME_ROOT/hf_cache" "$NVME_ROOT/tmp"
export HF_HOME="$NVME_ROOT/hf_cache"
export HF_HUB_CACHE="$NVME_ROOT/hf_cache/hub"
export HF_DATASETS_CACHE="$NVME_ROOT/hf_cache/datasets"
export TRANSFORMERS_CACHE="$NVME_ROOT/hf_cache/transformers"
export TMPDIR="$NVME_ROOT/tmp"

wait_for_current_chain() {
  while pgrep -af "queue_l40_lora_box2.sh|queue_l40_box2_followup.sh" >/dev/null; do
    echo "WAIT_BOX2_CHAIN $(date -Is)"
    sleep 60
  done
}

cd "$REPO_DIR"
"$PYTHON_BIN" - <<'PY'
from huggingface_hub import get_token
assert get_token(), "Missing HF token for gated Llama-3.1-8B-Instruct"
PY
wait_for_current_chain
"$PYTHON_BIN" run_h100_quant_lora_suite.py \
  --output-root "$OUTPUT_ROOT" \
  --cache-root "$NVME_ROOT/hf_cache" \
  --models llama31_8b \
  --tasks domain_adaptation_code \
  --phases before_ft no_quant_ft quant_int4_ft quant_fp8_ft quant_int4_act_ft quant_fp8_act_ft \
  | tee "$LOG_ROOT/robustness_llama8b_code.log"
