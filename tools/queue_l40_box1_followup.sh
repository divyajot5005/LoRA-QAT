#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/LoRA-QAT}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/venvs/loraqat/bin/python}"
NVME_ROOT="${NVME_ROOT:-/opt/dlami/nvme}"
LORA_OUTPUT_ROOT="${LORA_OUTPUT_ROOT:-$NVME_ROOT/loraqat_outputs/lora_box1}"
FULLFT_OUTPUT_ROOT="${FULLFT_OUTPUT_ROOT:-$NVME_ROOT/loraqat_outputs/fullft_llama_1b}"
LOG_ROOT="${LOG_ROOT:-$NVME_ROOT/loraqat_logs}"

mkdir -p "$LOG_ROOT" "$NVME_ROOT/hf_cache" "$NVME_ROOT/tmp"
export HF_HOME="$NVME_ROOT/hf_cache"
export HF_HUB_CACHE="$NVME_ROOT/hf_cache/hub"
export HF_DATASETS_CACHE="$NVME_ROOT/hf_cache/datasets"
export TRANSFORMERS_CACHE="$NVME_ROOT/hf_cache/transformers"
export TMPDIR="$NVME_ROOT/tmp"

wait_for_lora_box1() {
  while pgrep -af "run_h100_quant_lora_suite.py --output-root ${LORA_OUTPUT_ROOT}" >/dev/null; do
    echo "WAIT_LORA_BOX1 $(date -Is)"
    sleep 60
  done
}

cd "$REPO_DIR"
wait_for_lora_box1
"$PYTHON_BIN" run_full_ft_quant_sweep.py \
  --output-root "$FULLFT_OUTPUT_ROOT" \
  --cache-root "$NVME_ROOT/hf_cache" \
  --models llama_1b \
  --device cuda \
  --proxy-device cuda \
  --skip-proxy-quant-stats \
  | tee "$LOG_ROOT/fullft_llama_1b_box1.log"
