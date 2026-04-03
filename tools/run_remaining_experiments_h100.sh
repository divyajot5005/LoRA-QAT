#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/LoRA-QAT}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda/envs/ndna/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/work/loraqat_outputs}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/work/hf_cache}"
LOG_ROOT="${LOG_ROOT:-/mnt/work/loraqat_logs}"
HF_TOKEN_VALUE="${HF_TOKEN:-}"

mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT" "$LOG_ROOT" \
  "$CACHE_ROOT/hub" "$CACHE_ROOT/datasets" "$CACHE_ROOT/transformers"

export HF_HOME="$CACHE_ROOT"
export HF_HUB_CACHE="$CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"

if [[ -n "$HF_TOKEN_VALUE" ]]; then
  mkdir -p "$HOME/.cache/huggingface"
  printf '%s' "$HF_TOKEN_VALUE" > "$HOME/.cache/huggingface/token"
  chmod 600 "$HOME/.cache/huggingface/token"
fi

cd "$REPO_DIR"

run_step() {
  local name="$1"
  shift
  local log_path="$LOG_ROOT/${name}.log"
  echo "START ${name} $(date -Is)" | tee -a "$log_path"
  "$PYTHON_BIN" "$@" 2>&1 | tee -a "$log_path"
  echo "DONE ${name} $(date -Is)" | tee -a "$log_path"
}

# Main LoRA matrix:
# - instruction tuning
# - all currently relevant Llama/Qwen/Gemma families
# - runner is resumable and now continues past failed runs
run_step "remaining_lora_main" \
  run_h100_quant_lora_suite.py \
  --output-root "$OUTPUT_ROOT/lora_remaining_main" \
  --cache-root "$CACHE_ROOT" \
  --models \
    gemma3_1b llama32_1b qwen3_0p6b \
    llama32_3b qwen3_4b gemma3_4b \
    llama31_8b qwen3_8b gemma2_9b \
  --tasks instruction_tuning \
  --phases before_ft no_quant_ft quant_int4_ft quant_fp8_ft quant_int4_act_ft quant_fp8_act_ft

# Robustness LoRA slice:
# - one family across 1B / 3B / 8B
# - second task to test whether the story transfers beyond instruction tuning
run_step "remaining_lora_robustness_llama_code" \
  run_h100_quant_lora_suite.py \
  --output-root "$OUTPUT_ROOT/lora_robustness_llama_code" \
  --cache-root "$CACHE_ROOT" \
  --models llama32_1b llama32_3b llama31_8b \
  --tasks domain_adaptation_code \
  --phases before_ft no_quant_ft quant_int4_ft quant_fp8_ft quant_int4_act_ft quant_fp8_act_ft

# Remaining 1B full-FT matrix:
# - resume-safe
# - all four quantizers and their lambda sweeps
run_step "remaining_fullft_1b" \
  run_full_ft_quant_sweep.py \
  --output-root "$OUTPUT_ROOT/fullft_remaining_1b" \
  --cache-root "$CACHE_ROOT" \
  --models llama_1b qwen3_0p6b gemma_1b \
  --device cuda \
  --proxy-device cuda \
  --skip-proxy-quant-stats
