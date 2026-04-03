#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda/envs/ndna/bin/python}"
REPO_DIR="${REPO_DIR:-/root/Quantization-through-LoRA}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cloud_instruction_suite}"
CACHE_ROOT="${CACHE_ROOT:-}"

cd "${REPO_DIR}"

cmd=(
  "${PYTHON_BIN}" run_h100_quant_lora_suite.py
  --output-root "${OUTPUT_ROOT}"
  --models gemma3_1b llama32_1b llama31_8b
  --tasks instruction_tuning
  --phases before_ft no_quant_ft quant_int4_ft quant_fp8_ft quant_int4_act_ft quant_fp8_act_ft
)

if [[ -n "${CACHE_ROOT}" ]]; then
  cmd+=(--cache-root "${CACHE_ROOT}")
fi

"${cmd[@]}"
