#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda/envs/ndna/bin/python}"
REPO_DIR="${REPO_DIR:-/root/Quantization-through-LoRA}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cloud_instruction_suite}"

cd "${REPO_DIR}"

"${PYTHON_BIN}" run_proxy_quant_eval_suite.py \
  --output-root "${OUTPUT_ROOT}" \
  --models gemma3_1b llama32_1b llama31_8b \
  --tasks instruction_tuning
