#!/usr/bin/env bash
set -euo pipefail

pkill -f '/home/ubuntu/loraqat_outputs/logit_kl_multiprecision_llama32/run_batch.sh' || true
pkill -f 'train_hf.py --config /home/ubuntu/loraqat_outputs/logit_kl_multiprecision_llama32' || true

cd /home/ubuntu/LoRA-QAT
source /home/ubuntu/venvs/loraqat/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - <<'PY'
import json
from pathlib import Path

base_path = Path("/home/ubuntu/LoRA-QAT/outputs/cloud_instruction_suite/configs/llama32_1b__instruction_tuning__quant_int4_ft.json")
base = json.loads(base_path.read_text(encoding="utf-8"))
base["training"]["output_dir"] = "/home/ubuntu/loraqat_outputs/llama32_no_quant_baseline"
base["quant_regularization"]["enable_quant_lora_regularization"] = False
base["quant_regularization"]["lambda_q"] = 0.0
base["quant_regularization"]["quantizer_type"] = "uniform_groupwise"
base["quant_regularization"]["bit_width"] = 4
base["quant_regularization"]["group_size"] = 128
base["quant_regularization"]["regularization_objective"] = "weight_mse"
base["quant_regularization"]["regularization_frequency"] = 1
base["quant_regularization"]["detach_layer_inputs"] = True
base["quant_regularization"]["max_activation_regularized_layers"] = 0
base["quant_regularization"]["log_per_layer_stats"] = False

cfg_path = Path("/home/ubuntu/loraqat_outputs/llama32_no_quant_baseline.json")
cfg_path.write_text(json.dumps(base, indent=2), encoding="utf-8")
print(cfg_path)
PY

python /home/ubuntu/LoRA-QAT/train_hf.py --config /home/ubuntu/loraqat_outputs/llama32_no_quant_baseline.json

python /home/ubuntu/LoRA-QAT/evaluate_quantized_proxy.py \
  --config /home/ubuntu/loraqat_outputs/llama32_no_quant_baseline.json \
  --adapter-dir /home/ubuntu/loraqat_outputs/llama32_no_quant_baseline/adapter \
  --output /home/ubuntu/loraqat_outputs/llama32_no_quant_baseline/quantized_proxy_metrics__int2.json \
  --quantizer-type uniform_int2_groupwise \
  --bit-width 2 \
  --group-size 128

python /home/ubuntu/LoRA-QAT/evaluate_quantized_proxy.py \
  --config /home/ubuntu/loraqat_outputs/llama32_no_quant_baseline.json \
  --adapter-dir /home/ubuntu/loraqat_outputs/llama32_no_quant_baseline/adapter \
  --output /home/ubuntu/loraqat_outputs/llama32_no_quant_baseline/quantized_proxy_metrics__ternary.json \
  --quantizer-type ternary_groupwise \
  --bit-width 2 \
  --group-size 128
