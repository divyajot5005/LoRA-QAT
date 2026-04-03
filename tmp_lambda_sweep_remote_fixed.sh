set -euo pipefail
cd /root/LoRA-QAT
ROOT=/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2
mkdir -p \"/configs\" \"/runs\"
cat > /tmp/template_int4.json <<'JSON1'

JSON1
cat > /tmp/template_int8.json <<'JSON2'
{
  "seed": 7,
  "model": {
    "task_type": "causal_lm",
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "max_length": 384,
    "gradient_checkpointing": true,
    "torch_dtype": "bfloat16",
    "attn_implementation": "sdpa"
  },
  "dataset": {
    "dataset_name": "databricks/databricks-dolly-15k",
    "dataset_config_name": null,
    "train_split": "train[:70%]",
    "eval_split": "train[90%:95%]",
    "max_train_samples": 9000,
    "max_eval_samples": 1000,
    "format_style": "instruction_response",
    "text_column": "instruction",
    "label_column": "label",
    "prompt_column": "instruction",
    "response_column": "response",
    "context_column": "context",
    "mask_prompt_tokens": true
  },
  "lora": {
    "enabled": true,
    "target_modules": [
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    "rank": 16,
    "alpha": 32.0,
    "dropout": 0.05
  },
  "training": {
    "output_dir": "outputs\\activation_suite_smoke\\runs\\llama32_1b__instruction_tuning__quant_fp8_act_ft",
    "device": "auto",
    "mixed_precision": "bf16",
    "num_steps": 400,
    "batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 0.0002,
    "weight_decay": 0.0,
    "warmup_steps": 20,
    "log_interval": 40,
    "eval_interval": 80,
    "max_eval_batches": null,
    "generation_prompts": [
      "Instruction:\nDraft a concise project-status update for a manager after a one-week delay.\n\nResponse:\n",
      "Instruction:\nExplain the difference between precision and recall in machine learning.\n\nResponse:\n",
      "Instruction:\nWrite a short, friendly email asking to reschedule a meeting.\n\nResponse:\n"
    ],
    "generation_max_new_tokens": 96,
    "generation_temperature": 0.7,
    "generation_top_p": 0.9
  },
  "quant_regularization": {
    "enable_quant_lora_regularization": true,
    "lambda_q": 0.003,
    "quantizer_type": "fp8_e4m3fn",
    "bit_width": 8,
    "group_size": 128,
    "regularization_objective": "activation_mse",
    "regularization_frequency": 1,
    "detach_layer_inputs": true,
    "max_activation_regularized_layers": 8,
    "log_per_layer_stats": false
  }
}
JSON2
python - <<'PY'
import json, os
root = '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2'
config_dir = os.path.join(root, 'configs')
os.makedirs(config_dir, exist_ok=True)
with open('/tmp/template_int4.json', 'r', encoding='utf-8') as f:
    template_int4 = json.load(f)
with open('/tmp/template_int8.json', 'r', encoding='utf-8') as f:
    template_int8 = json.load(f)
# force weight_mse and proper output roots
for tpl in (template_int4, template_int8):
    tpl['quant_regularization']['regularization_objective'] = 'weight_mse'
    tpl['quant_regularization']['regularization_frequency'] = 1
    tpl['quant_regularization']['detach_layer_inputs'] = True
    tpl['quant_regularization']['max_activation_regularized_layers'] = 0
    tpl['quant_regularization']['log_per_layer_stats'] = False
lambdas = [1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
variants = [
    ('int4', template_int4, 'uniform_groupwise', 4),
    ('int8', template_int8, 'uniform_groupwise', 8),
]
for label, tpl, qtype, bit_width in variants:
    for lam in lambdas:
        cfg = json.loads(json.dumps(tpl))
        lam_tag = format(lam, '.0e').replace('+', '')
        run_name = f'llama32_1b__instruction_tuning__{label}__lambda_{lam_tag}'
        cfg['training']['output_dir'] = os.path.join(root, 'runs', run_name)
        cfg['quant_regularization']['enable_quant_lora_regularization'] = True
        cfg['quant_regularization']['lambda_q'] = lam
        cfg['quant_regularization']['quantizer_type'] = qtype
        cfg['quant_regularization']['bit_width'] = bit_width
        cfg['quant_regularization']['group_size'] = 128
        path = os.path.join(config_dir, f'{run_name}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
print('WROTE_CONFIGS', len(lambdas) * len(variants))
PY
cat > \"/run_batch.sh\" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /root/LoRA-QAT
ROOT=/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2
for cfg in ""/configs/*.json; do
  run_name=
  out_dir="/runs/"
  mkdir -p ""
  if [ -f "/final_metrics.json" ]; then
    echo "SKIP   already_done"
    continue
  fi
  echo "START  "
  /opt/miniconda/envs/ndna/bin/python /root/LoRA-QAT/train_hf.py --config "" 2>&1 | tee "/train.log"
  status=
  echo "END   status="
  if [ "" -ne 0 ]; then
    exit ""
  fi
done
SH
chmod +x \"/run_batch.sh\"
pkill -f '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8/run_batch.sh' || true
if pgrep -f '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/run_batch.sh' >/dev/null; then
  echo 'SWEEP_ALREADY_RUNNING'
else
  nohup bash \"/run_batch.sh\" > \"/sweep.log\" 2>&1 &
  sleep 2
  echo \"SWEEP_PID \"
fi
tail -n 20 \"/sweep.log\" || true