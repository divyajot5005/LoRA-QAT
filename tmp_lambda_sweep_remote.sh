set -euo pipefail
cd /root/LoRA-QAT
ROOT=/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8
mkdir -p "$ROOT/configs" "$ROOT/runs"
python - <<'PY'
import json, os
root = '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8'
config_dir = os.path.join(root, 'configs')
os.makedirs(config_dir, exist_ok=True)
base = {
  'seed': 7,
  'task_type': 'causal_lm',
  'model_name': 'meta-llama/Llama-3.2-1B-Instruct',
  'dataset_name': 'databricks/databricks-dolly-15k',
  'dataset_config_name': None,
  'train_split': 'train[:70%]',
  'eval_split': 'train[90%:95%]',
  'num_steps': 400,
  'batch_size': 8,
  'gradient_accumulation_steps': 4,
  'learning_rate': 2e-4,
  'mixed_precision': 'bf16',
  'enable_quant_lora_regularization': True,
  'group_size': 128,
  'regularization_objective': 'weight_mse',
}
lambdas = [1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
variants = [
  ('int4', 'uniform_groupwise', 4),
  ('int8', 'uniform_groupwise', 8),
]
for label, quantizer, bit_width in variants:
    for lam in lambdas:
        lam_tag = format(lam, '.0e').replace('+', '')
        run_name = f'llama32_1b__instruction_tuning__{label}__lambda_{lam_tag}'
        cfg = dict(base)
        cfg['lambda_q'] = lam
        cfg['quantizer_type'] = quantizer
        cfg['bit_width'] = bit_width
        cfg['output_dir'] = os.path.join(root, 'runs', run_name)
        path = os.path.join(config_dir, f'{run_name}.json')
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)
print('WROTE_CONFIGS', len(lambdas) * len(variants))
PY
cat > "$ROOT/run_batch.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /root/LoRA-QAT
ROOT=/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8
for cfg in "$ROOT"/configs/*.json; do
  run_name=$(basename "$cfg" .json)
  out_dir="$ROOT/runs/$run_name"
  mkdir -p "$out_dir"
  if [ -f "$out_dir/final_metrics.json" ]; then
    echo "SKIP $(date -Is) $run_name already_done"
    continue
  fi
  echo "START $(date -Is) $run_name"
  /opt/miniconda/envs/ndna/bin/python /root/LoRA-QAT/train_hf.py --config "$cfg" 2>&1 | tee "$out_dir/train.log"
  status=${PIPESTATUS[0]}
  echo "END $(date -Is) $run_name status=$status"
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
done
SH
chmod +x "$ROOT/run_batch.sh"
if pgrep -f '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8/run_batch.sh' >/dev/null; then
  echo 'SWEEP_ALREADY_RUNNING'
else
  nohup bash "$ROOT/run_batch.sh" > "$ROOT/sweep.log" 2>&1 &
  sleep 2
  echo "SWEEP_PID $(pgrep -f '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8/run_batch.sh' | head -n1)"
fi
if [ -f "$ROOT/sweep.log" ]; then tail -n 20 "$ROOT/sweep.log"; fi