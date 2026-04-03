set -euo pipefail
cd /root/LoRA-QAT
ROOT=/mnt/loraqat_outputs/lambda_sweep_llama32_int2_ternary
mkdir -p "$ROOT/configs" "$ROOT/runs"
python - <<'PY'
import json, copy, os
from pathlib import Path
base_path = Path('/root/LoRA-QAT/outputs/cloud_instruction_suite/configs/llama32_1b__instruction_tuning__quant_int4_ft.json')
with base_path.open('r', encoding='utf-8') as f:
    base = json.load(f)
base['quant_regularization']['regularization_objective'] = 'weight_mse'
base['quant_regularization']['regularization_frequency'] = 1
base['quant_regularization']['detach_layer_inputs'] = True
base['quant_regularization']['max_activation_regularized_layers'] = 0
base['quant_regularization']['log_per_layer_stats'] = False
root = Path('/mnt/loraqat_outputs/lambda_sweep_llama32_int2_ternary')
configs = []
variants = {
    'int2': {
        'quantizer_type': 'uniform_int2_groupwise',
        'bit_width': 2,
        'lambdas': [3e4, 1e5, 3e5, 1e6],
    },
    'ternary': {
        'quantizer_type': 'ternary_groupwise',
        'bit_width': 2,
        'lambdas': [1e4, 3e4, 1e5, 3e5],
    },
}
for label, spec in variants.items():
    for lam in spec['lambdas']:
        cfg = copy.deepcopy(base)
        lam_tag = format(lam, '.0e').replace('+', '')
        run_name = f'llama32_1b__instruction_tuning__{label}__lambda_{lam_tag}'
        cfg['training']['output_dir'] = str(root / 'runs' / run_name)
        cfg['quant_regularization']['enable_quant_lora_regularization'] = True
        cfg['quant_regularization']['lambda_q'] = lam
        cfg['quant_regularization']['quantizer_type'] = spec['quantizer_type']
        cfg['quant_regularization']['bit_width'] = spec['bit_width']
        cfg['quant_regularization']['group_size'] = 128
        path = root / 'configs' / f'{run_name}.json'
        with path.open('w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
        configs.append(str(path))
print('WROTE_CONFIGS', len(configs))
PY
cat > "$ROOT/run_batch.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /root/LoRA-QAT
ROOT=/mnt/loraqat_outputs/lambda_sweep_llama32_int2_ternary
wait_for_gpu() {
  while true; do
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
    if [ -n "$free_mem" ] && [ "$free_mem" -ge 43000 ]; then
      break
    fi
    echo "WAIT $(date -Is) free_mem_mb=${free_mem:-unknown}" >> "$ROOT/sweep.log"
    sleep 60
  done
}
for cfg in "$ROOT"/configs/*.json; do
  run_name=$(basename "$cfg" .json)
  out_dir="$ROOT/runs/$run_name"
  mkdir -p "$out_dir"
  if [ -f "$out_dir/final_metrics.json" ]; then
    echo "SKIP $(date -Is) $run_name already_done" >> "$ROOT/sweep.log"
    continue
  fi
  wait_for_gpu
  echo "START $(date -Is) $run_name" >> "$ROOT/sweep.log"
  PYTHONPATH=/root/LoRA-QAT /opt/miniconda/envs/ndna/bin/python /root/LoRA-QAT/train_hf.py --config "$cfg" 2>&1 | tee "$out_dir/train.log"
  status=${PIPESTATUS[0]}
  echo "END $(date -Is) $run_name status=$status" >> "$ROOT/sweep.log"
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
done
SH
chmod +x "$ROOT/run_batch.sh"
if pgrep -f '/mnt/loraqat_outputs/lambda_sweep_llama32_int2_ternary/run_batch.sh' >/dev/null; then
  echo 'LOWBIT_SWEEP_ALREADY_RUNNING'
else
  nohup bash "$ROOT/run_batch.sh" > /dev/null 2>&1 &
  sleep 2
  pgrep -af '/mnt/loraqat_outputs/lambda_sweep_llama32_int2_ternary/run_batch.sh' || true
fi
tail -n 20 "$ROOT/sweep.log" || true