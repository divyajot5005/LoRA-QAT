set -euo pipefail
cd /root/LoRA-QAT
ROOT=/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2
BASECFG=/mnt/loraqat_outputs/gated_instruction_suite/configs/llama32_1b__instruction_tuning__no_quant_ft.json
BASERUN=/mnt/loraqat_outputs/highlambda_instruction_suite/runs/llama32_1b__instruction_tuning__no_quant_ft/adapter
for run in \
  llama32_1b__instruction_tuning__int4__lambda_1e06 \
  llama32_1b__instruction_tuning__int4__lambda_1e07
  do
  cfg="$ROOT/configs/${run}.json"
  adapter="$ROOT/runs/${run}/adapter"
  out="$ROOT/runs/${run}/quantized_proxy_metrics__int4_group128.json"
  if [ -f "$out" ]; then
    echo "SKIP $run"
    continue
  fi
  echo "EVAL $run"
  /opt/miniconda/envs/ndna/bin/python /root/LoRA-QAT/evaluate_quantized_proxy.py \
    --config "$cfg" \
    --adapter-dir "$adapter" \
    --output "$out" \
    --quantizer-type uniform_groupwise \
    --bit-width 4 \
    --group-size 128
 done
BASEOUT=/mnt/loraqat_outputs/highlambda_instruction_suite/runs/llama32_1b__instruction_tuning__no_quant_ft/quantized_proxy_metrics__int4_group128.json
if [ ! -f "$BASEOUT" ]; then
  echo "EVAL baseline"
  /opt/miniconda/envs/ndna/bin/python /root/LoRA-QAT/evaluate_quantized_proxy.py \
    --config "$BASECFG" \
    --adapter-dir "$BASERUN" \
    --output "$BASEOUT" \
    --quantizer-type uniform_groupwise \
    --bit-width 4 \
    --group-size 128
fi
python - <<'PY'
import json, os
paths = [
('/mnt/loraqat_outputs/highlambda_instruction_suite/runs/llama32_1b__instruction_tuning__no_quant_ft/final_metrics.json', '/mnt/loraqat_outputs/highlambda_instruction_suite/runs/llama32_1b__instruction_tuning__no_quant_ft/quantized_proxy_metrics__int4_group128.json', 'baseline_no_quant'),
('/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/runs/llama32_1b__instruction_tuning__int4__lambda_1e06/final_metrics.json', '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/runs/llama32_1b__instruction_tuning__int4__lambda_1e06/quantized_proxy_metrics__int4_group128.json', 'int4_lambda_1e06'),
('/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/runs/llama32_1b__instruction_tuning__int4__lambda_1e07/final_metrics.json', '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/runs/llama32_1b__instruction_tuning__int4__lambda_1e07/quantized_proxy_metrics__int4_group128.json', 'int4_lambda_1e07'),
]
for fpath, qpath, name in paths:
    row = {'run': name}
    if os.path.exists(fpath):
        with open(fpath) as f:
            d = json.load(f)
        fm = d.get('final_metrics', d)
        row['eval_loss'] = fm.get('eval_loss')
        row['eval_perplexity'] = fm.get('eval_perplexity')
    if os.path.exists(qpath):
        with open(qpath) as f:
            d = json.load(f)
        row['proxy_loss'] = d.get('eval_loss')
        row['proxy_perplexity'] = d.get('eval_perplexity')
        row['raw_quant_error'] = d.get('raw_quant_error')
        if row.get('eval_loss') is not None and row.get('proxy_loss') is not None:
            row['loss_delta_after_quant'] = row['proxy_loss'] - row['eval_loss']
    print(json.dumps(row))
PY