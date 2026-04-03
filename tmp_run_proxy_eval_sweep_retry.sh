set -euo pipefail
cd /root/LoRA-QAT
ROOT=/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2
BASECFG=/mnt/loraqat_outputs/gated_instruction_suite/configs/llama32_1b__instruction_tuning__no_quant_ft.json
BASERUN=/mnt/loraqat_outputs/highlambda_instruction_suite/runs/llama32_1b__instruction_tuning__no_quant_ft/adapter
run_eval() {
  cfg="$1"
  adapter="$2"
  out="$3"
  qtype="$4"
  bw="$5"
  gsize="$6"
  if [ -f "$out" ]; then
    echo "SKIP $(basename "$out")"
    return 0
  fi
  /opt/miniconda/envs/ndna/bin/python /root/LoRA-QAT/evaluate_quantized_proxy.py \
    --config "$cfg" \
    --adapter-dir "$adapter" \
    --output "$out" \
    --quantizer-type "$qtype" \
    --bit-width "$bw" \
    --group-size "$gsize"
}
run_eval \
  "$BASECFG" \
  "$BASERUN" \
  "/mnt/loraqat_outputs/highlambda_instruction_suite/runs/llama32_1b__instruction_tuning__no_quant_ft/quantized_proxy_metrics__int4_group128.json" \
  uniform_groupwise 4 128
for run in \
  llama32_1b__instruction_tuning__int4__lambda_1e06 \
  llama32_1b__instruction_tuning__int4__lambda_1e07
  do
  run_eval \
    "$ROOT/configs/${run}.json" \
    "$ROOT/runs/${run}/adapter" \
    "$ROOT/runs/${run}/quantized_proxy_metrics__int4_group128.json" \
    uniform_groupwise 4 128
 done
python - <<'PY'
import json, os
rows = [
('baseline_no_quant', '/mnt/loraqat_outputs/highlambda_instruction_suite/runs/llama32_1b__instruction_tuning__no_quant_ft/final_metrics.json', '/mnt/loraqat_outputs/highlambda_instruction_suite/runs/llama32_1b__instruction_tuning__no_quant_ft/quantized_proxy_metrics__int4_group128.json'),
('int4_lambda_1e06', '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/runs/llama32_1b__instruction_tuning__int4__lambda_1e06/final_metrics.json', '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/runs/llama32_1b__instruction_tuning__int4__lambda_1e06/quantized_proxy_metrics__int4_group128.json'),
('int4_lambda_1e07', '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/runs/llama32_1b__instruction_tuning__int4__lambda_1e07/final_metrics.json', '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/runs/llama32_1b__instruction_tuning__int4__lambda_1e07/quantized_proxy_metrics__int4_group128.json'),
]
for name, fpath, qpath in rows:
    out = {'run': name}
    if os.path.exists(fpath):
        with open(fpath) as f:
            d = json.load(f)
        fm = d.get('final_metrics', d)
        out['float_loss'] = fm.get('eval_loss')
        out['float_ppl'] = fm.get('eval_perplexity')
    if os.path.exists(qpath):
        with open(qpath) as f:
            d = json.load(f)
        out['quant_loss'] = d.get('eval_loss')
        out['quant_ppl'] = d.get('eval_perplexity')
        out['raw_quant_error'] = d.get('raw_quant_error')
        if out.get('float_loss') is not None and out.get('quant_loss') is not None:
            out['quant_gap'] = out['quant_loss'] - out['float_loss']
    print(json.dumps(out))
PY