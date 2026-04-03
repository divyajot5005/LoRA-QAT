set -euo pipefail
cd /root/LoRA-QAT
ROOT=/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2
for run in \
  llama32_1b__instruction_tuning__int4__lambda_1e08 \
  llama32_1b__instruction_tuning__int4__lambda_1e09 \
  llama32_1b__instruction_tuning__int4__lambda_1e10
  do
  cfg="$ROOT/configs/${run}.json"
  adapter="$ROOT/runs/${run}/adapter"
  out="$ROOT/runs/${run}/quantized_proxy_metrics__int4_group128.json"
  if [ -f "$out" ]; then
    echo "SKIP $run"
  else
    echo "EVAL $run"
    /opt/miniconda/envs/ndna/bin/python /root/LoRA-QAT/evaluate_quantized_proxy.py \
      --config "$cfg" \
      --adapter-dir "$adapter" \
      --output "$out" \
      --quantizer-type uniform_groupwise \
      --bit-width 4 \
      --group-size 128
  fi
done
BEFORE_OUT=/mnt/loraqat_outputs/gated_instruction_suite/runs/llama32_1b__instruction_tuning__before_ft/proxy_quantized_base__int4_group128.json
if [ ! -f "$BEFORE_OUT" ]; then
python - <<'PY'
import json
from pathlib import Path
import torch
from quant_lora.hf_config import load_hf_experiment_config
from quant_lora.hf_data import build_dataloaders
from quant_lora.hf_model import build_model, load_tokenizer
from quant_lora.hf_trainer import _evaluate, _resolve_device, _set_seed
from quant_lora.quantization import quantize_merged_weight
cfg_path = '/mnt/loraqat_outputs/gated_instruction_suite/configs/llama32_1b__instruction_tuning__before_ft.json'
out_path = Path('/mnt/loraqat_outputs/gated_instruction_suite/runs/llama32_1b__instruction_tuning__before_ft/proxy_quantized_base__int4_group128.json')
out_path.parent.mkdir(parents=True, exist_ok=True)
config = load_hf_experiment_config(cfg_path)
_set_seed(config.seed)
device = _resolve_device(config.training.device)
tokenizer = load_tokenizer(config)
data_bundle = build_dataloaders(config, tokenizer)
config.lora.enabled = False
model = build_model(config, data_bundle.num_labels).to(device)
if config.model.task_type == 'causal_lm' and getattr(model.config, 'pad_token_id', None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id
active = _evaluate(model, data_bundle.eval_loader, device, config, desc='base_eval')
quantized_layers = 0
for name, module in model.named_modules():
    weight = getattr(module, 'weight', None)
    if weight is None or weight.ndim != 2:
        continue
    leaf = name.split('.')[-1]
    if leaf not in set(config.lora.target_modules):
        continue
    q = quantize_merged_weight(module.weight.data.float(), quantizer_type='uniform_groupwise', bit_width=4, group_size=128)
    module.weight.data.copy_(q.to(device=module.weight.device, dtype=module.weight.dtype))
    quantized_layers += 1
proxy = _evaluate(model, data_bundle.eval_loader, device, config, desc='base_proxy_quant_eval')
payload = {
    'config_path': str(Path(cfg_path).resolve()),
    'task_type': config.model.task_type,
    'model_name': config.model.model_name,
    'dataset_name': config.dataset.dataset_name,
    'quantized_proxy_target': {'quantizer_type':'uniform_groupwise','bit_width':4,'group_size':128},
    'active_metrics': active,
    'proxy_quantized_metrics': proxy,
    'eval_loss_delta': float(proxy['eval_loss'] - active['eval_loss']),
    'num_proxy_quantized_layers': quantized_layers,
}
with out_path.open('w', encoding='utf-8') as f:
    json.dump(payload, f, indent=2)
print(json.dumps(payload, indent=2))
PY
fi
python - <<'PY'
import json, os
rows = [
('before_ft_base', '/mnt/loraqat_outputs/gated_instruction_suite/runs/llama32_1b__instruction_tuning__before_ft/proxy_quantized_base__int4_group128.json', None),
('baseline_no_quant', '/mnt/loraqat_outputs/highlambda_instruction_suite/runs/llama32_1b__instruction_tuning__no_quant_ft/final_metrics.json', '/mnt/loraqat_outputs/highlambda_instruction_suite/runs/llama32_1b__instruction_tuning__no_quant_ft/quantized_proxy_metrics__int4_group128.json'),
('int4_lambda_1e08', '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/runs/llama32_1b__instruction_tuning__int4__lambda_1e08/final_metrics.json', '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/runs/llama32_1b__instruction_tuning__int4__lambda_1e08/quantized_proxy_metrics__int4_group128.json'),
('int4_lambda_1e09', '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/runs/llama32_1b__instruction_tuning__int4__lambda_1e09/final_metrics.json', '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/runs/llama32_1b__instruction_tuning__int4__lambda_1e09/quantized_proxy_metrics__int4_group128.json'),
('int4_lambda_1e10', '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/runs/llama32_1b__instruction_tuning__int4__lambda_1e10/final_metrics.json', '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/runs/llama32_1b__instruction_tuning__int4__lambda_1e10/quantized_proxy_metrics__int4_group128.json'),
]
for name, fpath, qpath in rows:
    out = {'run': name}
    if qpath is None:
        with open(fpath) as f:
            d = json.load(f)
        a = d['active_metrics']; q = d['proxy_quantized_metrics']
        out['float_loss'] = a.get('eval_loss'); out['float_ppl'] = a.get('eval_perplexity')
        out['quant_loss'] = q.get('eval_loss'); out['quant_ppl'] = q.get('eval_perplexity')
        out['quant_gap'] = d.get('eval_loss_delta')
        out['raw_quant_error'] = None
    else:
        if os.path.exists(fpath):
            with open(fpath) as f:
                d = json.load(f)
            fm = d.get('final_metrics', d)
            out['float_loss'] = fm.get('eval_loss'); out['float_ppl'] = fm.get('eval_perplexity')
        if os.path.exists(qpath):
            with open(qpath) as f:
                d = json.load(f)
            q = d['proxy_quantized_metrics']
            out['quant_loss'] = q.get('eval_loss'); out['quant_ppl'] = q.get('eval_perplexity')
            out['quant_gap'] = d.get('eval_loss_delta')
            out['raw_quant_error'] = d.get('raw_quant_error')
    print(json.dumps(out))
PY