set -euo pipefail
cd /root/LoRA-QAT
ROOT=/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2
cat > /tmp/run_full_proxy_matrix.py <<'PY'
import json, os, sys, subprocess
from pathlib import Path
sys.path.insert(0, '/root/LoRA-QAT')
from quant_lora.hf_config import load_hf_experiment_config
from quant_lora.hf_data import build_dataloaders
from quant_lora.hf_model import build_model, load_tokenizer
from quant_lora.hf_trainer import _evaluate, _resolve_device, _set_seed
from quant_lora.quantization import quantize_merged_weight

ROOT = Path('/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2')
SUMMARY_DIR = ROOT / 'proxy_matrix'
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
base_cfg = Path('/mnt/loraqat_outputs/gated_instruction_suite/configs/llama32_1b__instruction_tuning__before_ft.json')
lora_cfg = Path('/mnt/loraqat_outputs/gated_instruction_suite/configs/llama32_1b__instruction_tuning__no_quant_ft.json')
lora_adapter = Path('/mnt/loraqat_outputs/highlambda_instruction_suite/runs/llama32_1b__instruction_tuning__no_quant_ft/adapter')

def eval_base_quantized(cfg_path: Path, bit_width: int, out_path: Path):
    if out_path.exists():
        with out_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    config = load_hf_experiment_config(str(cfg_path))
    _set_seed(config.seed)
    device = _resolve_device(config.training.device)
    tokenizer = load_tokenizer(config)
    data_bundle = build_dataloaders(config, tokenizer)
    config.lora.enabled = False
    model = build_model(config, data_bundle.num_labels).to(device)
    if config.model.task_type == 'causal_lm' and getattr(model.config, 'pad_token_id', None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    active = _evaluate(model, data_bundle.eval_loader, device, config, desc='base_eval')
    targets = set(config.lora.target_modules)
    quantized_layers = 0
    for name, module in model.named_modules():
        weight = getattr(module, 'weight', None)
        if weight is None or weight.ndim != 2:
            continue
        if name.split('.')[-1] not in targets:
            continue
        q = quantize_merged_weight(module.weight.data.float(), quantizer_type='uniform_groupwise', bit_width=bit_width, group_size=128)
        module.weight.data.copy_(q.to(device=module.weight.device, dtype=module.weight.dtype))
        quantized_layers += 1
    proxy = _evaluate(model, data_bundle.eval_loader, device, config, desc='base_proxy_quant_eval')
    payload = {
        'active_metrics': active,
        'proxy_quantized_metrics': proxy,
        'eval_loss_delta': float(proxy['eval_loss'] - active['eval_loss']),
        'num_proxy_quantized_layers': quantized_layers,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return payload

def eval_adapter_quantized(cfg_path: Path, adapter_dir: Path, bit_width: int, out_path: Path):
    if out_path.exists():
        with out_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    cmd = [
        '/opt/miniconda/envs/ndna/bin/python', '/root/LoRA-QAT/evaluate_quantized_proxy.py',
        '--config', str(cfg_path), '--adapter-dir', str(adapter_dir), '--output', str(out_path),
        '--quantizer-type', 'uniform_groupwise', '--bit-width', str(bit_width), '--group-size', '128',
    ]
    subprocess.run(cmd, check=True)
    with out_path.open('r', encoding='utf-8') as f:
        return json.load(f)

base4 = eval_base_quantized(base_cfg, 4, SUMMARY_DIR / 'before_ft__proxy_int4.json')
base8 = eval_base_quantized(base_cfg, 8, SUMMARY_DIR / 'before_ft__proxy_int8.json')
lora4 = eval_adapter_quantized(lora_cfg, lora_adapter, 4, SUMMARY_DIR / 'no_quant_ft__proxy_int4.json')
lora8 = eval_adapter_quantized(lora_cfg, lora_adapter, 8, SUMMARY_DIR / 'no_quant_ft__proxy_int8.json')
base_float = base4['active_metrics']['eval_loss']
lora_float = lora4['active_metrics']['eval_loss']
rows = []
for name, bit_width, payload, raw in [
    ('before_ft_base', 4, base4, None), ('before_ft_base', 8, base8, None),
    ('no_quant_ft', 4, lora4, lora4.get('raw_quant_error')), ('no_quant_ft', 8, lora8, lora8.get('raw_quant_error')),
]:
    rows.append({
        'model_variant': name, 'quant_target': f'int{bit_width}',
        'float_loss': payload['active_metrics']['eval_loss'], 'float_ppl': payload['active_metrics']['eval_perplexity'],
        'quant_loss': payload['proxy_quantized_metrics']['eval_loss'], 'quant_ppl': payload['proxy_quantized_metrics']['eval_perplexity'],
        'quant_gap': payload['eval_loss_delta'], 'raw_quant_error': raw,
        'delta_vs_fp_base': payload['proxy_quantized_metrics']['eval_loss'] - base_float,
        'delta_vs_fp_lora': payload['proxy_quantized_metrics']['eval_loss'] - lora_float,
    })
for cfg_path in sorted((ROOT / 'configs').glob('*.json')):
    run_name = cfg_path.stem
    adapter_dir = ROOT / 'runs' / run_name / 'adapter'
    final_metrics = ROOT / 'runs' / run_name / 'final_metrics.json'
    if not adapter_dir.exists() or not final_metrics.exists():
        continue
    for bit_width in (4, 8):
        payload = eval_adapter_quantized(cfg_path, adapter_dir, bit_width, SUMMARY_DIR / f'{run_name}__proxy_int{bit_width}.json')
        rows.append({
            'model_variant': run_name, 'quant_target': f'int{bit_width}',
            'float_loss': payload['active_metrics']['eval_loss'], 'float_ppl': payload['active_metrics']['eval_perplexity'],
            'quant_loss': payload['proxy_quantized_metrics']['eval_loss'], 'quant_ppl': payload['proxy_quantized_metrics']['eval_perplexity'],
            'quant_gap': payload['eval_loss_delta'], 'raw_quant_error': payload.get('raw_quant_error'),
            'delta_vs_fp_base': payload['proxy_quantized_metrics']['eval_loss'] - base_float,
            'delta_vs_fp_lora': payload['proxy_quantized_metrics']['eval_loss'] - lora_float,
        })
rows.sort(key=lambda r: (r['quant_target'], r['model_variant']))
with (SUMMARY_DIR / 'summary.json').open('w', encoding='utf-8') as f:
    json.dump(rows, f, indent=2)
with (SUMMARY_DIR / 'summary.csv').open('w', encoding='utf-8') as f:
    f.write('model_variant,quant_target,float_loss,float_ppl,quant_loss,quant_ppl,quant_gap,raw_quant_error,delta_vs_fp_base,delta_vs_fp_lora\n')
    for r in rows:
        f.write(','.join('' if r[k] is None else str(r[k]) for k in ['model_variant','quant_target','float_loss','float_ppl','quant_loss','quant_ppl','quant_gap','raw_quant_error','delta_vs_fp_base','delta_vs_fp_lora']) + '\n')
print('WROTE', len(rows), 'rows')
print(str(SUMMARY_DIR / 'summary.csv'))
PY
cat > "$ROOT/run_proxy_matrix.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /root/LoRA-QAT
PYTHONPATH=/root/LoRA-QAT /opt/miniconda/envs/ndna/bin/python /tmp/run_full_proxy_matrix.py
SH
chmod +x "$ROOT/run_proxy_matrix.sh"
pkill -f '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/run_proxy_matrix.sh' || true
nohup bash "$ROOT/run_proxy_matrix.sh" > "$ROOT/proxy_matrix.log" 2>&1 &
sleep 2
pgrep -af '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/run_proxy_matrix.sh' || true
tail -n 20 "$ROOT/proxy_matrix.log" || true