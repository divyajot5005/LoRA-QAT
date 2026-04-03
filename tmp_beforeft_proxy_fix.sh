set -euo pipefail
cd /root/LoRA-QAT
/opt/miniconda/envs/ndna/bin/python - <<'PY'
import json
from pathlib import Path
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
targets = set(config.lora.target_modules)
quantized_layers = 0
for name, module in model.named_modules():
    weight = getattr(module, 'weight', None)
    if weight is None or weight.ndim != 2:
        continue
    if name.split('.')[-1] not in targets:
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
print(json.dumps(payload))
PY