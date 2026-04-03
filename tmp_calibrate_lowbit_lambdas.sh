set -euo pipefail
cd /root/LoRA-QAT
PYTHONPATH=/root/LoRA-QAT /opt/miniconda/envs/ndna/bin/python - <<'PY'
import copy, json
from quant_lora.hf_config import load_hf_experiment_config
from quant_lora.hf_data import build_dataloaders
from quant_lora.hf_model import build_model, load_tokenizer
from quant_lora.hf_trainer import _resolve_device, _set_seed
from quant_lora.peft_regularization import compute_quant_lora_regularization
cfg_path = '/mnt/loraqat_outputs/gated_instruction_suite/configs/llama32_1b__instruction_tuning__before_ft.json'
config = load_hf_experiment_config(cfg_path)
_set_seed(config.seed)
device = _resolve_device(config.training.device)
tokenizer = load_tokenizer(config)
data = build_dataloaders(config, tokenizer)
model = build_model(config, data.num_labels).to(device)
if config.model.task_type == 'causal_lm' and getattr(model.config, 'pad_token_id', None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id
batch = next(iter(data.train_loader))
batch = {k: v.to(device) for k, v in batch.items()}
outputs = model(**batch)
task_loss = float(outputs.loss.detach().cpu().item())
variants = [
    ('int4', 'uniform_groupwise', 4),
    ('int8', 'fp8_e4m3fn', 8),
    ('int2', 'uniform_int2_groupwise', 2),
    ('ternary', 'ternary_groupwise', 2),
]
rows = []
for name, qtype, bit_width in variants:
    cfg = copy.deepcopy(config.quant_regularization)
    cfg.enable_quant_lora_regularization = True
    cfg.quantizer_type = qtype
    cfg.bit_width = bit_width
    cfg.group_size = 128
    cfg.regularization_objective = 'weight_mse'
    res = compute_quant_lora_regularization(model, cfg)
    raw = float(res.raw_loss.detach().cpu().item())
    rows.append({
        'name': name,
        'quantizer_type': qtype,
        'bit_width': bit_width,
        'task_loss_step0': task_loss,
        'raw_loss_step0': raw,
        'lambda_task_equal': task_loss / raw if raw > 0 else None,
    })
print(json.dumps(rows, indent=2))
PY