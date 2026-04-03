set -euo pipefail
cd /root/LoRA-QAT
PYTHONPATH=/root/LoRA-QAT /opt/miniconda/envs/ndna/bin/python - <<'PY'
import copy, json
from quant_lora.hf_config import load_hf_experiment_config
from quant_lora.hf_model import build_model
from quant_lora.peft_regularization import compute_quant_lora_regularization
cfg_path = '/mnt/loraqat_outputs/gated_instruction_suite/configs/llama32_1b__instruction_tuning__before_ft.json'
config = load_hf_experiment_config(cfg_path)
config.training.device = 'cpu'
config.model.torch_dtype = 'float32'
config.model.gradient_checkpointing = False
model = build_model(config, None)
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
        'raw_loss_step0': raw,
        'lambda_eq_task2p4': 2.4 / raw if raw > 0 else None,
    })
print(json.dumps(rows, indent=2))
PY