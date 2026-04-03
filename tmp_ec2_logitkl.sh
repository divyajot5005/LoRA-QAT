#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/LoRA-QAT
source /home/ubuntu/venvs/loraqat/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOWBIT_ROOT=/home/ubuntu/loraqat_outputs/lambda_sweep_llama32_int2_ternary
ROOT=/home/ubuntu/loraqat_outputs/logit_kl_multiprecision_llama32
mkdir -p "$ROOT/configs" "$ROOT/runs"

cat > "$ROOT/run_batch.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/LoRA-QAT
source /home/ubuntu/venvs/loraqat/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOWBIT_ROOT=/home/ubuntu/loraqat_outputs/lambda_sweep_llama32_int2_ternary
ROOT=/home/ubuntu/loraqat_outputs/logit_kl_multiprecision_llama32
LOG="$ROOT/sweep.log"

wait_for_gpu() {
  while true; do
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
    if [ -n "$free_mem" ] && [ "$free_mem" -ge 34000 ]; then
      break
    fi
    echo "WAIT_GPU $(date -Is) free_mem_mb=${free_mem:-unknown}" >> "$LOG"
    sleep 30
  done
}

wait_for_lowbit() {
  while pgrep -f '/home/ubuntu/loraqat_outputs/lambda_sweep_llama32_int2_ternary/run_batch.sh' >/dev/null 2>&1; do
    done_count=$(find "$LOWBIT_ROOT/runs" -name final_metrics.json 2>/dev/null | wc -l)
    total_count=$(find "$LOWBIT_ROOT/configs" -name '*.json' 2>/dev/null | wc -l)
    echo "WAIT_LOWBIT $(date -Is) completed=${done_count}/${total_count}" >> "$LOG"
    sleep 30
  done
}

echo "SCRIPT_START $(date -Is)" >> "$LOG"
wait_for_lowbit
wait_for_gpu

echo "CALIBRATE_START $(date -Is)" >> "$LOG"
python - <<'PY'
import copy
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from quant_lora.hf_config import load_hf_experiment_config
from quant_lora.hf_data import build_dataloaders
from quant_lora.hf_model import build_model, load_tokenizer
from quant_lora.hf_trainer import _move_batch_to_device, _resolve_autocast_dtype, _resolve_device, _set_seed
from quant_lora.peft_regularization import compute_logit_kl_loss, temporarily_patch_lora_merged_forwards

root = Path("/home/ubuntu/loraqat_outputs/logit_kl_multiprecision_llama32")
base_path = Path("/home/ubuntu/LoRA-QAT/outputs/cloud_instruction_suite/configs/llama32_1b__instruction_tuning__quant_int4_ft.json")
base = load_hf_experiment_config(str(base_path))
base.training.output_dir = str(root / "unused")
base.training.batch_size = 1
base.training.eval_batch_size = 1
base.training.gradient_accumulation_steps = 8
base.training.max_eval_batches = 1
base.training.num_steps = 1
base.training.log_interval = 1
base.training.eval_interval = 1
base.model.gradient_checkpointing = False
base.quant_regularization.enable_quant_lora_regularization = True
base.quant_regularization.regularization_objective = "logit_kl"
base.quant_regularization.logit_kl_temperature = 1.0

specs = {
    "int8": {"quantizer_type": "uniform_groupwise", "bit_width": 8, "group_size": 128},
    "int4": {"quantizer_type": "uniform_groupwise", "bit_width": 4, "group_size": 128},
    "int2": {"quantizer_type": "uniform_int2_groupwise", "bit_width": 2, "group_size": 128},
    "ternary": {"quantizer_type": "ternary_groupwise", "bit_width": 2, "group_size": 128},
}
scales = [1.0, 3.0, 10.0, 30.0]

_set_seed(base.seed)
device = _resolve_device(base.training.device)
tokenizer = load_tokenizer(base)
data_bundle = build_dataloaders(base, tokenizer)
model = build_model(base, data_bundle.num_labels).to(device)
if base.model.task_type == "causal_lm" and getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id
model.train()

batch = next(iter(data_bundle.train_loader))
batch = _move_batch_to_device(batch, device)
autocast_dtype = _resolve_autocast_dtype(base.training.mixed_precision)
autocast_context = torch.amp.autocast("cuda", dtype=autocast_dtype) if autocast_dtype is not None and device.type == "cuda" else nullcontext()

with autocast_context:
    active_outputs = model(**batch)
    task_loss = float(active_outputs.loss.detach().cpu().item())

calibration = {}
for name, spec in specs.items():
    base.quant_regularization.quantizer_type = spec["quantizer_type"]
    base.quant_regularization.bit_width = spec["bit_width"]
    base.quant_regularization.group_size = spec["group_size"]
    with torch.no_grad():
        with autocast_context:
            with temporarily_patch_lora_merged_forwards(model, base.quant_regularization, use_quantized=False):
                reference_outputs = model(**batch)
    with autocast_context:
        with temporarily_patch_lora_merged_forwards(model, base.quant_regularization, use_quantized=True):
            quantized_outputs = model(**batch)
        raw_kl = compute_logit_kl_loss(reference_outputs.logits, quantized_outputs.logits, batch.get("labels"), 1.0)
    raw_value = float(raw_kl.detach().cpu().item())
    parity_lambda = 1.0 if raw_value <= 0 else task_loss / raw_value
    lambdas = [min(max(parity_lambda * scale, 1e-2), 1e12) for scale in scales]
    deduped = []
    for value in lambdas:
        if not deduped or abs(value - deduped[-1]) / max(abs(deduped[-1]), 1.0) > 1e-9:
            deduped.append(value)
    while len(deduped) < 4:
        deduped.append(min(deduped[-1] * 3.0, 1e12))
    calibration[name] = {
        "task_loss": task_loss,
        "raw_logit_kl": raw_value,
        "parity_lambda": parity_lambda,
        "lambdas": deduped[:4],
        **spec,
    }

(root / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
base_payload = json.loads(base_path.read_text(encoding="utf-8"))
for name, row in calibration.items():
    for lam in row["lambdas"]:
        cfg = copy.deepcopy(base_payload)
        lam_tag = format(lam, ".2e").replace("+", "").replace(".", "p")
        run_name = f"llama32_1b__instruction_tuning__logitkl_{name}__lambda_{lam_tag}"
        cfg["training"]["output_dir"] = str(root / "runs" / run_name)
        cfg["training"]["batch_size"] = 1
        cfg["training"]["eval_batch_size"] = 1
        cfg["training"]["gradient_accumulation_steps"] = 16
        cfg["quant_regularization"]["enable_quant_lora_regularization"] = True
        cfg["quant_regularization"]["lambda_q"] = lam
        cfg["quant_regularization"]["quantizer_type"] = row["quantizer_type"]
        cfg["quant_regularization"]["bit_width"] = row["bit_width"]
        cfg["quant_regularization"]["group_size"] = row["group_size"]
        cfg["quant_regularization"]["regularization_objective"] = "logit_kl"
        cfg["quant_regularization"]["logit_kl_temperature"] = 1.0
        cfg["quant_regularization"]["regularization_frequency"] = 1
        cfg["quant_regularization"]["detach_layer_inputs"] = True
        cfg["quant_regularization"]["max_activation_regularized_layers"] = 0
        cfg["quant_regularization"]["log_per_layer_stats"] = False
        (root / "configs" / f"{run_name}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("CALIBRATION_DONE")
PY
echo "CALIBRATE_END $(date -Is)" >> "$LOG"

for cfg in "$ROOT"/configs/*.json; do
  run_name=$(basename "$cfg" .json)
  out_dir="$ROOT/runs/$run_name"
  mkdir -p "$out_dir"

  if [ -f "$out_dir/final_metrics.json" ] && [ -f "$out_dir/quantized_proxy_metrics__match.json" ]; then
    echo "SKIP $(date -Is) $run_name already_done" >> "$LOG"
    continue
  fi

  wait_for_gpu
  echo "START $(date -Is) $run_name" >> "$LOG"
  python /home/ubuntu/LoRA-QAT/train_hf.py --config "$cfg" 2>&1 | tee "$out_dir/train.log"
  status=${PIPESTATUS[0]}
  echo "END $(date -Is) $run_name status=$status" >> "$LOG"
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi

  qtype=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['quant_regularization']['quantizer_type'])" "$cfg")
  bw=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['quant_regularization']['bit_width'])" "$cfg")
  gs=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['quant_regularization']['group_size'])" "$cfg")

  python /home/ubuntu/LoRA-QAT/evaluate_quantized_proxy.py \
    --config "$cfg" \
    --adapter-dir "$out_dir/adapter" \
    --output "$out_dir/quantized_proxy_metrics__match.json" \
    --quantizer-type "$qtype" \
    --bit-width "$bw" \
    --group-size "$gs" 2>&1 | tee "$out_dir/proxy_eval.log"
  status=${PIPESTATUS[0]}
  echo "PROXY_END $(date -Is) $run_name status=$status" >> "$LOG"
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
done
SH

chmod +x "$ROOT/run_batch.sh"
pkill -f '/home/ubuntu/loraqat_outputs/logit_kl_multiprecision_llama32/run_batch.sh' || true
nohup bash "$ROOT/run_batch.sh" > /dev/null 2>&1 &
sleep 2
pgrep -af '/home/ubuntu/loraqat_outputs/logit_kl_multiprecision_llama32/run_batch.sh' || true
tail -n 10 "$ROOT/sweep.log" || true
