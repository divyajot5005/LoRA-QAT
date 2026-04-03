#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/LoRA-QAT
source /home/ubuntu/venvs/loraqat/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/home/ubuntu/loraqat_outputs/lambda_sweep_llama32_int2_ternary
mkdir -p "$ROOT/configs" "$ROOT/runs"

python - <<'PY'
import copy
import json
from pathlib import Path

base_path = Path("/home/ubuntu/LoRA-QAT/outputs/cloud_instruction_suite/configs/llama32_1b__instruction_tuning__quant_int4_ft.json")
base = json.loads(base_path.read_text(encoding="utf-8"))
base["quant_regularization"]["regularization_objective"] = "weight_mse"
base["quant_regularization"]["regularization_frequency"] = 1
base["quant_regularization"]["detach_layer_inputs"] = True
base["quant_regularization"]["max_activation_regularized_layers"] = 0
base["quant_regularization"]["log_per_layer_stats"] = False

root = Path("/home/ubuntu/loraqat_outputs/lambda_sweep_llama32_int2_ternary")
variants = {
    "int2": {
        "quantizer_type": "uniform_int2_groupwise",
        "bit_width": 2,
        "lambdas": [3e4, 1e5, 3e5, 1e6],
    },
    "ternary": {
        "quantizer_type": "ternary_groupwise",
        "bit_width": 2,
        "lambdas": [1e4, 3e4, 1e5, 3e5],
    },
}

for label, spec in variants.items():
    for lam in spec["lambdas"]:
        cfg = copy.deepcopy(base)
        lam_tag = format(lam, ".0e").replace("+", "")
        run_name = f"llama32_1b__instruction_tuning__{label}__lambda_{lam_tag}"
        cfg["training"]["output_dir"] = str(root / "runs" / run_name)
        cfg["quant_regularization"]["enable_quant_lora_regularization"] = True
        cfg["quant_regularization"]["lambda_q"] = lam
        cfg["quant_regularization"]["quantizer_type"] = spec["quantizer_type"]
        cfg["quant_regularization"]["bit_width"] = spec["bit_width"]
        cfg["quant_regularization"]["group_size"] = 128
        (root / "configs" / f"{run_name}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

print("LOWBIT_CONFIGS_READY")
PY

cat > "$ROOT/run_batch.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/LoRA-QAT
source /home/ubuntu/venvs/loraqat/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/home/ubuntu/loraqat_outputs/lambda_sweep_llama32_int2_ternary

wait_for_gpu() {
  while true; do
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
    if [ -n "$free_mem" ] && [ "$free_mem" -ge 34000 ]; then
      break
    fi
    echo "WAIT $(date -Is) free_mem_mb=${free_mem:-unknown}" >> "$ROOT/sweep.log"
    sleep 30
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
  python /home/ubuntu/LoRA-QAT/train_hf.py --config "$cfg" 2>&1 | tee "$out_dir/train.log"
  status=${PIPESTATUS[0]}
  echo "END $(date -Is) $run_name status=$status" >> "$ROOT/sweep.log"
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
  echo "PROXY_END $(date -Is) $run_name status=$status" >> "$ROOT/sweep.log"
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
done
SH

chmod +x "$ROOT/run_batch.sh"
pkill -f '/home/ubuntu/loraqat_outputs/lambda_sweep_llama32_int2_ternary/run_batch.sh' || true
nohup bash "$ROOT/run_batch.sh" > /dev/null 2>&1 &
sleep 2
pgrep -af '/home/ubuntu/loraqat_outputs/lambda_sweep_llama32_int2_ternary/run_batch.sh' || true
tail -n 10 "$ROOT/sweep.log" || true
