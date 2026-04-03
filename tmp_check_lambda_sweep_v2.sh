set -euo pipefail
ROOT=/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2
echo "PID_STATUS"
ps -fp 3476943 || true
echo "RECENT_LOG"
tail -n 60 "$ROOT/sweep.log" || true
echo "COMPLETED"
python - <<'PY'
import json, os, glob
root = '/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2/runs'
for path in sorted(glob.glob(os.path.join(root, '*', 'final_metrics.json'))):
    run = os.path.basename(os.path.dirname(path))
    with open(path) as f:
        data = json.load(f)
    fm = data.get('final_metrics', data)
    print(run, fm.get('eval_loss'), fm.get('eval_perplexity'))
PY