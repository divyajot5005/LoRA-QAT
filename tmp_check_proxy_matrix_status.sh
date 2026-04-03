set -euo pipefail
ROOT=/mnt/loraqat_outputs/lambda_sweep_llama32_int4_int8_v2
echo "PID_STATUS"
ps -fp 3490261 || true
echo "LOG"
tail -n 60 "$ROOT/proxy_matrix.log" || true
echo "SUMMARY"
if [ -f "$ROOT/proxy_matrix/summary.csv" ]; then
  tail -n 40 "$ROOT/proxy_matrix/summary.csv"
fi