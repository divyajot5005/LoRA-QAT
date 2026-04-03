set -euo pipefail
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits || true
ps -eo pid,cmd --sort=-rss | head -n 20