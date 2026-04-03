import os
import subprocess
from pathlib import Path


HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
CLOUDEXE_APIKEY = os.environ.get("CLOUDEXE_APIKEY", "").strip()


def main() -> int:
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN is required")
    if not CLOUDEXE_APIKEY:
        raise SystemExit("CLOUDEXE_APIKEY is required")

    cache_root = Path("/root/.cache/huggingface_h100_big")
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "token").write_text(HF_TOKEN, encoding="utf-8")

    env = os.environ.copy()
    env["CLOUDEXE_APIKEY"] = CLOUDEXE_APIKEY
    env["HF_HOME"] = str(cache_root)
    env["HF_TOKEN"] = HF_TOKEN
    env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    env["HF_HUB_TOKEN"] = HF_TOKEN

    cmd = [
        "/usr/bin/cloudexe",
        "--gpuspec",
        "H100x1",
        "--queue",
        "--",
        "/opt/miniconda/envs/ndna/bin/python",
        "/root/LoRA-QAT/run_h100_quant_lora_suite.py",
        "--output-root",
        "/root/outputs/lora_lowbit_h100_big",
        "--cache-root",
        str(cache_root),
        "--models",
        "llama32_3b",
        "qwen3_4b",
        "gemma3_4b",
        "--tasks",
        "domain_adaptation_code",
        "--phases",
        "quant_int2_ft",
        "quant_ternary_ft",
    ]
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
