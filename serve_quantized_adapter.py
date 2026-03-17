import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a Quant-LoRA adapter with quantized base weights via vLLM.")
    parser.add_argument("--base-model", required=True, help="HF model name or local path for the base model.")
    parser.add_argument("--adapter-dir", required=True, help="Path to the LoRA adapter directory.")
    parser.add_argument(
        "--quantization",
        required=True,
        choices=["int4", "fp8"],
        help="Quantized serving mode to use for the base model.",
    )
    parser.add_argument("--served-model-name", default="quant-lora-model", help="Logical model name exposed by vLLM.")
    parser.add_argument("--dtype", default="bfloat16", help="Activation dtype for serving.")
    parser.add_argument("--max-lora-rank", type=int, default=16, help="Maximum LoRA rank accepted by the server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host address for the vLLM server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the vLLM server.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92, help="Target GPU memory utilization.")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel degree. Keep 1 for a single H100 unless you are sharding across multiple GPUs.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum sequence length to reserve KV cache for.",
    )
    parser.add_argument(
        "--offline-fp8-model",
        help="Optional local path to a pre-exported offline FP8 checkpoint. If set, it overrides --base-model for FP8 serving.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved vLLM command without launching the server.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    adapter_dir = str(Path(args.adapter_dir).resolve())
    base_model = args.base_model

    if args.quantization == "fp8" and args.offline_fp8_model:
        base_model = str(Path(args.offline_fp8_model).resolve())

    command = [
        "vllm",
        "serve",
        base_model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--enable-lora",
        "--max-lora-rank",
        str(args.max_lora_rank),
        "--lora-modules",
        f"{args.served_model_name}={adapter_dir}",
    ]

    if args.quantization == "int4":
        command.extend(["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"])
    else:
        command.extend(["--quantization", "fp8"])

    return command


def main() -> None:
    args = parse_args()
    command = build_command(args)
    print("Resolved command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))

    if args.dry_run:
        return

    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
