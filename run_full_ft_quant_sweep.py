import argparse
import csv
import gc
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, TypeVar

import torch


REPO_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = REPO_ROOT / "train_hf.py"
EVAL_SCRIPT = REPO_ROOT / "evaluate_quantized_proxy.py"


@dataclass(frozen=True)
class ModelSpec:
    family: str
    model_name: str
    size_bucket: str


@dataclass(frozen=True)
class QuantTarget:
    name: str
    quantizer_type: str
    bit_width: int
    group_size: int
    lambdas: List[float]


MODELS: List[ModelSpec] = [
    ModelSpec("llama_1b", "meta-llama/Llama-3.2-1B-Instruct", "1b"),
    ModelSpec("qwen3_0p6b", "Qwen/Qwen3-0.6B", "1b"),
    ModelSpec("gemma_1b", "google/gemma-3-1b-it", "1b"),
]


QUANT_TARGETS: List[QuantTarget] = [
    QuantTarget("int8", "uniform_groupwise", 8, 128, [1e7, 3e7, 1e8, 3e8]),
    QuantTarget("int4", "uniform_groupwise", 4, 128, [1e5, 3e5, 1e6, 3e6]),
    QuantTarget("int2", "uniform_int2_groupwise", 2, 128, [3e4, 1e5, 3e5, 1e6]),
    QuantTarget("ternary", "ternary_groupwise", 2, 128, [1e4, 3e4, 1e5, 3e5]),
]

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run barebones full-FT quantization-regularized sweeps.")
    parser.add_argument("--output-root", default="outputs/full_ft_quant_sweep", help="Directory for configs and outputs.")
    parser.add_argument("--models", nargs="*", help="Optional subset of model families.")
    parser.add_argument("--quantizers", nargs="*", help="Optional subset of quantizer names.")
    parser.add_argument("--write-only", action="store_true", help="Only write configs, do not execute.")
    parser.add_argument("--baseline-only", action="store_true", help="Only run the no-regularizer baseline and its proxy evals.")
    parser.add_argument("--device", default="cuda", help="Training/eval device.")
    parser.add_argument("--proxy-device", help="Optional proxy-eval device override.")
    parser.add_argument("--skip-proxy-quant-stats", action="store_true", help="Skip raw quant-stat computation during proxy eval to reduce memory use.")
    parser.add_argument("--cache-root", help="Optional Hugging Face cache root.")
    parser.add_argument("--max-train-samples", type=int, help="Optional override for training samples.")
    parser.add_argument("--max-eval-samples", type=int, help="Optional override for eval samples.")
    return parser.parse_args()


def _configure_hf_cache(cache_root: str | None) -> None:
    if not cache_root:
        return
    root = Path(cache_root).expanduser().resolve()
    hub_cache = root / "hub"
    datasets_cache = root / "datasets"
    transformers_cache = root / "transformers"
    hub_cache.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)
    import os

    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)


def _size_defaults(size_bucket: str) -> Dict[str, object]:
    defaults = {
        "1b": {"num_steps": 200, "batch_size": 1, "eval_batch_size": 1, "grad_accum": 16, "lr": 1e-5},
        "3b": {"num_steps": 150, "batch_size": 1, "eval_batch_size": 1, "grad_accum": 16, "lr": 8e-6},
        "8b": {"num_steps": 100, "batch_size": 1, "eval_batch_size": 1, "grad_accum": 16, "lr": 5e-6},
    }
    return defaults[size_bucket]


def _base_config(model_spec: ModelSpec, output_dir: Path, args: argparse.Namespace) -> Dict[str, object]:
    defaults = _size_defaults(model_spec.size_bucket)
    return {
        "seed": 7,
        "model": {
            "task_type": "causal_lm",
            "model_name": model_spec.model_name,
            "max_length": 384,
            "gradient_checkpointing": True,
            "torch_dtype": "bfloat16",
        },
        "dataset": {
            "dataset_name": "databricks/databricks-dolly-15k",
            "dataset_config_name": None,
            "train_split": "train[:70%]",
            "eval_split": "train[90%:95%]",
            "max_train_samples": args.max_train_samples or 4000,
            "max_eval_samples": args.max_eval_samples or 500,
            "format_style": "instruction_response",
            "text_column": "instruction",
            "label_column": "label",
            "prompt_column": "instruction",
            "response_column": "response",
            "context_column": "context",
            "mask_prompt_tokens": True,
            "num_proc": 1,
        },
        "lora": {
            "enabled": False,
            "target_modules": [],
            "rank": 0,
            "alpha": 0.0,
            "dropout": 0.0,
        },
        "training": {
            "output_dir": str(output_dir),
            "device": args.device,
            "mixed_precision": "bf16",
            "optimizer_type": "adafactor",
            "num_steps": defaults["num_steps"],
            "batch_size": defaults["batch_size"],
            "eval_batch_size": defaults["eval_batch_size"],
            "gradient_accumulation_steps": defaults["grad_accum"],
            "learning_rate": defaults["lr"],
            "weight_decay": 0.0,
            "warmup_steps": 10,
            "log_interval": 10,
            "eval_interval": 50,
            "max_eval_batches": 20,
            "generation_prompts": [],
        },
        "quant_regularization": {
            "enable_quant_lora_regularization": False,
            "lambda_q": 0.0,
            "quantizer_type": "uniform_groupwise",
            "bit_width": 4,
            "group_size": 128,
            "regularization_objective": "weight_mse",
            "regularization_frequency": 1,
            "detach_layer_inputs": True,
            "max_activation_regularized_layers": 0,
            "log_per_layer_stats": False,
        },
    }


def _run(command: List[str], cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_lambda(lambda_q: float) -> str:
    return f"{lambda_q:.0e}".replace("+0", "").replace("+", "")


def _selected_specs(all_items: Iterable[T], names: List[str] | None, attr: str) -> List[T]:
    items = list(all_items)
    if not names:
        return items
    wanted = set(names)
    return [item for item in items if getattr(item, attr) in wanted]


def _artifact_dir(output_dir: Path) -> Path:
    model_dir = output_dir / "model"
    if model_dir.exists():
        return model_dir
    return output_dir / "adapter"


def _proxy_eval(
    config_path: Path,
    artifact_dir: Path,
    output_path: Path,
    target: QuantTarget,
    args: argparse.Namespace,
) -> None:
    if output_path.exists():
        return
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--config",
        str(config_path),
        "--artifact-dir",
        str(artifact_dir),
        "--output",
        str(output_path),
        "--device",
        args.proxy_device or args.device,
        "--quantizer-type",
        target.quantizer_type,
        "--bit-width",
        str(target.bit_width),
        "--group-size",
        str(target.group_size),
    ]
    if args.skip_proxy_quant_stats:
        command.append("--skip-quant-stats")
    _run(command, REPO_ROOT)


def _collect_row(
    model_spec: ModelSpec,
    run_name: str,
    target_name: str,
    lambda_q: float,
    final_metrics_path: Path,
    proxy_metrics_path: Path,
) -> Dict[str, object]:
    final_payload = _load_json(final_metrics_path)
    proxy_payload = _load_json(proxy_metrics_path)
    return {
        "model_family": model_spec.family,
        "model_name": model_spec.model_name,
        "size_bucket": model_spec.size_bucket,
        "run_name": run_name,
        "target_quantizer": target_name,
        "lambda_q": lambda_q,
        "float_loss": final_payload["final_metrics"]["eval_loss"],
        "float_perplexity": final_payload["final_metrics"].get("eval_perplexity"),
        "quant_loss": proxy_payload["proxy_quantized_metrics"]["eval_loss"],
        "quant_perplexity": proxy_payload["proxy_quantized_metrics"].get("eval_perplexity"),
        "quant_gap": proxy_payload["proxy_quantized_metrics"]["eval_loss"] - final_payload["final_metrics"]["eval_loss"],
        "raw_quant_error": proxy_payload["raw_quant_error"],
        "avg_distance_to_grid": proxy_payload["avg_distance_to_grid"],
        "num_proxy_quantized_layers": proxy_payload["num_proxy_quantized_layers"],
    }


def main() -> None:
    args = parse_args()
    _configure_hf_cache(args.cache_root)

    output_root = Path(args.output_root).expanduser().resolve()
    config_root = output_root / "configs"
    run_root = output_root / "runs"
    config_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    selected_models = _selected_specs(MODELS, args.models, "family")
    selected_targets = _selected_specs(QUANT_TARGETS, args.quantizers, "name")
    rows: List[Dict[str, object]] = []

    for model_spec in selected_models:
        baseline_run_name = f"{model_spec.family}__fullft__no_quant"
        baseline_output_dir = run_root / baseline_run_name
        baseline_config = _base_config(model_spec, baseline_output_dir, args)
        baseline_config_path = config_root / f"{baseline_run_name}.json"
        _write_json(baseline_config_path, baseline_config)

        if not args.write_only and not (baseline_output_dir / "final_metrics.json").exists():
            _run([sys.executable, str(TRAIN_SCRIPT), "--config", str(baseline_config_path)], REPO_ROOT)

        if not args.write_only and (baseline_output_dir / "final_metrics.json").exists():
            baseline_artifact_dir = _artifact_dir(baseline_output_dir)
            for target in selected_targets:
                proxy_path = baseline_output_dir / f"proxy_{target.name}.json"
                _proxy_eval(baseline_config_path, baseline_artifact_dir, proxy_path, target, args)
                rows.append(
                    _collect_row(
                        model_spec,
                        baseline_run_name,
                        target.name,
                        0.0,
                        baseline_output_dir / "final_metrics.json",
                        proxy_path,
                    )
                )

        if args.baseline_only:
            continue

        for target in selected_targets:
            for lambda_q in target.lambdas:
                run_name = f"{model_spec.family}__fullft__{target.name}__lambda_{_format_lambda(lambda_q)}"
                run_output_dir = run_root / run_name
                run_config = _base_config(model_spec, run_output_dir, args)
                run_config["quant_regularization"] = {
                    "enable_quant_lora_regularization": True,
                    "lambda_q": lambda_q,
                    "quantizer_type": target.quantizer_type,
                    "bit_width": target.bit_width,
                    "group_size": target.group_size,
                    "regularization_objective": "weight_mse",
                    "regularization_frequency": 1,
                    "detach_layer_inputs": True,
                    "max_activation_regularized_layers": 0,
                    "log_per_layer_stats": False,
                }
                run_config_path = config_root / f"{run_name}.json"
                _write_json(run_config_path, run_config)

                if not args.write_only and not (run_output_dir / "final_metrics.json").exists():
                    _run([sys.executable, str(TRAIN_SCRIPT), "--config", str(run_config_path)], REPO_ROOT)

                if not args.write_only and (run_output_dir / "final_metrics.json").exists():
                    proxy_path = run_output_dir / f"proxy_{target.name}.json"
                    _proxy_eval(run_config_path, _artifact_dir(run_output_dir), proxy_path, target, args)
                    rows.append(
                        _collect_row(
                            model_spec,
                            run_name,
                            target.name,
                            lambda_q,
                            run_output_dir / "final_metrics.json",
                            proxy_path,
                        )
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    summary_csv = output_root / "summary.csv"
    if rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (output_root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"saved_summary={summary_csv}")
    else:
        print("No completed runs to summarize yet.")


if __name__ == "__main__":
    main()
