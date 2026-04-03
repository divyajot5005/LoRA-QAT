import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm.auto import tqdm

from quant_lora.hf_config import load_hf_experiment_config
from quant_lora.hf_data import build_dataloaders
from quant_lora.hf_model import build_model, load_tokenizer
from quant_lora.hf_trainer import _evaluate, _resolve_device, _set_seed

REPO_ROOT = Path(__file__).resolve().parent


def _resolve_hf_token() -> Optional[str]:
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        value = os.environ.get(env_name)
        if value:
            return value.strip()

    candidate_paths = [
        Path(os.environ.get("HF_HOME", "")).expanduser() / "token" if os.environ.get("HF_HOME") else None,
        Path.home() / ".cache" / "huggingface" / "token",
        Path("/root/.cache/huggingface/token"),
    ]
    for candidate in candidate_paths:
        if candidate and candidate.exists():
            value = candidate.read_text(encoding="utf-8").strip()
            if value:
                return value
    return None


@dataclass(frozen=True)
class ModelSpec:
    family: str
    model_name: str
    size_bucket: str
    rank: int
    alpha: float
    target_modules: List[str]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: str
    dataset_name: str
    dataset_config_name: Optional[str]
    format_style: str
    text_column: str
    label_column: str
    prompt_column: Optional[str]
    response_column: Optional[str]
    context_column: Optional[str]
    mask_prompt_tokens: bool
    train_split: str
    eval_split: str
    max_train_samples: Optional[int]
    max_eval_samples: Optional[int]
    max_length: int
    generation_prompts: List[str]


MODELS: List[ModelSpec] = [
    ModelSpec(
        family="gemma3_1b",
        model_name="google/gemma-3-1b-it",
        size_bucket="1b",
        rank=16,
        alpha=32.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    ModelSpec(
        family="llama32_1b",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        size_bucket="1b",
        rank=16,
        alpha=32.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    ModelSpec(
        family="qwen3_0p6b",
        model_name="Qwen/Qwen3-0.6B",
        size_bucket="1b",
        rank=16,
        alpha=32.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    ModelSpec(
        family="llama32_3b",
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        size_bucket="3b",
        rank=16,
        alpha=32.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    ModelSpec(
        family="gemma3_4b",
        model_name="google/gemma-3-4b-it",
        size_bucket="4b",
        rank=16,
        alpha=32.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    ModelSpec(
        family="qwen3_4b",
        model_name="Qwen/Qwen3-4B",
        size_bucket="4b",
        rank=16,
        alpha=32.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    ModelSpec(
        family="llama31_8b",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        size_bucket="8b",
        rank=16,
        alpha=32.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    ModelSpec(
        family="qwen3_8b",
        model_name="Qwen/Qwen3-8B",
        size_bucket="8b",
        rank=16,
        alpha=32.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    ModelSpec(
        family="gemma2_9b",
        model_name="google/gemma-2-9b-it",
        size_bucket="9b",
        rank=16,
        alpha=32.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
]


TASKS: List[TaskSpec] = [
    TaskSpec(
        name="customer_support",
        task_type="causal_lm",
        dataset_name="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        dataset_config_name=None,
        format_style="instruction_response",
        text_column="instruction",
        label_column="label",
        prompt_column="instruction",
        response_column="response",
        context_column=None,
        mask_prompt_tokens=True,
        train_split="train[:35%]",
        eval_split="train[90%:95%]",
        max_train_samples=10000,
        max_eval_samples=1500,
        max_length=384,
        generation_prompts=[
            "Instruction:\nquestion about cancelling order 81342\n\nResponse:\n",
            "Instruction:\nneed help tracking delayed package 55201\n\nResponse:\n",
            "Instruction:\nrequesting refund because product arrived defective\n\nResponse:\n",
        ],
    ),
    TaskSpec(
        name="instruction_tuning",
        task_type="causal_lm",
        dataset_name="databricks/databricks-dolly-15k",
        dataset_config_name=None,
        format_style="instruction_response",
        text_column="instruction",
        label_column="label",
        prompt_column="instruction",
        response_column="response",
        context_column="context",
        mask_prompt_tokens=True,
        train_split="train[:70%]",
        eval_split="train[90%:95%]",
        max_train_samples=9000,
        max_eval_samples=1000,
        max_length=384,
        generation_prompts=[
            "Instruction:\nDraft a concise project-status update for a manager after a one-week delay.\n\nResponse:\n",
            "Instruction:\nExplain the difference between precision and recall in machine learning.\n\nResponse:\n",
            "Instruction:\nWrite a short, friendly email asking to reschedule a meeting.\n\nResponse:\n",
        ],
    ),
    TaskSpec(
        name="domain_adaptation_code",
        task_type="causal_lm",
        dataset_name="code_search_net",
        dataset_config_name="python",
        format_style="plain_text",
        text_column="func_code_string",
        label_column="label",
        prompt_column=None,
        response_column=None,
        context_column=None,
        mask_prompt_tokens=False,
        train_split="train",
        eval_split="validation",
        max_train_samples=20000,
        max_eval_samples=2000,
        max_length=512,
        generation_prompts=[
            "def moving_average(values, window):\n",
            "class CustomerRecord:\n    def __init__(self, name, email):\n",
            "def format_currency(amount):\n",
        ],
    ),
]


PHASES = [
    "before_ft",
    "no_quant_ft",
    "quant_int4_ft",
    "quant_fp8_ft",
    "quant_int2_ft",
    "quant_ternary_ft",
    "quant_int4_act_ft",
    "quant_fp8_act_ft",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the H100 Quant-LoRA experiment suite.")
    parser.add_argument("--output-root", default="outputs/h100_suite", help="Root directory for configs and results.")
    parser.add_argument("--cache-root", help="Optional root directory for Hugging Face model and dataset caches.")
    parser.add_argument("--models", nargs="*", help="Optional subset of model families to run.")
    parser.add_argument("--tasks", nargs="*", help="Optional subset of task names to run.")
    parser.add_argument("--phases", nargs="*", default=PHASES, help="Subset of phases to run.")
    parser.add_argument("--write-only", action="store_true", help="Only write configs/manifests, do not run experiments.")
    parser.add_argument("--export-fp8-offline", action="store_true", help="Try to export merged FP8 checkpoints with AutoFP8.")
    return parser.parse_args()


def _configure_hf_cache(cache_root: Optional[str]) -> None:
    if not cache_root:
        cache_root_path = None
    else:
        cache_root_path = Path(cache_root).expanduser().resolve()
        hub_cache = cache_root_path / "hub"
        datasets_cache = cache_root_path / "datasets"
        transformers_cache = cache_root_path / "transformers"
        for path in (hub_cache, datasets_cache, transformers_cache):
            path.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_root_path)
        os.environ["HF_HUB_CACHE"] = str(hub_cache)
        os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
        os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)

    token = _resolve_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        os.environ["HF_HUB_TOKEN"] = token
        if cache_root_path is not None:
            token_path = cache_root_path / "token"
            token_path.write_text(token, encoding="utf-8")


def _size_defaults(model_family: str, size_bucket: str) -> Dict[str, object]:
    if model_family == "llama31_8b":
        return {"num_steps": 300, "batch_size": 1, "eval_batch_size": 1, "grad_accum": 16, "lr": 1e-4}
    defaults = {
        "1b": {"num_steps": 400, "batch_size": 1, "eval_batch_size": 1, "grad_accum": 16, "lr": 2e-4},
        "3b": {"num_steps": 350, "batch_size": 1, "eval_batch_size": 1, "grad_accum": 16, "lr": 1.5e-4},
        "4b": {"num_steps": 320, "batch_size": 1, "eval_batch_size": 1, "grad_accum": 16, "lr": 1.25e-4},
        "8b": {"num_steps": 280, "batch_size": 1, "eval_batch_size": 1, "grad_accum": 16, "lr": 1e-4},
        "9b": {"num_steps": 250, "batch_size": 1, "eval_batch_size": 1, "grad_accum": 16, "lr": 8e-5},
    }
    return defaults[size_bucket]


def _quant_payload(size_bucket: str, phase: str) -> Dict[str, object]:
    activation_layers_by_size = {"1b": 8, "3b": 8, "4b": 6, "8b": 4, "9b": 4}
    int4_reg_frequency = {"1b": 1, "3b": 8, "4b": 8, "8b": 8, "9b": 8}
    fp8_reg_frequency = {"1b": 1, "3b": 4, "4b": 4, "8b": 4, "9b": 4}
    int2_reg_frequency = {"1b": 1, "3b": 8, "4b": 8, "8b": 8, "9b": 8}
    ternary_reg_frequency = {"1b": 1, "3b": 8, "4b": 8, "8b": 8, "9b": 8}
    if phase == "no_quant_ft":
        return {
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
        }

    if phase == "quant_int4_ft":
        lambda_by_size = {"1b": 0.01, "3b": 0.008, "4b": 0.006, "8b": 0.004, "9b": 0.003}
        return {
            "enable_quant_lora_regularization": True,
            "lambda_q": lambda_by_size[size_bucket],
            "quantizer_type": "uniform_groupwise",
            "bit_width": 4,
            "group_size": 128,
            "regularization_objective": "weight_mse",
            "regularization_frequency": int4_reg_frequency[size_bucket],
            "detach_layer_inputs": True,
            "max_activation_regularized_layers": 0,
            "log_per_layer_stats": False,
        }

    if phase == "quant_fp8_ft":
        lambda_by_size = {"1b": 0.003, "3b": 0.0025, "4b": 0.002, "8b": 0.0015, "9b": 0.001}
        return {
            "enable_quant_lora_regularization": True,
            "lambda_q": lambda_by_size[size_bucket],
            "quantizer_type": "fp8_e4m3fn",
            "bit_width": 8,
            "group_size": 128,
            "regularization_objective": "weight_mse",
            "regularization_frequency": fp8_reg_frequency[size_bucket],
            "detach_layer_inputs": True,
            "max_activation_regularized_layers": 0,
            "log_per_layer_stats": False,
        }

    if phase == "quant_int2_ft":
        lambda_by_size = {"1b": 3e5, "3b": 1e5, "4b": 1e5, "8b": 3e4, "9b": 3e4}
        return {
            "enable_quant_lora_regularization": True,
            "lambda_q": lambda_by_size[size_bucket],
            "quantizer_type": "uniform_int2_groupwise",
            "bit_width": 2,
            "group_size": 128,
            "regularization_objective": "weight_mse",
            "regularization_frequency": int2_reg_frequency[size_bucket],
            "detach_layer_inputs": True,
            "max_activation_regularized_layers": 0,
            "log_per_layer_stats": False,
        }

    if phase == "quant_ternary_ft":
        lambda_by_size = {"1b": 1e5, "3b": 3e4, "4b": 3e4, "8b": 1e4, "9b": 1e4}
        return {
            "enable_quant_lora_regularization": True,
            "lambda_q": lambda_by_size[size_bucket],
            "quantizer_type": "ternary_groupwise",
            "bit_width": 2,
            "group_size": 128,
            "regularization_objective": "weight_mse",
            "regularization_frequency": ternary_reg_frequency[size_bucket],
            "detach_layer_inputs": True,
            "max_activation_regularized_layers": 0,
            "log_per_layer_stats": False,
        }

    if phase == "quant_int4_act_ft":
        lambda_by_size = {"1b": 0.01, "3b": 0.008, "4b": 0.006, "8b": 0.004, "9b": 0.003}
        return {
            "enable_quant_lora_regularization": True,
            "lambda_q": lambda_by_size[size_bucket],
            "quantizer_type": "uniform_groupwise",
            "bit_width": 4,
            "group_size": 128,
            "regularization_objective": "activation_mse",
            "regularization_frequency": int4_reg_frequency[size_bucket],
            "detach_layer_inputs": True,
            "max_activation_regularized_layers": activation_layers_by_size[size_bucket],
            "log_per_layer_stats": False,
        }

    if phase == "quant_fp8_act_ft":
        lambda_by_size = {"1b": 0.003, "3b": 0.0025, "4b": 0.002, "8b": 0.0015, "9b": 0.001}
        return {
            "enable_quant_lora_regularization": True,
            "lambda_q": lambda_by_size[size_bucket],
            "quantizer_type": "fp8_e4m3fn",
            "bit_width": 8,
            "group_size": 128,
            "regularization_objective": "activation_mse",
            "regularization_frequency": fp8_reg_frequency[size_bucket],
            "detach_layer_inputs": True,
            "max_activation_regularized_layers": activation_layers_by_size[size_bucket],
            "log_per_layer_stats": False,
        }

    raise ValueError(f"Unsupported phase for quant payload: {phase}")


def _build_config(model: ModelSpec, task: TaskSpec, phase: str, output_root: Path) -> Dict[str, object]:
    defaults = _size_defaults(model.family, model.size_bucket)
    experiment_name = f"{model.family}__{task.name}__{phase}"
    output_dir = output_root / "runs" / experiment_name

    return {
        "seed": 7,
        "model": {
            "task_type": task.task_type,
            "model_name": model.model_name,
            "max_length": task.max_length,
            "gradient_checkpointing": True,
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa",
        },
        "dataset": {
            "dataset_name": task.dataset_name,
            "dataset_config_name": task.dataset_config_name,
            "train_split": task.train_split,
            "eval_split": task.eval_split,
            "max_train_samples": task.max_train_samples,
            "max_eval_samples": task.max_eval_samples,
            "format_style": task.format_style,
            "text_column": task.text_column,
            "label_column": task.label_column,
            "prompt_column": task.prompt_column,
            "response_column": task.response_column,
            "context_column": task.context_column,
            "mask_prompt_tokens": task.mask_prompt_tokens,
        },
        "lora": {
            "enabled": True,
            "target_modules": model.target_modules,
            "rank": model.rank,
            "alpha": model.alpha,
            "dropout": 0.05,
        },
        "training": {
            "output_dir": str(output_dir),
            "device": "auto",
            "mixed_precision": "bf16",
            "num_steps": defaults["num_steps"],
            "batch_size": defaults["batch_size"],
            "eval_batch_size": defaults["eval_batch_size"],
            "gradient_accumulation_steps": defaults["grad_accum"],
            "learning_rate": defaults["lr"],
            "weight_decay": 0.0,
            "warmup_steps": max(10, defaults["num_steps"] // 20),
            "log_interval": max(10, defaults["num_steps"] // 10),
            "eval_interval": max(25, defaults["num_steps"] // 5),
            "max_eval_batches": None,
            "generation_prompts": task.generation_prompts,
            "generation_max_new_tokens": 96,
            "generation_temperature": 0.7,
            "generation_top_p": 0.9,
        },
        "quant_regularization": _quant_payload(model.size_bucket, phase),
    }


def _write_config(config: Dict[str, object], config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    temp_path.replace(config_path)


def _run_train(config_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "train_hf.py"), "--config", str(config_path)],
        check=True,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
    )


def _run_before_ft_eval(config_path: Path, output_path: Path) -> None:
    config = load_hf_experiment_config(str(config_path))
    config.lora.enabled = False

    _set_seed(config.seed)
    device = _resolve_device(config.training.device)
    tokenizer = load_tokenizer(config)
    data_bundle = build_dataloaders(config, tokenizer)
    model = build_model(config, data_bundle.num_labels)
    if config.model.task_type == "causal_lm" and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    metrics = _evaluate(model, data_bundle.eval_loader, device, config)

    payload = {
        "phase": "before_ft",
        "model_name": config.model.model_name,
        "task_type": config.model.task_type,
        "dataset_name": config.dataset.dataset_name,
        "dataset_config_name": config.dataset.dataset_config_name,
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    del model, data_bundle, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _slugify(value: str) -> str:
    return value.replace("/", "__").replace(":", "_")


def _load_metrics(metrics_path: Path) -> Dict[str, object]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_summary_rows(output_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for metrics_path in output_root.glob("runs/*/final_metrics.json"):
        data = _load_metrics(metrics_path)
        run_dir = metrics_path.parent
        phase = run_dir.name.split("__")[-1]
        row = {
            "run_name": run_dir.name,
            "model_name": data["model_name"],
            "dataset_name": data["dataset_name"],
            "phase": phase,
            "quantizer_type": data["quantizer_type"],
            "lambda_q": data["lambda_q"],
            "peak_gpu_mem_mb": round(float(data["peak_gpu_mem_mb"]), 2),
        }
        row.update(data["final_metrics"])
        rows.append(row)

    for metrics_path in output_root.glob("runs/*/before_ft_metrics.json"):
        data = _load_metrics(metrics_path)
        run_dir = metrics_path.parent
        row = {
            "run_name": run_dir.name,
            "model_name": data["model_name"],
            "dataset_name": data["dataset_name"],
            "phase": "before_ft",
            "quantizer_type": "none",
            "lambda_q": 0.0,
            "peak_gpu_mem_mb": "",
        }
        row.update(data["metrics"])
        rows.append(row)
    return sorted(rows, key=lambda item: item["run_name"])


def _write_summary(rows: List[Dict[str, object]], output_root: Path) -> None:
    summary_csv = output_root / "summary.csv"
    summary_md = output_root / "summary.md"
    fieldnames = sorted({key for row in rows for key in row.keys()})

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# H100 Quant-LoRA Suite Summary", "", "| Run | Phase | Metric 1 | Metric 2 | Quantizer | Lambda | Peak GPU MB |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        metric_parts = []
        if "eval_accuracy" in row:
            metric_parts.append(f"accuracy={row['eval_accuracy']:.4f}")
        if "eval_perplexity" in row:
            metric_parts.append(f"ppl={row['eval_perplexity']:.4f}")
        metric_one = metric_parts[0] if metric_parts else ""
        metric_two = f"loss={row['eval_loss']:.4f}" if "eval_loss" in row else ""
        lines.append(
            f"| {row['run_name']} | {row['phase']} | {metric_one} | {metric_two} | {row['quantizer_type']} | {row['lambda_q']} | {row['peak_gpu_mem_mb']} |"
        )
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def _write_serving_recipes(selected_runs: List[Dict[str, str]], output_root: Path) -> None:
    serve_md = output_root / "serving_recipes.md"
    int4_sh = output_root / "serve_int4.sh"
    fp8_sh = output_root / "serve_fp8.sh"

    md_lines = [
        "# Serving Recipes",
        "",
        "INT4 serving uses vLLM BitsAndBytes in-flight quantization.",
        "FP8 serving uses vLLM online dynamic FP8 quantization by default; for lower load-time memory, optionally export offline FP8 with AutoFP8.",
        "",
    ]
    int4_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    fp8_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]

    for run in selected_runs:
        base_model = run["model_name"]
        adapter_dir = run["adapter_dir"]
        serve_name = run["serve_name"]
        rank = run["rank"]

        int4_cmd = (
            f'vllm serve "{base_model}" --dtype bfloat16 --quantization bitsandbytes '
            f'--load-format bitsandbytes --enable-lora --max-lora-rank {rank} '
            f'--lora-modules {serve_name}="{adapter_dir}"'
        )
        fp8_cmd = (
            f'vllm serve "{base_model}" --dtype bfloat16 --quantization fp8 '
            f'--enable-lora --max-lora-rank {rank} '
            f'--lora-modules {serve_name}="{adapter_dir}"'
        )

        md_lines.append(f"## {serve_name}")
        md_lines.append("")
        md_lines.append(f"- Base model: `{base_model}`")
        md_lines.append(f"- Adapter: `{adapter_dir}`")
        md_lines.append(f"- INT4 command: `{int4_cmd}`")
        md_lines.append(f"- FP8 command: `{fp8_cmd}`")
        md_lines.append("")

        int4_lines.append(int4_cmd)
        fp8_lines.append(fp8_cmd)

    serve_md.write_text("\n".join(md_lines), encoding="utf-8")
    int4_sh.write_text("\n".join(int4_lines) + "\n", encoding="utf-8")
    fp8_sh.write_text("\n".join(fp8_lines) + "\n", encoding="utf-8")


def _maybe_export_fp8_offline(selected_runs: List[Dict[str, str]], output_root: Path) -> None:
    try:
        from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        message = "AutoFP8 is not installed. Skipping offline FP8 export."
        (output_root / "fp8_offline_export.log").write_text(message + "\n", encoding="utf-8")
        return

    export_root = output_root / "fp8_offline_exports"
    export_root.mkdir(parents=True, exist_ok=True)
    quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="dynamic")
    log_lines = []

    for run in selected_runs:
        base_model = run["model_name"]
        adapter_dir = Path(run["adapter_dir"])
        export_dir = export_root / run["serve_name"]
        merged_dir = export_dir / "merged_bf16"
        fp8_dir = export_dir / "fp8_dynamic"
        if fp8_dir.exists():
            log_lines.append(f"Skipping existing export: {fp8_dir}")
            continue

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

        fp8_model = AutoFP8ForCausalLM.from_pretrained(str(merged_dir), quantize_config)
        fp8_model.quantize([])
        fp8_model.save_quantized(str(fp8_dir))
        log_lines.append(f"Exported offline FP8 checkpoint: {fp8_dir}")

        del fp8_model, merged, base, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (output_root / "fp8_offline_export.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _configure_hf_cache(args.cache_root)
    output_root = Path(args.output_root)
    configs_root = output_root / "configs"
    output_root.mkdir(parents=True, exist_ok=True)

    selected_models = [model for model in MODELS if not args.models or model.family in set(args.models)]
    selected_tasks = [task for task in TASKS if not args.tasks or task.name in set(args.tasks)]
    selected_phases = args.phases

    selected_runs_for_serving: List[Dict[str, str]] = []
    failed_runs_path = output_root / "failed_runs.jsonl"
    work_items = [(model, task, phase) for model in selected_models for task in selected_tasks for phase in selected_phases]

    suite_bar = tqdm(work_items, desc="suite", dynamic_ncols=True)
    for model, task, phase in suite_bar:
        run_name = f"{model.family}__{task.name}__{phase}"
        suite_bar.set_postfix(run=run_name)
        if phase == "before_ft":
            phase_for_config = "no_quant_ft"
        else:
            phase_for_config = phase

        config = _build_config(model, task, phase_for_config, output_root)
        config_path = configs_root / f"{run_name}.json"
        config["training"]["output_dir"] = str(output_root / "runs" / run_name)
        _write_config(config, config_path)

        run_dir = Path(config["training"]["output_dir"])
        metrics_path = run_dir / ("before_ft_metrics.json" if phase == "before_ft" else "final_metrics.json")
        if metrics_path.exists():
            continue
        if args.write_only:
            continue

        try:
            if phase == "before_ft":
                _run_before_ft_eval(config_path, metrics_path)
            else:
                _run_train(config_path)
                if phase in {"quant_int4_ft", "quant_fp8_ft"}:
                    selected_runs_for_serving.append(
                        {
                            "model_name": model.model_name,
                            "adapter_dir": str(run_dir / "adapter"),
                            "serve_name": run_name,
                            "rank": str(model.rank),
                        }
                    )
        except Exception as exc:
            # Keep the queue moving: log this run and continue to the next work item.
            failure_payload = {
                "run_name": run_name,
                "model_family": model.family,
                "model_name": model.model_name,
                "task": task.name,
                "phase": phase,
                "config_path": str(config_path),
                "output_dir": str(run_dir),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            with failed_runs_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure_payload) + "\n")
            continue
    suite_bar.close()

    rows = _collect_summary_rows(output_root)
    _write_summary(rows, output_root)
    _write_serving_recipes(selected_runs_for_serving, output_root)
    if args.export_fp8_offline and not args.write_only:
        _maybe_export_fp8_offline(selected_runs_for_serving, output_root)

    print(f"configs_root={configs_root}")
    print(f"summary_csv={output_root / 'summary.csv'}")
    print(f"summary_md={output_root / 'summary.md'}")
    print(f"serving_recipes={output_root / 'serving_recipes.md'}")


if __name__ == "__main__":
    main()
