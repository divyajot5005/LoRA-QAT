import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: str
    model_name: str
    max_length: int
    gradient_checkpointing: bool
    target_modules: list[str]
    dataset_payload: dict[str, object]
    training_payload: dict[str, object]


TASKS = [
    TaskSpec(
        name="distilbert_agnews",
        task_type="sequence_classification",
        model_name="distilbert-base-uncased",
        max_length=128,
        gradient_checkpointing=False,
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
        dataset_payload={
            "dataset_name": "ag_news",
            "dataset_config_name": None,
            "train_split": "train[:5%]",
            "eval_split": "test[:10%]",
            "max_train_samples": None,
            "max_eval_samples": None,
            "format_style": "plain_text",
            "text_column": "text",
            "label_column": "label",
            "prompt_column": None,
            "response_column": None,
            "context_column": None,
            "mask_prompt_tokens": False,
        },
        training_payload={
            "num_steps": 120,
            "batch_size": 16,
            "eval_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-4,
            "warmup_steps": 12,
            "log_interval": 20,
            "eval_interval": 40,
            "max_eval_batches": None,
            "generation_prompts": [],
        },
    ),
    TaskSpec(
        name="gpt2_customer_support",
        task_type="causal_lm",
        model_name="gpt2",
        max_length=160,
        gradient_checkpointing=True,
        target_modules=["c_attn", "c_proj", "c_fc"],
        dataset_payload={
            "dataset_name": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            "dataset_config_name": None,
            "train_split": "train[:20%]",
            "eval_split": "train[90%:93%]",
            "max_train_samples": None,
            "max_eval_samples": None,
            "format_style": "instruction_response",
            "text_column": "instruction",
            "label_column": "label",
            "prompt_column": "instruction",
            "response_column": "response",
            "context_column": None,
            "mask_prompt_tokens": True,
        },
        training_payload={
            "num_steps": 90,
            "batch_size": 2,
            "eval_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "warmup_steps": 10,
            "log_interval": 15,
            "eval_interval": 30,
            "max_eval_batches": None,
            "generation_prompts": [
                "Instruction:\nquestion about cancelling order 81342\n\nResponse:\n",
                "Instruction:\nneed help tracking delayed package 55201\n\nResponse:\n",
            ],
            "generation_max_new_tokens": 80,
            "generation_temperature": 0.7,
            "generation_top_p": 0.9,
        },
    ),
]


PHASES = {
    "baseline": {
        "train_quant": {
            "enable_quant_lora_regularization": False,
            "lambda_q": 0.0,
            "quantizer_type": "uniform_groupwise",
            "bit_width": 4,
            "group_size": 64,
            "regularization_frequency": 1,
            "log_per_layer_stats": False,
        },
        "eval_targets": [
            {"name": "int4", "quantizer_type": "uniform_groupwise", "bit_width": 4, "group_size": 64},
            {"name": "fp8", "quantizer_type": "fp8_e4m3fn", "bit_width": 8, "group_size": 64},
        ],
    },
    "qat_int4": {
        "train_quant": {
            "enable_quant_lora_regularization": True,
            "lambda_q": 0.005,
            "quantizer_type": "uniform_groupwise",
            "bit_width": 4,
            "group_size": 64,
            "regularization_frequency": 1,
            "log_per_layer_stats": False,
        },
        "eval_targets": [
            {"name": "int4", "quantizer_type": "uniform_groupwise", "bit_width": 4, "group_size": 64},
        ],
    },
    "qat_fp8": {
        "train_quant": {
            "enable_quant_lora_regularization": True,
            "lambda_q": 0.002,
            "quantizer_type": "fp8_e4m3fn",
            "bit_width": 8,
            "group_size": 64,
            "regularization_frequency": 1,
            "log_per_layer_stats": False,
        },
        "eval_targets": [
            {"name": "fp8", "quantizer_type": "fp8_e4m3fn", "bit_width": 8, "group_size": 64},
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local LoRA-QAT benchmark with proxy quantized evaluation.")
    parser.add_argument("--python", default=r"C:\conda-envs\torch-gpu\python.exe", help="Python interpreter to use.")
    parser.add_argument("--output-root", default="outputs/local_benchmark", help="Root output directory.")
    parser.add_argument("--write-only", action="store_true", help="Only write configs, do not run experiments.")
    parser.add_argument("--eval-only", action="store_true", help="Reuse existing adapters and rerun evaluation only.")
    return parser.parse_args()


def _build_config(task: TaskSpec, phase_name: str, output_root: Path) -> dict[str, object]:
    experiment_name = f"{task.name}__{phase_name}"
    return {
        "seed": 7,
        "model": {
            "task_type": task.task_type,
            "model_name": task.model_name,
            "max_length": task.max_length,
            "gradient_checkpointing": task.gradient_checkpointing,
        },
        "dataset": task.dataset_payload,
        "lora": {
            "enabled": True,
            "target_modules": task.target_modules,
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.05,
        },
        "training": {
            "output_dir": str((output_root / "runs" / experiment_name).resolve()),
            "device": "auto",
            "mixed_precision": "fp16",
            **task.training_payload,
        },
        "quant_regularization": PHASES[phase_name]["train_quant"],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _run(command: list[str], workdir: Path) -> None:
    subprocess.run(command, cwd=workdir, check=True)


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary_rows(output_root: Path) -> list[dict[str, object]]:
    rows = []
    for metrics_path in sorted((output_root / "runs").glob("*/*quantized_proxy_metrics.json")):
        payload = _load_json(metrics_path)
        experiment_name = metrics_path.parent.name
        task_name, phase_name = experiment_name.split("__", 1)
        rows.append(
            {
                "experiment": experiment_name,
                "task": task_name,
                "phase": phase_name,
                "target_quantizer": payload["quantized_proxy_target"]["quantizer_type"],
                "active_eval_loss": payload["active_metrics"]["eval_loss"],
                "proxy_eval_loss": payload["proxy_quantized_metrics"]["eval_loss"],
                "eval_loss_delta": payload["eval_loss_delta"],
                "primary_metric_name": payload["primary_metric_name"],
                "active_primary_metric": payload["active_primary_metric"],
                "proxy_primary_metric": payload["proxy_primary_metric"],
                "primary_metric_delta": payload["primary_metric_delta"],
                "raw_quant_error": payload["raw_quant_error"],
                "avg_distance_to_grid": payload["avg_distance_to_grid"],
                "num_proxy_quantized_layers": payload["num_proxy_quantized_layers"],
            }
        )
    return rows


def _write_summary(output_root: Path, rows: list[dict[str, object]]) -> None:
    summary_csv = output_root / "summary.csv"
    summary_md = output_root / "summary.md"
    if not rows:
        return

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Local LoRA-QAT Benchmark Summary", ""]
    current_task = None
    for row in rows:
        if row["task"] != current_task:
            current_task = row["task"]
            lines.extend([f"## {current_task}", ""])
        lines.append(
            "- "
            + f"`{row['phase']}` targeting `{row['target_quantizer']}`: "
            + f"active {row['primary_metric_name']}={row['active_primary_metric']:.6f}, "
            + f"proxy_quantized {row['primary_metric_name']}={row['proxy_primary_metric']:.6f}, "
            + f"delta={row['primary_metric_delta']:.6f}, "
            + f"raw_quant_error={row['raw_quant_error']:.6e}"
        )
    lines.append("")

    summary_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    workdir = Path(__file__).resolve().parent

    for task in TASKS:
        for phase_name in PHASES:
            config = _build_config(task, phase_name, output_root)
            config_path = output_root / "configs" / f"{task.name}__{phase_name}.json"
            _write_json(config_path, config)

            if args.write_only:
                continue

            run_dir = Path(config["training"]["output_dir"])
            adapter_dir = run_dir / "adapter"
            if not args.eval_only:
                _run([args.python, "train_hf.py", "--config", str(config_path)], workdir)

            for eval_target in PHASES[phase_name]["eval_targets"]:
                metrics_name = f"quantized_proxy_metrics__{eval_target['name']}.json"
                metrics_path = run_dir / metrics_name
                _run(
                    [
                        args.python,
                        "evaluate_quantized_proxy.py",
                        "--config",
                        str(config_path),
                        "--adapter-dir",
                        str(adapter_dir),
                        "--output",
                        str(metrics_path),
                        "--quantizer-type",
                        eval_target["quantizer_type"],
                        "--bit-width",
                        str(eval_target["bit_width"]),
                        "--group-size",
                        str(eval_target["group_size"]),
                    ],
                    workdir,
                )

    rows = []
    for task in TASKS:
        for phase_name, phase_payload in PHASES.items():
            run_dir = output_root / "runs" / f"{task.name}__{phase_name}"
            for eval_target in phase_payload["eval_targets"]:
                metrics_path = run_dir / f"quantized_proxy_metrics__{eval_target['name']}.json"
                if not metrics_path.exists():
                    continue
                payload = _load_json(metrics_path)
                rows.append(
                    {
                        "experiment": f"{task.name}__{phase_name}",
                        "task": task.name,
                        "phase": phase_name,
                        "target_quantizer": eval_target["name"],
                        "active_eval_loss": payload["active_metrics"]["eval_loss"],
                        "proxy_eval_loss": payload["proxy_quantized_metrics"]["eval_loss"],
                        "eval_loss_delta": payload["eval_loss_delta"],
                        "primary_metric_name": payload["primary_metric_name"],
                        "active_primary_metric": payload["active_primary_metric"],
                        "proxy_primary_metric": payload["proxy_primary_metric"],
                        "primary_metric_delta": payload["primary_metric_delta"],
                        "raw_quant_error": payload["raw_quant_error"],
                        "avg_distance_to_grid": payload["avg_distance_to_grid"],
                        "num_proxy_quantized_layers": payload["num_proxy_quantized_layers"],
                    }
                )

    _write_summary(output_root, rows)
    if rows:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
