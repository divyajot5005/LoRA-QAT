import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import torch
from transformers import get_linear_schedule_with_warmup

from quant_lora.hf_config import HFExperimentConfig
from quant_lora.hf_data import build_dataloaders
from quant_lora.hf_model import build_model, load_tokenizer
from quant_lora.peft_regularization import QuantRegularizationResult, compute_quant_lora_regularization


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _resolve_autocast_dtype(mixed_precision: str):
    mapping = {
        "no": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if mixed_precision not in mapping:
        raise ValueError(f"Unsupported mixed_precision mode: {mixed_precision}")
    return mapping[mixed_precision]


def _build_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters were found for the Hugging Face run.")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def _format_layer_stats(layer_stats: Dict[str, object]) -> str:
    if not layer_stats:
        return ""
    parts = []
    for layer_name, stats in sorted(layer_stats.items()):
        parts.append(f"{layer_name}:mse={stats.mse:.6f},avg_dist={stats.avg_distance:.6f}")
    return " | " + " ; ".join(parts)


def _should_regularize(update_step: int, config: HFExperimentConfig) -> bool:
    quant_config = config.quant_regularization
    if not quant_config.enable_quant_lora_regularization:
        return False
    frequency = max(1, quant_config.regularization_frequency)
    return update_step % frequency == 0


def _move_batch_to_device(batch, device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def _evaluate(model, dataloader, device: torch.device, config: HFExperimentConfig):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if config.training.max_eval_batches is not None and batches >= config.training.max_eval_batches:
                break
            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            if config.model.task_type == "sequence_classification":
                predictions = outputs.logits.argmax(dim=-1)
                total_correct += (predictions == batch["labels"]).sum().item()

            batches += 1

    avg_loss = total_loss / max(total_examples, 1)
    metrics = {"eval_loss": avg_loss}
    if config.model.task_type == "sequence_classification":
        metrics["eval_accuracy"] = total_correct / max(total_examples, 1)
    else:
        metrics["eval_perplexity"] = math.exp(min(avg_loss, 20.0))
    return metrics


def _save_final_metrics(output_dir: Path, config: HFExperimentConfig, metrics: Dict[str, float], peak_gpu_mem_mb: float) -> None:
    payload = {
        "seed": config.seed,
        "task_type": config.model.task_type,
        "model_name": config.model.model_name,
        "dataset_name": config.dataset.dataset_name,
        "dataset_config_name": config.dataset.dataset_config_name,
        "train_split": config.dataset.train_split,
        "eval_split": config.dataset.eval_split,
        "num_steps": config.training.num_steps,
        "batch_size": config.training.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "mixed_precision": config.training.mixed_precision,
        "enable_quant_lora_regularization": config.quant_regularization.enable_quant_lora_regularization,
        "lambda_q": config.quant_regularization.lambda_q,
        "quantizer_type": config.quant_regularization.quantizer_type,
        "bit_width": config.quant_regularization.bit_width,
        "group_size": config.quant_regularization.group_size,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "final_metrics": metrics,
    }
    with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _save_model_artifacts(model, tokenizer, output_dir: Path) -> None:
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)


def _generate_and_save_samples(
    model,
    tokenizer,
    device: torch.device,
    config: HFExperimentConfig,
    output_dir: Path,
) -> None:
    prompts = config.training.generation_prompts
    if config.model.task_type != "causal_lm" or not prompts:
        return

    model.eval()
    samples = []
    with torch.no_grad():
        for prompt in prompts:
            tokenized = tokenizer(prompt, return_tensors="pt").to(device)
            generated = model.generate(
                **tokenized,
                max_new_tokens=config.training.generation_max_new_tokens,
                do_sample=True,
                temperature=config.training.generation_temperature,
                top_p=config.training.generation_top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            samples.append({"prompt": prompt, "generation": text})

    with (output_dir / "sample_generations.json").open("w", encoding="utf-8") as handle:
        json.dump(samples, handle, indent=2)


def run_hf_training(config: HFExperimentConfig) -> None:
    _set_seed(config.seed)
    device = _resolve_device(config.training.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    tokenizer = load_tokenizer(config)
    data_bundle = build_dataloaders(config, tokenizer)
    model = build_model(config, data_bundle.num_labels)

    if config.model.task_type == "causal_lm" and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model = model.to(device)
    optimizer = _build_optimizer(model, config.training.learning_rate, config.training.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.num_steps,
    )

    autocast_dtype = _resolve_autocast_dtype(config.training.mixed_precision)
    use_scaler = autocast_dtype == torch.float16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    accumulation_steps = max(1, config.training.gradient_accumulation_steps)

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(json.loads(json.dumps(config, default=lambda x: x.__dict__)), handle, indent=2)

    print("Experiment configuration:")
    print(config)
    print()
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
        print()

    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    micro_step = 0
    train_iterator = iter(data_bundle.train_loader)

    while update_step < config.training.num_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(data_bundle.train_loader)
            batch = next(train_iterator)

        batch = _move_batch_to_device(batch, device)
        quant_result = QuantRegularizationResult(
            raw_loss=torch.zeros((), device=device),
            weighted_loss=torch.zeros((), device=device),
            avg_distance_to_grid=0.0,
            num_regularized_layers=0,
        )

        if autocast_dtype is not None and device.type == "cuda":
            autocast_context = torch.amp.autocast("cuda", dtype=autocast_dtype)
        else:
            autocast_context = nullcontext()

        with autocast_context:
            outputs = model(**batch)
            task_loss = outputs.loss
            if _should_regularize(update_step, config):
                quant_result = compute_quant_lora_regularization(model, config.quant_regularization)
            total_loss = (task_loss + quant_result.weighted_loss) / accumulation_steps

        if use_scaler:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        micro_step += 1
        if micro_step % accumulation_steps != 0:
            continue

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if update_step % config.training.log_interval == 0:
            if config.model.task_type == "sequence_classification":
                predictions = outputs.logits.argmax(dim=-1)
                train_metric = (predictions == batch["labels"]).float().mean().item()
                metric_str = f"train_acc={train_metric:.4f}"
            else:
                metric_str = f"train_ppl={math.exp(min(task_loss.item(), 20.0)):.4f}"

            gpu_mem = ""
            if device.type == "cuda":
                gpu_mem = f" gpu_mem_mb={torch.cuda.max_memory_allocated(device) / 1024**2:.1f}"
            print(
                f"step={update_step:04d} "
                f"task_loss={task_loss.item():.6f} "
                f"quant_reg_loss={quant_result.weighted_loss.item():.6e} "
                f"quant_raw_loss={quant_result.raw_loss.item():.6e} "
                f"avg_distance_to_grid={quant_result.avg_distance_to_grid:.6e} "
                f"regularized_layers={quant_result.num_regularized_layers} "
                f"{metric_str}"
                f"{gpu_mem}"
                f"{_format_layer_stats(quant_result.layer_stats)}"
            )

        if (update_step + 1) % config.training.eval_interval == 0:
            metrics = _evaluate(model, data_bundle.eval_loader, device, config)
            metric_str = " ".join(f"{name}={value:.6f}" for name, value in metrics.items())
            print(f"eval step={update_step + 1:04d} {metric_str}")

        update_step += 1

    final_metrics = _evaluate(model, data_bundle.eval_loader, device, config)
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    _save_final_metrics(output_dir, config, final_metrics, peak_gpu_mem_mb)
    _save_model_artifacts(model, tokenizer, output_dir)
    _generate_and_save_samples(model, tokenizer, device, config, output_dir)

    metric_str = " ".join(f"{name}={value:.6f}" for name, value in final_metrics.items())
    print()
    print(f"final {metric_str}")
    print(f"saved_metrics={output_dir / 'final_metrics.json'}")
    print(f"saved_adapter={output_dir / 'adapter'}")
    if config.model.task_type == "causal_lm" and config.training.generation_prompts:
        print(f"saved_generations={output_dir / 'sample_generations.json'}")
