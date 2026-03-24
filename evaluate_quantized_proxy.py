import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel

from quant_lora.hf_config import load_hf_experiment_config
from quant_lora.hf_data import build_dataloaders
from quant_lora.hf_model import build_model, load_tokenizer
from quant_lora.hf_trainer import _evaluate, _resolve_device, _set_seed
from quant_lora.peft_regularization import compute_quant_lora_regularization, iter_peft_lora_weights
from quant_lora.quantization import quantize_merged_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained adapter before and after merged-weight proxy quantization.")
    parser.add_argument("--config", required=True, help="Path to the HF experiment config used for training.")
    parser.add_argument("--adapter-dir", required=True, help="Path to the saved adapter directory.")
    parser.add_argument("--output", required=True, help="Path to write the evaluation JSON.")
    parser.add_argument("--device", help="Optional device override, e.g. cuda or cpu.")
    parser.add_argument("--quantizer-type", help="Optional quantizer override.")
    parser.add_argument("--bit-width", type=int, help="Optional bit-width override.")
    parser.add_argument("--group-size", type=int, help="Optional group-size override.")
    return parser.parse_args()


def _resolve_quant_config(config, args: argparse.Namespace) -> dict[str, object]:
    quant_config = config.quant_regularization
    return {
        "quantizer_type": args.quantizer_type or quant_config.quantizer_type,
        "bit_width": args.bit_width if args.bit_width is not None else quant_config.bit_width,
        "group_size": args.group_size if args.group_size is not None else quant_config.group_size,
    }


def _name_candidates(module_name: str) -> list[str]:
    candidates = [module_name]
    prefixes = ["base_model.model.", "base_model."]
    for prefix in prefixes:
        if module_name.startswith(prefix):
            candidates.append(module_name[len(prefix) :])
    return candidates


def _resolve_module_names(merged_model, module_names: set[str]) -> list[tuple[str, torch.nn.Module]]:
    modules_by_name = dict(merged_model.named_modules())
    resolved = []
    for module_name in sorted(module_names):
        module = None
        for candidate in _name_candidates(module_name):
            module = modules_by_name.get(candidate)
            if module is not None:
                module_name = candidate
                break
            suffix_matches = [name for name in modules_by_name if name.endswith(candidate)]
            if len(suffix_matches) == 1:
                module_name = suffix_matches[0]
                module = modules_by_name[module_name]
                break
        if module is None:
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            continue
        resolved.append((module_name, module))
    return resolved


def _apply_proxy_quantization(merged_model, module_names: set[str], quant_payload: dict[str, object]) -> int:
    quantized_layers = 0
    for _, module in _resolve_module_names(merged_model, module_names):
        quantized = quantize_merged_weight(
            module.weight.data.float(),
            quantizer_type=str(quant_payload["quantizer_type"]),
            bit_width=int(quant_payload["bit_width"]),
            group_size=int(quant_payload["group_size"]),
        )
        module.weight.data.copy_(quantized.to(device=module.weight.device, dtype=module.weight.dtype))
        quantized_layers += 1
    return quantized_layers


def _primary_metric(task_type: str, metrics: dict[str, float]) -> tuple[str, float]:
    if task_type == "sequence_classification":
        return "eval_accuracy", float(metrics["eval_accuracy"])
    return "eval_perplexity", float(metrics["eval_perplexity"])


def main() -> None:
    args = parse_args()
    config = load_hf_experiment_config(args.config)
    if args.device:
        config.training.device = args.device

    quant_payload = _resolve_quant_config(config, args)
    adapter_dir = Path(args.adapter_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _set_seed(config.seed)
    device = _resolve_device(config.training.device)
    tokenizer = load_tokenizer(config)
    data_bundle = build_dataloaders(config, tokenizer)

    config.lora.enabled = False
    base_model = build_model(config, data_bundle.num_labels)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    if config.model.task_type == "causal_lm" and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    active_metrics = _evaluate(model, data_bundle.eval_loader, device, config, desc="adapter_eval")
    quant_stats = compute_quant_lora_regularization(model, config.quant_regularization)
    adapted_layers = {module_name for module_name, _ in iter_peft_lora_weights(model)}

    merged_model = model.merge_and_unload()
    merged_model = merged_model.to(device)
    quantized_layers = _apply_proxy_quantization(merged_model, adapted_layers, quant_payload)
    proxy_metrics = _evaluate(merged_model, data_bundle.eval_loader, device, config, desc="proxy_quant_eval")

    primary_name, active_primary = _primary_metric(config.model.task_type, active_metrics)
    _, proxy_primary = _primary_metric(config.model.task_type, proxy_metrics)

    payload = {
        "config_path": str(Path(args.config).resolve()),
        "adapter_dir": str(adapter_dir.resolve()),
        "task_type": config.model.task_type,
        "model_name": config.model.model_name,
        "dataset_name": config.dataset.dataset_name,
        "dataset_config_name": config.dataset.dataset_config_name,
        "quantized_proxy_target": quant_payload,
        "active_metrics": active_metrics,
        "proxy_quantized_metrics": proxy_metrics,
        "primary_metric_name": primary_name,
        "active_primary_metric": active_primary,
        "proxy_primary_metric": proxy_primary,
        "primary_metric_delta": proxy_primary - active_primary,
        "eval_loss_delta": float(proxy_metrics["eval_loss"] - active_metrics["eval_loss"]),
        "raw_quant_error": float(quant_stats.raw_loss.detach().cpu().item()),
        "avg_distance_to_grid": float(quant_stats.avg_distance_to_grid),
        "num_regularized_layers": int(quant_stats.num_regularized_layers),
        "num_proxy_quantized_layers": int(quantized_layers),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
