import math
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable

import torch
import torch.nn.functional as F

from quant_lora.config import QuantRegularizationConfig
from quant_lora.quantization import LayerQuantStats, quantize_merged_weight


@dataclass
class QuantRegularizationResult:
    raw_loss: torch.Tensor
    weighted_loss: torch.Tensor
    avg_distance_to_grid: float
    num_regularized_layers: int = 0
    layer_stats: Dict[str, LayerQuantStats] = field(default_factory=dict)


def _looks_like_lora_layer(module) -> bool:
    return (
        hasattr(module, "get_base_layer")
        and hasattr(module, "get_delta_weight")
        and hasattr(module, "lora_A")
    )


def _iter_adapter_names(module) -> Iterable[str]:
    active_adapters = getattr(module, "active_adapters", None)
    if isinstance(active_adapters, str):
        yield active_adapters
        return
    if active_adapters:
        for name in active_adapters:
            yield name
        return

    lora_a = getattr(module, "lora_A", {})
    if hasattr(lora_a, "keys"):
        for name in lora_a.keys():
            yield name


def _iter_peft_lora_modules(model):
    for module_name, module in model.named_modules():
        if not _looks_like_lora_layer(module):
            continue
        base_layer = module.get_base_layer()
        weight = getattr(base_layer, "weight", None)
        if weight is None or weight.ndim != 2:
            continue

        merged_weight = weight
        found_adapter = False
        for adapter_name in _iter_adapter_names(module):
            if adapter_name not in getattr(module, "lora_A", {}):
                continue
            merged_weight = merged_weight + module.get_delta_weight(adapter_name)
            found_adapter = True
        if not found_adapter:
            continue
        yield module_name, module, merged_weight


def _looks_like_linear_target(module) -> bool:
    if isinstance(module, torch.nn.Linear):
        return True
    return module.__class__.__name__ == "Conv1D"


def _iter_full_ft_modules(model):
    for module_name, module in model.named_modules():
        if _looks_like_lora_layer(module):
            continue
        if not _looks_like_linear_target(module):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2 or not weight.requires_grad:
            continue
        yield module_name, module, weight


def _iter_regularized_modules(model):
    found_lora_modules = False
    for module_name, module, merged_weight in _iter_peft_lora_modules(model):
        found_lora_modules = True
        yield module_name, module, merged_weight
    if found_lora_modules:
        return
    yield from _iter_full_ft_modules(model)


def iter_regularized_weights(model):
    for module_name, _, merged_weight in _iter_regularized_modules(model):
        yield module_name, merged_weight


def iter_peft_lora_weights(model):
    yield from iter_regularized_weights(model)


@contextmanager
def capture_regularized_layer_inputs(model, detach: bool = True):
    cached_inputs: Dict[str, torch.Tensor] = {}
    hooks = []

    for module_name, module, _ in _iter_regularized_modules(model):
        def _capture_inputs(module_inputs, layer_name=module_name):
            if not module_inputs:
                return
            layer_input = module_inputs[0]
            if not isinstance(layer_input, torch.Tensor):
                return
            cached_inputs[layer_name] = layer_input.detach() if detach else layer_input

        hooks.append(module.register_forward_pre_hook(lambda _, inputs, capture=_capture_inputs: capture(inputs)))

    try:
        yield cached_inputs
    finally:
        for hook in hooks:
            hook.remove()


@contextmanager
def capture_lora_layer_inputs(model, detach: bool = True):
    with capture_regularized_layer_inputs(model, detach=detach) as cached_inputs:
        yield cached_inputs


def _resolve_objective(config: QuantRegularizationConfig, objective_override: str | None) -> str:
    objective = objective_override or config.regularization_objective
    if objective not in {"weight_mse", "activation_mse", "logit_kl"}:
        raise ValueError(f"Unsupported regularization_objective: {objective}")
    return objective


def _quant_working_weight(weight: torch.Tensor) -> torch.Tensor:
    if weight.dtype in {torch.float16, torch.bfloat16, torch.float32}:
        return weight
    return weight.float()


def _apply_weight_error(base_layer, module, layer_input: torch.Tensor, weight_error: torch.Tensor) -> torch.Tensor:
    input_fp32 = layer_input.float()
    weight_error_fp32 = weight_error.float()
    if getattr(module, "fan_in_fan_out", False) or base_layer.__class__.__name__ == "Conv1D":
        return torch.matmul(input_fp32, weight_error_fp32)
    return F.linear(input_fp32, weight_error_fp32)


def _apply_merged_weight(base_layer, module, layer_input: torch.Tensor, merged_weight: torch.Tensor) -> torch.Tensor:
    bias = getattr(base_layer, "bias", None)
    if getattr(module, "fan_in_fan_out", False) or base_layer.__class__.__name__ == "Conv1D":
        output = torch.matmul(layer_input, merged_weight.float())
        if bias is not None:
            output = output + bias
        return output
    return F.linear(layer_input, merged_weight, bias)


@contextmanager
def temporarily_patch_regularized_forwards(model, config: QuantRegularizationConfig, use_quantized: bool):
    patched_modules = []
    try:
        for _, module, merged_weight in _iter_regularized_modules(model):
            base_layer = module.get_base_layer() if _looks_like_lora_layer(module) else module
            effective_weight = merged_weight
            if use_quantized:
                quant_source = _quant_working_weight(merged_weight)
                quantized_weight = quantize_merged_weight(
                    quant_source,
                    quantizer_type=config.quantizer_type,
                    bit_width=config.bit_width,
                    group_size=config.group_size,
                ).to(device=merged_weight.device, dtype=merged_weight.dtype)
                # Straight-through estimator: forward uses quantized weights, backward uses merged weights.
                effective_weight = merged_weight + (quantized_weight - merged_weight).detach()

            original_forward = module.forward

            def _patched_forward(self, layer_input, *args, _base_layer=base_layer, _module=module, _weight=effective_weight, **kwargs):
                return _apply_merged_weight(_base_layer, _module, layer_input, _weight)

            module.forward = types.MethodType(_patched_forward, module)
            patched_modules.append((module, original_forward))
        yield
    finally:
        for module, original_forward in reversed(patched_modules):
            module.forward = original_forward


@contextmanager
def temporarily_patch_lora_merged_forwards(model, config: QuantRegularizationConfig, use_quantized: bool):
    with temporarily_patch_regularized_forwards(model, config, use_quantized=use_quantized):
        yield


def compute_logit_kl_loss(
    reference_logits: torch.Tensor,
    quantized_logits: torch.Tensor,
    labels: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    if labels is not None and reference_logits.ndim == labels.ndim + 1:
        valid_mask = labels != -100
        if valid_mask.any():
            reference_logits = reference_logits[valid_mask]
            quantized_logits = quantized_logits[valid_mask]

    reference_logits = reference_logits.float()
    quantized_logits = quantized_logits.float()
    return (
        F.kl_div(
            F.log_softmax(quantized_logits / temperature, dim=-1),
            F.softmax(reference_logits.detach() / temperature, dim=-1),
            reduction="batchmean",
        )
        * (temperature**2)
    )



def compute_quant_lora_regularization(
    model,
    config: QuantRegularizationConfig,
    layer_inputs: Dict[str, torch.Tensor] | None = None,
    objective_override: str | None = None,
) -> QuantRegularizationResult:
    objective = _resolve_objective(config, objective_override)
    device = next(model.parameters()).device
    total_objective_error = torch.zeros((), device=device)
    total_objective_numel = 0
    total_abs_error = 0.0
    total_numel = 0
    num_regularized_layers = 0
    layer_stats: Dict[str, LayerQuantStats] = {}

    for layer_name, module, merged_weight in _iter_regularized_modules(model):
        quant_source = _quant_working_weight(merged_weight)
        quantized_weight = quantize_merged_weight(
            quant_source,
            quantizer_type=config.quantizer_type,
            bit_width=config.bit_width,
            group_size=config.group_size,
        )
        diff = quant_source - quantized_weight
        squared_error_sum = diff.square().sum(dtype=torch.float32)
        abs_error_sum = diff.abs().sum(dtype=torch.float32)
        mse = squared_error_sum / max(numel := diff.numel(), 1)
        avg_distance = (abs_error_sum / max(numel, 1)).item()

        total_abs_error += abs_error_sum.item()
        total_numel += numel
        if config.log_per_layer_stats:
            layer_stats[layer_name] = LayerQuantStats(
                mse=float(mse.detach().item()),
                avg_distance=avg_distance,
                numel=numel,
            )

        if objective == "weight_mse":
            total_objective_error = total_objective_error + squared_error_sum
            total_objective_numel += numel
            num_regularized_layers += 1
            continue

        if layer_inputs is None:
            raise RuntimeError("activation_mse regularization requires captured layer inputs.")
        layer_input = layer_inputs.get(layer_name)
        if layer_input is None:
            continue
        base_layer = module.get_base_layer() if _looks_like_lora_layer(module) else module
        output_error = _apply_weight_error(base_layer, module, layer_input, diff)
        total_objective_error = total_objective_error + output_error.pow(2).sum()
        total_objective_numel += output_error.numel()
        num_regularized_layers += 1

        max_layers = max(0, config.max_activation_regularized_layers)
        if objective == "activation_mse" and max_layers > 0 and num_regularized_layers >= max_layers:
            break

    if total_numel == 0:
        raw_loss = torch.zeros((), device=device)
        avg_distance_to_grid = 0.0
    else:
        avg_distance_to_grid = total_abs_error / total_numel
        if total_objective_numel == 0:
            raw_loss = torch.zeros((), device=device)
        else:
            raw_loss = total_objective_error / total_objective_numel

    weighted_loss = config.lambda_q * raw_loss
    if math.isnan(float(weighted_loss.detach().cpu().item())):
        raise RuntimeError("Quantization regularization produced NaN.")

    return QuantRegularizationResult(
        raw_loss=raw_loss,
        weighted_loss=weighted_loss,
        avg_distance_to_grid=avg_distance_to_grid,
        num_regularized_layers=num_regularized_layers,
        layer_stats=layer_stats,
    )
