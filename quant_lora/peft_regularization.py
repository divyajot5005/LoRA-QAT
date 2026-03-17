import math
from dataclasses import dataclass, field
from typing import Dict

import torch
from peft.tuners.lora.layer import LoraLayer

from quant_lora.config import QuantRegularizationConfig
from quant_lora.quantization import LayerQuantStats, quantize_merged_weight


@dataclass
class QuantRegularizationResult:
    raw_loss: torch.Tensor
    weighted_loss: torch.Tensor
    avg_distance_to_grid: float
    layer_stats: Dict[str, LayerQuantStats] = field(default_factory=dict)



def iter_peft_lora_weights(model):
    for module_name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        base_layer = module.get_base_layer()
        weight = getattr(base_layer, "weight", None)
        if weight is None or weight.ndim != 2:
            continue

        active_adapters = getattr(module, "active_adapters", None)
        if not active_adapters:
            active_adapters = [name for name in module.lora_A.keys()]
        merged_weight = weight
        for adapter_name in active_adapters:
            if adapter_name not in module.lora_A:
                continue
            merged_weight = merged_weight + module.get_delta_weight(adapter_name)
        yield module_name, merged_weight



def compute_quant_lora_regularization(model, config: QuantRegularizationConfig) -> QuantRegularizationResult:
    device = next(model.parameters()).device
    total_squared_error = torch.zeros((), device=device)
    total_abs_error = 0.0
    total_numel = 0
    layer_stats: Dict[str, LayerQuantStats] = {}

    for layer_name, merged_weight in iter_peft_lora_weights(model):
        merged_weight_fp32 = merged_weight.float()
        quantized_weight = quantize_merged_weight(
            merged_weight_fp32,
            quantizer_type=config.quantizer_type,
            bit_width=config.bit_width,
            group_size=config.group_size,
        )
        diff = merged_weight_fp32 - quantized_weight
        mse = diff.pow(2).mean()
        avg_distance = diff.abs().mean().item()
        numel = diff.numel()

        total_squared_error = total_squared_error + diff.pow(2).sum()
        total_abs_error += diff.abs().sum().item()
        total_numel += numel
        if config.log_per_layer_stats:
            layer_stats[layer_name] = LayerQuantStats(
                mse=float(mse.detach().item()),
                avg_distance=avg_distance,
                numel=numel,
            )

    if total_numel == 0:
        raw_loss = torch.zeros((), device=device)
        avg_distance_to_grid = 0.0
    else:
        raw_loss = total_squared_error / total_numel
        avg_distance_to_grid = total_abs_error / total_numel

    weighted_loss = config.lambda_q * raw_loss
    if math.isnan(float(weighted_loss.detach().cpu().item())):
        raise RuntimeError("Quantization regularization produced NaN.")

    return QuantRegularizationResult(
        raw_loss=raw_loss,
        weighted_loss=weighted_loss,
        avg_distance_to_grid=avg_distance_to_grid,
        layer_stats=layer_stats,
    )
