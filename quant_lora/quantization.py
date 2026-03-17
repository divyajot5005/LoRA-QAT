from dataclasses import dataclass, field
from typing import Dict

import torch


def _symmetric_quantize(weights: torch.Tensor, scale: torch.Tensor, qmax: int) -> torch.Tensor:
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.clamp(torch.round(weights / safe_scale), -qmax, qmax) * safe_scale
    return torch.where(scale > 0, quantized, torch.zeros_like(weights))


def _quantize_per_tensor(weights: torch.Tensor, bit_width: int) -> torch.Tensor:
    qmax = 2 ** (bit_width - 1) - 1
    scale = weights.abs().amax()
    scale = scale / max(qmax, 1)
    return _symmetric_quantize(weights, scale, qmax)


def _quantize_per_channel(weights: torch.Tensor, bit_width: int) -> torch.Tensor:
    qmax = 2 ** (bit_width - 1) - 1
    scale = weights.abs().amax(dim=1, keepdim=True) / max(qmax, 1)
    return _symmetric_quantize(weights, scale, qmax)


def _quantize_groupwise(weights: torch.Tensor, bit_width: int, group_size: int) -> torch.Tensor:
    if group_size <= 0:
        raise ValueError("group_size must be positive for uniform_groupwise quantization.")

    qmax = 2 ** (bit_width - 1) - 1
    out_features, in_features = weights.shape
    remainder = in_features % group_size
    pad_size = (group_size - remainder) % group_size
    if pad_size > 0:
        padded_weights = torch.nn.functional.pad(weights, (0, pad_size))
    else:
        padded_weights = weights

    num_groups = padded_weights.size(1) // group_size
    grouped = padded_weights.view(out_features, num_groups, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True) / max(qmax, 1)
    quantized = _symmetric_quantize(grouped, scale, qmax).view(out_features, -1)
    return quantized[:, :in_features]


def _quantize_fp8(weights: torch.Tensor, quantizer_type: str) -> torch.Tensor:
    if quantizer_type == "fp8_e4m3fn":
        dtype = torch.float8_e4m3fn
    elif quantizer_type == "fp8_e5m2":
        dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 quantizer_type: {quantizer_type}")

    if not hasattr(torch, str(dtype).split(".")[-1]):
        raise RuntimeError(f"Current PyTorch build does not support {quantizer_type}.")
    return weights.to(dtype).to(weights.dtype)


def quantize_merged_weight(
    weights: torch.Tensor,
    quantizer_type: str,
    bit_width: int,
    group_size: int,
) -> torch.Tensor:
    if quantizer_type in {"fp8_e4m3fn", "fp8_e5m2"}:
        return _quantize_fp8(weights, quantizer_type)

    if bit_width < 2:
        raise ValueError("bit_width must be at least 2.")

    if quantizer_type == "uniform_per_tensor":
        return _quantize_per_tensor(weights, bit_width)
    if quantizer_type == "uniform_per_channel":
        return _quantize_per_channel(weights, bit_width)
    if quantizer_type == "uniform_groupwise":
        return _quantize_groupwise(weights, bit_width, group_size)
    raise ValueError(f"Unsupported quantizer_type: {quantizer_type}")


@dataclass
class LayerQuantStats:
    mse: float
    avg_distance: float
    numel: int


@dataclass
class QuantRegularizationResult:
    raw_loss: torch.Tensor
    weighted_loss: torch.Tensor
    avg_distance_to_grid: float
    layer_stats: Dict[str, LayerQuantStats] = field(default_factory=dict)
