from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from quant_lora.config import LoRAConfig


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.weight = nn.Parameter(base_layer.weight.detach().clone(), requires_grad=False)
        if base_layer.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(base_layer.bias.detach().clone(), requires_grad=False)

        self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)

    def lora_update(self) -> torch.Tensor:
        return self.lora_b @ self.lora_a

    def merged_weight(self) -> torch.Tensor:
        return self.weight + self.scaling * self.lora_update()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.merged_weight(), self.bias)


def _get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    if "." not in module_name:
        return model, module_name
    parent_name, child_name = module_name.rsplit(".", 1)
    return model.get_submodule(parent_name), child_name


def apply_lora_to_model(model: nn.Module, lora_config: LoRAConfig) -> List[str]:
    if not lora_config.enabled:
        return []

    replaced_modules: List[str] = []
    target_set = set(lora_config.target_modules)
    named_modules = list(model.named_modules())
    for module_name, module in named_modules:
        if module_name not in target_set:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Target module '{module_name}' is not an nn.Linear.")
        parent, child_name = _get_parent_module(model, module_name)
        setattr(parent, child_name, LoRALinear(module, rank=lora_config.rank, alpha=lora_config.alpha))
        replaced_modules.append(module_name)

    missing = target_set.difference(replaced_modules)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Could not find target modules for LoRA: {missing_str}")
    return replaced_modules


def iter_lora_layers(model: nn.Module) -> Iterable[Tuple[str, LoRALinear]]:
    for module_name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield module_name, module


def lora_parameter_dict(model: nn.Module) -> Dict[str, nn.Parameter]:
    parameters: Dict[str, nn.Parameter] = {}
    for module_name, module in iter_lora_layers(model):
        parameters[f"{module_name}.lora_a"] = module.lora_a
        parameters[f"{module_name}.lora_b"] = module.lora_b
    return parameters
