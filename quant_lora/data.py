from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset

from quant_lora.config import DataConfig, ModelConfig


@dataclass
class TeacherState:
    base_state: Dict[str, torch.Tensor]
    teacher_state: Dict[str, torch.Tensor]


def _sample_linear_weight(shape: Tuple[int, int], generator: torch.Generator, scale: float = 0.25) -> torch.Tensor:
    return torch.randn(shape, generator=generator) * scale


def _sample_linear_bias(size: int, generator: torch.Generator, scale: float = 0.05) -> torch.Tensor:
    return torch.randn(size, generator=generator) * scale


def _sample_low_rank_delta(
    out_features: int,
    in_features: int,
    rank: int,
    delta_scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    a = torch.randn(rank, in_features, generator=generator) * delta_scale
    b = torch.randn(out_features, rank, generator=generator) * delta_scale
    return b @ a


def build_teacher_state(model_config: ModelConfig, data_config: DataConfig, seed: int) -> TeacherState:
    generator = torch.Generator().manual_seed(seed)

    fc1_weight = _sample_linear_weight((model_config.hidden_dim, model_config.input_dim), generator)
    fc1_bias = _sample_linear_bias(model_config.hidden_dim, generator)
    fc2_weight = _sample_linear_weight((model_config.output_dim, model_config.hidden_dim), generator)
    fc2_bias = _sample_linear_bias(model_config.output_dim, generator)

    base_state = {
        "fc1.weight": fc1_weight,
        "fc1.bias": fc1_bias,
        "fc2.weight": fc2_weight,
        "fc2.bias": fc2_bias,
    }

    teacher_state = {
        "fc1.weight": fc1_weight + _sample_low_rank_delta(
            model_config.hidden_dim,
            model_config.input_dim,
            data_config.teacher_rank,
            data_config.teacher_delta_scale,
            generator,
        ),
        "fc1.bias": fc1_bias.clone(),
        "fc2.weight": fc2_weight + _sample_low_rank_delta(
            model_config.output_dim,
            model_config.hidden_dim,
            data_config.teacher_rank,
            data_config.teacher_delta_scale,
            generator,
        ),
        "fc2.bias": fc2_bias.clone(),
    }

    return TeacherState(base_state=base_state, teacher_state=teacher_state)


def _forward_teacher(inputs: torch.Tensor, teacher_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    hidden = torch.relu(torch.nn.functional.linear(inputs, teacher_state["fc1.weight"], teacher_state["fc1.bias"]))
    return torch.nn.functional.linear(hidden, teacher_state["fc2.weight"], teacher_state["fc2.bias"])


class SyntheticTeacherStudentDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor) -> None:
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return int(self.inputs.size(0))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.labels[index]


def build_datasets(
    model_config: ModelConfig,
    data_config: DataConfig,
    seed: int,
) -> Tuple[SyntheticTeacherStudentDataset, SyntheticTeacherStudentDataset, TeacherState]:
    teacher_state = build_teacher_state(model_config, data_config, seed)
    total_examples = data_config.num_train + data_config.num_val

    generator = torch.Generator().manual_seed(seed + 1)
    inputs = torch.randn(total_examples, model_config.input_dim, generator=generator)
    logits = _forward_teacher(inputs, teacher_state.teacher_state)
    if data_config.noise_std > 0.0:
        noise = torch.randn(logits.shape, generator=generator, device=logits.device, dtype=logits.dtype)
        logits = logits + noise * data_config.noise_std
    labels = logits.argmax(dim=-1)

    train_inputs = inputs[: data_config.num_train]
    train_labels = labels[: data_config.num_train]
    val_inputs = inputs[data_config.num_train :]
    val_labels = labels[data_config.num_train :]

    train_dataset = SyntheticTeacherStudentDataset(train_inputs, train_labels)
    val_dataset = SyntheticTeacherStudentDataset(val_inputs, val_labels)
    return train_dataset, val_dataset, teacher_state
