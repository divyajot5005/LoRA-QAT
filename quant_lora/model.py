from typing import Dict

from torch import nn

from quant_lora.config import ExperimentConfig
from quant_lora.data import TeacherState
from quant_lora.lora import apply_lora_to_model


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        hidden = self.act(self.fc1(inputs))
        return self.fc2(hidden)


def _load_base_state(model: nn.Module, base_state: Dict[str, object]) -> None:
    model.load_state_dict(base_state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False


def build_model(config: ExperimentConfig, teacher_state: TeacherState) -> nn.Module:
    model = TinyMLP(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim,
    )
    _load_base_state(model, teacher_state.base_state)
    replaced_modules = apply_lora_to_model(model, config.lora)
    if not replaced_modules:
        for parameter in model.parameters():
            parameter.requires_grad = True
    return model
