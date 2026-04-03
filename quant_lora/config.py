import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ModelConfig:
    input_dim: int = 32
    hidden_dim: int = 64
    output_dim: int = 6


@dataclass
class DataConfig:
    num_train: int = 512
    num_val: int = 256
    teacher_rank: int = 4
    teacher_delta_scale: float = 0.18
    noise_std: float = 0.02


@dataclass
class LoRAConfig:
    enabled: bool = True
    target_modules: List[str] = field(default_factory=lambda: ["fc1", "fc2"])
    rank: int = 4
    alpha: float = 8.0


@dataclass
class TrainingConfig:
    num_steps: int = 250
    batch_size: int = 64
    learning_rate: float = 0.02
    weight_decay: float = 0.0
    log_interval: int = 25
    eval_interval: int = 50


@dataclass
class QuantRegularizationConfig:
    enable_quant_lora_regularization: bool = False
    lambda_q: float = 0.0
    quantizer_type: str = "uniform_per_tensor"
    bit_width: int = 4
    group_size: int = 16
    regularization_objective: str = "weight_mse"
    logit_kl_temperature: float = 1.0
    regularization_frequency: int = 1
    detach_layer_inputs: bool = True
    max_activation_regularized_layers: int = 0
    log_per_layer_stats: bool = False


@dataclass
class ExperimentConfig:
    seed: int = 7
    device: str = "cpu"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quant_regularization: QuantRegularizationConfig = field(default_factory=QuantRegularizationConfig)


def _load_dataclass(dataclass_type: Any, payload: Dict[str, Any]) -> Any:
    return dataclass_type(**payload)


def load_experiment_config(path: str) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    defaults = ExperimentConfig()
    return ExperimentConfig(
        seed=payload.get("seed", defaults.seed),
        device=payload.get("device", defaults.device),
        model=_load_dataclass(ModelConfig, payload.get("model", {})),
        data=_load_dataclass(DataConfig, payload.get("data", {})),
        lora=_load_dataclass(LoRAConfig, payload.get("lora", {})),
        training=_load_dataclass(TrainingConfig, payload.get("training", {})),
        quant_regularization=_load_dataclass(
            QuantRegularizationConfig,
            payload.get("quant_regularization", {}),
        ),
    )
