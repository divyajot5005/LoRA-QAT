import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from quant_lora.config import QuantRegularizationConfig


@dataclass
class HFModelConfig:
    task_type: str = "sequence_classification"
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    gradient_checkpointing: bool = False
    torch_dtype: str = "auto"
    attn_implementation: Optional[str] = None


@dataclass
class HFDatasetConfig:
    dataset_name: str = "ag_news"
    dataset_config_name: Optional[str] = None
    train_split: str = "train[:2%]"
    eval_split: str = "test[:2%]"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    format_style: str = "plain_text"
    text_column: str = "text"
    label_column: str = "label"
    prompt_column: Optional[str] = None
    response_column: Optional[str] = None
    context_column: Optional[str] = None
    mask_prompt_tokens: bool = False
    num_proc: int = 1


@dataclass
class HFLoraConfig:
    enabled: bool = True
    target_modules: List[str] = field(default_factory=lambda: ["q_lin", "v_lin"])
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05


@dataclass
class HFTrainingConfig:
    output_dir: str = "outputs"
    device: str = "auto"
    mixed_precision: str = "fp16"
    num_steps: int = 40
    batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.0002
    weight_decay: float = 0.0
    warmup_steps: int = 5
    log_interval: int = 5
    eval_interval: int = 20
    max_eval_batches: Optional[int] = 20
    generation_prompts: List[str] = field(default_factory=list)
    generation_max_new_tokens: int = 80
    generation_temperature: float = 0.8
    generation_top_p: float = 0.95


@dataclass
class HFExperimentConfig:
    seed: int = 7
    model: HFModelConfig = field(default_factory=HFModelConfig)
    dataset: HFDatasetConfig = field(default_factory=HFDatasetConfig)
    lora: HFLoraConfig = field(default_factory=HFLoraConfig)
    training: HFTrainingConfig = field(default_factory=HFTrainingConfig)
    quant_regularization: QuantRegularizationConfig = field(default_factory=QuantRegularizationConfig)



def _load_dataclass(dataclass_type: Any, payload: Dict[str, Any]) -> Any:
    return dataclass_type(**payload)



def load_hf_experiment_config(path: str) -> HFExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    training_payload = dict(payload.get("training", {}))
    if "use_fp16" in training_payload and "mixed_precision" not in training_payload:
        training_payload["mixed_precision"] = "fp16" if training_payload.pop("use_fp16") else "no"

    return HFExperimentConfig(
        seed=payload.get("seed", 7),
        model=_load_dataclass(HFModelConfig, payload.get("model", {})),
        dataset=_load_dataclass(HFDatasetConfig, payload.get("dataset", {})),
        lora=_load_dataclass(HFLoraConfig, payload.get("lora", {})),
        training=_load_dataclass(HFTrainingConfig, training_payload),
        quant_regularization=_load_dataclass(
            QuantRegularizationConfig,
            payload.get("quant_regularization", {}),
        ),
    )
