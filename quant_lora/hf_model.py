import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from quant_lora.hf_config import HFExperimentConfig


def load_tokenizer(config: HFExperimentConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    return mapping[dtype_name]


def build_model(config: HFExperimentConfig, num_labels: int | None):
    model_kwargs = {}
    torch_dtype = _resolve_torch_dtype(config.model.torch_dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if config.model.attn_implementation:
        model_kwargs["attn_implementation"] = config.model.attn_implementation

    if config.model.task_type == "sequence_classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model.model_name,
            num_labels=num_labels,
            **model_kwargs,
        )
    elif config.model.task_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(config.model.model_name, **model_kwargs)
    else:
        raise ValueError(f"Unsupported task_type: {config.model.task_type}")

    if config.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if not config.lora.enabled:
        return model

    task_type = TaskType.SEQ_CLS if config.model.task_type == "sequence_classification" else TaskType.CAUSAL_LM
    peft_config = LoraConfig(
        task_type=task_type,
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        target_modules=config.lora.target_modules,
    )
    model = get_peft_model(model, peft_config)
    return model
