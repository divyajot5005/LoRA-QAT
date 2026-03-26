import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from quant_lora.hf_config import HFExperimentConfig


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class DataBundle:
    train_loader: DataLoader
    eval_loader: DataLoader
    num_labels: Optional[int] = None


class CausalLMCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch_inputs = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(batch_inputs, padding=True, return_tensors="pt")
        max_length = batch["input_ids"].size(1)

        padded_labels = []
        for feature in features:
            labels = feature["labels"]
            pad_size = max_length - len(labels)
            padded_labels.append(labels + [-100] * pad_size)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def build_dataloaders(config: HFExperimentConfig, tokenizer: PreTrainedTokenizerBase) -> DataBundle:
    if config.model.task_type == "sequence_classification":
        return _build_sequence_classification_loaders(config, tokenizer)
    if config.model.task_type == "causal_lm":
        return _build_causal_lm_loaders(config, tokenizer)
    raise ValueError(f"Unsupported task_type: {config.model.task_type}")


def _load_split(config: HFExperimentConfig, split: str):
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    dataset = load_dataset(
        config.dataset.dataset_name,
        name=config.dataset.dataset_config_name,
        split=split,
        cache_dir=cache_dir,
    )
    return dataset.shuffle(seed=config.seed)


def _infer_num_labels(dataset, label_column: str) -> int:
    label_feature = dataset.features[label_column]
    if hasattr(label_feature, "num_classes") and label_feature.num_classes is not None:
        return int(label_feature.num_classes)
    unique_labels = dataset.unique(label_column)
    return len(unique_labels)


def _build_sequence_classification_loaders(
    config: HFExperimentConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> DataBundle:
    raw_train_dataset = _load_split(config, config.dataset.train_split)
    raw_eval_dataset = _load_split(config, config.dataset.eval_split)
    if config.dataset.max_train_samples is not None:
        raw_train_dataset = raw_train_dataset.select(range(min(len(raw_train_dataset), config.dataset.max_train_samples)))
    if config.dataset.max_eval_samples is not None:
        raw_eval_dataset = raw_eval_dataset.select(range(min(len(raw_eval_dataset), config.dataset.max_eval_samples)))
    num_labels = _infer_num_labels(raw_train_dataset, config.dataset.label_column)

    def tokenize_batch(batch):
        tokens = tokenizer(
            batch[config.dataset.text_column],
            truncation=True,
            max_length=config.model.max_length,
        )
        tokens["labels"] = batch[config.dataset.label_column]
        return tokens

    train_dataset = raw_train_dataset.map(tokenize_batch, batched=True, remove_columns=raw_train_dataset.column_names)
    eval_dataset = raw_eval_dataset.map(tokenize_batch, batched=True, remove_columns=raw_eval_dataset.column_names)
    train_dataset.set_format(type="torch")
    eval_dataset.set_format(type="torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=config.training.eval_batch_size, shuffle=False, collate_fn=collator)
    return DataBundle(train_loader=train_loader, eval_loader=eval_loader, num_labels=num_labels)


def _build_causal_lm_loaders(
    config: HFExperimentConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> DataBundle:
    raw_train_dataset = _load_split(config, config.dataset.train_split)
    raw_eval_dataset = _load_split(config, config.dataset.eval_split)
    if config.dataset.max_train_samples is not None:
        raw_train_dataset = raw_train_dataset.select(range(min(len(raw_train_dataset), config.dataset.max_train_samples)))
    if config.dataset.max_eval_samples is not None:
        raw_eval_dataset = raw_eval_dataset.select(range(min(len(raw_eval_dataset), config.dataset.max_eval_samples)))
    format_style = config.dataset.format_style

    if format_style == "plain_text":
        train_dataset = raw_train_dataset.filter(lambda example: bool(example[config.dataset.text_column].strip()))
        eval_dataset = raw_eval_dataset.filter(lambda example: bool(example[config.dataset.text_column].strip()))

        def tokenize_example(example):
            return _tokenize_plain_text_example(example[config.dataset.text_column], tokenizer, config.model.max_length)

    elif format_style == "instruction_response":
        train_dataset = raw_train_dataset
        eval_dataset = raw_eval_dataset

        def tokenize_example(example):
            return _tokenize_instruction_response_example(example, tokenizer, config)

    else:
        raise ValueError(f"Unsupported causal LM format_style: {format_style}")

    train_dataset = train_dataset.map(tokenize_example, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(tokenize_example, remove_columns=eval_dataset.column_names)

    collator = CausalLMCollator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=config.training.eval_batch_size, shuffle=False, collate_fn=collator)
    return DataBundle(train_loader=train_loader, eval_loader=eval_loader, num_labels=None)


def _tokenize_plain_text_example(text: str, tokenizer: PreTrainedTokenizerBase, max_length: int) -> Dict[str, List[int]]:
    tokens = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=False)
    input_ids = list(tokens["input_ids"])
    attention_mask = list(tokens["attention_mask"])
    if tokenizer.eos_token_id is not None and len(input_ids) < max_length:
        input_ids.append(tokenizer.eos_token_id)
        attention_mask.append(1)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": list(input_ids),
    }


def _tokenize_instruction_response_example(
    example: Dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    config: HFExperimentConfig,
) -> Dict[str, List[int]]:
    prompt = _format_instruction_prompt(example, config)
    response_column = config.dataset.response_column
    if response_column is None:
        raise ValueError("response_column must be set for instruction_response format.")

    response_text = str(example[response_column]).strip()
    if not response_text:
        response_text = "I do not have enough information to answer that."

    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=config.model.max_length,
        add_special_tokens=False,
    )["input_ids"]
    response_ids = tokenizer(
        response_text,
        truncation=True,
        max_length=config.model.max_length,
        add_special_tokens=False,
    )["input_ids"]

    eos_tokens = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    available_response_tokens = max(config.model.max_length - len(prompt_ids) - len(eos_tokens), 0)
    response_ids = response_ids[:available_response_tokens]

    input_ids = prompt_ids + response_ids + eos_tokens
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)

    if config.dataset.mask_prompt_tokens:
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _format_instruction_prompt(example: Dict[str, str], config: HFExperimentConfig) -> str:
    prompt_column = config.dataset.prompt_column or config.dataset.text_column
    prompt_text = str(example[prompt_column]).strip()

    parts = [f"Instruction:\n{prompt_text}"]
    if config.dataset.context_column:
        context_value = str(example[config.dataset.context_column]).strip()
        if context_value:
            parts.append(f"Context:\n{context_value}")
    parts.append("Response:\n")
    return "\n\n".join(parts)
