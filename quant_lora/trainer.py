import random
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from quant_lora.config import ExperimentConfig
from quant_lora.data import build_datasets
from quant_lora.model import build_model
from quant_lora.quantization import QuantRegularizationResult, compute_quant_lora_regularization


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters were found. Enable LoRA or unfreeze the model.")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def _evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            total_correct += (logits.argmax(dim=-1) == targets).sum().item()

    return total_loss / total_examples, total_correct / total_examples


def _format_layer_stats(layer_stats: Dict[str, object]) -> str:
    if not layer_stats:
        return ""
    parts = []
    for layer_name, stats in sorted(layer_stats.items()):
        parts.append(f"{layer_name}:mse={stats.mse:.6f},avg_dist={stats.avg_distance:.6f}")
    return " | " + " ; ".join(parts)


def _should_regularize(step: int, config: ExperimentConfig) -> bool:
    quant_config = config.quant_regularization
    if not quant_config.enable_quant_lora_regularization:
        return False
    frequency = max(1, quant_config.regularization_frequency)
    return step % frequency == 0


def run_training(config: ExperimentConfig) -> None:
    _set_seed(config.seed)
    device = torch.device(config.device)

    train_dataset, val_dataset, teacher_state = build_datasets(config.model, config.data, config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    model = build_model(config, teacher_state).to(device)
    optimizer = _build_optimizer(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    print("Experiment configuration:")
    print(config)
    print()

    step = 0
    while step < config.training.num_steps:
        for batch_inputs, batch_targets in train_loader:
            if step >= config.training.num_steps:
                break

            model.train()
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            logits = model(batch_inputs)
            task_loss = criterion(logits, batch_targets)

            quant_result = QuantRegularizationResult(
                raw_loss=torch.zeros((), device=device),
                weighted_loss=torch.zeros((), device=device),
                avg_distance_to_grid=0.0,
            )
            if _should_regularize(step, config):
                quant_result = compute_quant_lora_regularization(model, config.quant_regularization)

            total_loss = task_loss + quant_result.weighted_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            if step % config.training.log_interval == 0:
                predictions = logits.argmax(dim=-1)
                accuracy = (predictions == batch_targets).float().mean().item()
                print(
                    f"step={step:04d} "
                    f"task_loss={task_loss.item():.6f} "
                    f"quant_reg_loss={quant_result.weighted_loss.item():.6f} "
                    f"avg_distance_to_grid={quant_result.avg_distance_to_grid:.6f} "
                    f"train_acc={accuracy:.4f}"
                    f"{_format_layer_stats(quant_result.layer_stats)}"
                )

            if (step + 1) % config.training.eval_interval == 0:
                val_loss, val_accuracy = _evaluate(model, val_loader, device)
                print(f"eval step={step + 1:04d} val_loss={val_loss:.6f} val_acc={val_accuracy:.4f}")

            step += 1

    final_val_loss, final_val_accuracy = _evaluate(model, val_loader, device)
    print()
    print(f"final val_loss={final_val_loss:.6f} final val_acc={final_val_accuracy:.4f}")
