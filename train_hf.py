import argparse

from quant_lora.hf_config import load_hf_experiment_config
from quant_lora.hf_trainer import run_hf_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantization-aware LoRA Hugging Face trainer")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_hf_experiment_config(args.config)
    run_hf_training(config)


if __name__ == "__main__":
    main()
