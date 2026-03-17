import argparse

from quant_lora.config import load_experiment_config
from quant_lora.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantization-aware LoRA prototype")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
