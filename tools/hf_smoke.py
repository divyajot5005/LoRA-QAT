import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small Hugging Face access smoke test.")
    parser.add_argument("--model", required=True, help="Model id to load from Hugging Face.")
    parser.add_argument("--cache-root", default=None, help="Optional HF cache root.")
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Load the full causal LM in addition to the tokenizer.",
    )
    return parser.parse_args()


def configure_cache(cache_root: str | None) -> None:
    if not cache_root:
        return
    os.environ["HF_HOME"] = cache_root
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_root, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_root, "datasets")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_root, "transformers")


def main() -> None:
    args = parse_args()
    configure_cache(args.cache_root)

    print(f"Loading tokenizer for {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_root)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}", flush=True)

    if args.load_model:
        print(f"Loading model for {args.model}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_root)
        print(f"Model loaded: {model.__class__.__name__}", flush=True)


if __name__ == "__main__":
    main()
