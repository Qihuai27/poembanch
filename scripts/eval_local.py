#!/usr/bin/env python3
"""
Evaluate a single local model on PoemBench.

Usage:
    python scripts/eval_local.py --model-path ./models/Qwen3-4B
    python scripts/eval_local.py --config configs/models/qwen3_4b_local.yaml
    python scripts/eval_local.py --model-path ./models/Qwen3-4B --sample-n 100  # Random sample 100
"""
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.base import ModelConfig
from src.models.registry import create_model
from src.evaluation.pipeline import EvaluationPipeline
from src.evaluation.dataset import DEFAULT_SEED
from src.utils.config import load_config, merge_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate local model on PoemBench")

    # Config file
    parser.add_argument("--config", type=str, help="Path to model config file")
    parser.add_argument("--eval-config", type=str, help="Path to evaluation config file")

    # Model settings (override config)
    parser.add_argument("--model-path", type=str, help="Path to local model")
    parser.add_argument("--model-name", type=str, help="Model name for logging")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"], help="Device to use")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"], help="Torch dtype")

    # Generation settings
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")

    # Evaluation settings
    parser.add_argument("--datasets", type=str, nargs="+", help="Datasets to evaluate")
    parser.add_argument("--max-samples", type=int, help="Max samples per dataset (sequential)")
    parser.add_argument("--sample-n", type=int, help="Random sample N samples per dataset")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for sampling")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--output-file", type=str, help="Output filename")

    # Other options
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument("--no-details", action="store_true", help="Don't save detailed results")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config from file if provided
    config = {}
    if args.config:
        config = load_config(args.config)

    # Load eval config if provided
    eval_config = {}
    if args.eval_config:
        eval_config = load_config(args.eval_config)
        config = merge_configs(config, eval_config)

    # Override with command line arguments
    if args.model_path:
        config["model_path"] = args.model_path
    if args.model_name:
        config["model_name"] = args.model_name
    if args.device:
        config["device"] = args.device
    if args.dtype:
        config["torch_dtype"] = args.dtype
    if args.max_tokens:
        config["max_new_tokens"] = args.max_tokens
    if args.temperature is not None:
        config["temperature"] = args.temperature
    if args.top_p is not None:
        config["top_p"] = args.top_p
    if args.datasets:
        config["datasets"] = args.datasets
    if args.max_samples:
        config["max_samples"] = args.max_samples
    if args.sample_n:
        config["sample_n"] = args.sample_n
    if args.seed:
        config["seed"] = args.seed
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir

    # Set model type to local
    config["model_type"] = "local"

    # Validate required fields
    if "model_path" not in config:
        print("Error: model_path is required. Use --model-path or specify in config file.")
        sys.exit(1)

    # Check model path exists
    if not os.path.exists(config["model_path"]):
        print(f"Error: Model path does not exist: {config['model_path']}")
        sys.exit(1)

    # Generate model name from path if not specified
    if "model_name" not in config:
        config["model_name"] = os.path.basename(config["model_path"])

    # Create model config
    model_config = ModelConfig(
        name=config["model_name"],
        model_type="local",
        model_path=config["model_path"],
        device=config.get("device", "auto"),
        torch_dtype=config.get("torch_dtype", "float16"),
        max_new_tokens=config.get("max_new_tokens", 128),
        temperature=config.get("temperature", 0.1),
        top_p=config.get("top_p", 0.9),
    )

    # Create model
    print(f"Creating model: {model_config.name}")
    print(f"  Path: {model_config.model_path}")
    print(f"  Device: {model_config.device}")
    print(f"  Dtype: {model_config.torch_dtype}")

    model = create_model(model_config)
    model.load()

    # Create pipeline
    pipeline = EvaluationPipeline(
        model=model,
        data_dir=config.get("data_dir", "./data"),
        output_dir=config.get("output_dir", "./results")
    )

    # Run evaluation
    datasets = config.get("datasets")
    max_samples = config.get("max_samples")
    sample_n = config.get("sample_n")
    seed = config.get("seed", DEFAULT_SEED)

    if sample_n:
        print(f"Random sampling {sample_n} samples per dataset (seed={seed})")

    if datasets:
        pipeline.evaluate_datasets(
            datasets,
            max_samples=max_samples,
            sample_n=sample_n,
            seed=seed
        )
    else:
        pipeline.evaluate_all(
            max_samples=max_samples,
            sample_n=sample_n,
            seed=seed
        )

    # Print summary
    pipeline.print_summary()

    # Save results
    if not args.no_save:
        pipeline.save_results(
            filename=args.output_file,
            include_details=not args.no_details
        )

    # Cleanup
    model.unload()


if __name__ == "__main__":
    main()
