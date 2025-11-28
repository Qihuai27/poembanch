#!/usr/bin/env python3
"""
Batch evaluation script for multiple models.

Usage:
    # Evaluate all configs in a directory
    python scripts/eval_batch.py --config-dir configs/models/

    # Evaluate specific configs
    python scripts/eval_batch.py --configs configs/models/qwen3_4b_local.yaml configs/models/openai_gpt4o.yaml

    # With eval config and sampling
    python scripts/eval_batch.py --config-dir configs/models/ --eval-config configs/eval/standard.yaml --sample-n 100
"""
import argparse
import glob
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.base import ModelConfig
from src.models.registry import create_model
from src.evaluation.pipeline import EvaluationPipeline
from src.evaluation.dataset import DEFAULT_SEED
from src.utils.config import load_config, merge_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate models on PoemBench")

    # Config sources
    parser.add_argument("--config-dir", type=str,
                        help="Directory containing model configs")
    parser.add_argument("--configs", type=str, nargs="+",
                        help="Specific config files to evaluate")
    parser.add_argument("--eval-config", type=str,
                        help="Path to evaluation config file")

    # Evaluation settings
    parser.add_argument("--datasets", type=str, nargs="+",
                        help="Datasets to evaluate (overrides eval-config)")
    parser.add_argument("--max-samples", type=int,
                        help="Max samples per dataset (sequential, overrides eval-config)")
    parser.add_argument("--sample-n", type=int,
                        help="Random sample N samples per dataset")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for sampling (default: 1127)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Output directory")

    # Batch options
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue with next model if one fails")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip models that already have results")

    return parser.parse_args()


def find_config_files(args) -> List[str]:
    """Find all config files to process."""
    configs = []

    if args.configs:
        configs.extend(args.configs)

    if args.config_dir:
        patterns = ["*.yaml", "*.yml", "*.json"]
        for pattern in patterns:
            full_pattern = os.path.join(args.config_dir, pattern)
            configs.extend(glob.glob(full_pattern))

    return sorted(set(configs))


def create_model_from_config(config: Dict[str, Any]):
    """Create model from config dictionary."""
    model_type = config.get("model_type", "local")

    # Handle environment variable for api_key
    if config.get("api_key", "").startswith("$"):
        env_var = config["api_key"][1:]
        config["api_key"] = os.getenv(env_var, "")

    model_config = ModelConfig(
        name=config.get("model_name", "unnamed"),
        model_type=model_type,
        model_path=config.get("model_path"),
        device=config.get("device", "auto"),
        torch_dtype=config.get("torch_dtype", "float16"),
        api_key=config.get("api_key"),
        api_base=config.get("api_base"),
        model_name=config.get("api_model_name"),
        max_new_tokens=config.get("max_new_tokens", 128),
        temperature=config.get("temperature", 0.1),
        top_p=config.get("top_p", 0.9),
    )

    return create_model(model_config)


def evaluate_model(
    config_path: str,
    eval_config: Dict[str, Any],
    args
) -> Dict[str, Any]:
    """Evaluate a single model."""
    # Load model config
    model_config = load_config(config_path)
    merged_config = merge_configs(model_config, eval_config)

    # Override with command line args
    if args.datasets:
        merged_config["datasets"] = args.datasets
    if args.max_samples:
        merged_config["max_samples"] = args.max_samples
    if args.sample_n:
        merged_config["sample_n"] = args.sample_n
    if args.seed:
        merged_config["seed"] = args.seed
    if args.data_dir:
        merged_config["data_dir"] = args.data_dir
    if args.output_dir:
        merged_config["output_dir"] = args.output_dir

    model_name = merged_config.get("model_name", os.path.basename(config_path))
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")

    # Create model
    model = create_model_from_config(merged_config)
    model.load()

    # Create pipeline
    pipeline = EvaluationPipeline(
        model=model,
        data_dir=merged_config.get("data_dir", "./data"),
        output_dir=merged_config.get("output_dir", "./results")
    )

    # Run evaluation
    datasets = merged_config.get("datasets")
    max_samples = merged_config.get("max_samples")
    sample_n = merged_config.get("sample_n")
    seed = merged_config.get("seed", DEFAULT_SEED)

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

    # Get results
    summary = pipeline.metrics_calculator.get_summary()

    # Save results
    output_path = pipeline.save_results()

    # Cleanup for local models
    if merged_config.get("model_type") == "local":
        model.unload()

    return {
        "model_name": model_name,
        "config_path": config_path,
        "output_path": output_path,
        "summary": summary
    }


def main():
    args = parse_args()

    # Find config files
    config_files = find_config_files(args)

    if not config_files:
        print("Error: No config files found. Use --configs or --config-dir.")
        sys.exit(1)

    print(f"Found {len(config_files)} config files to evaluate:")
    for f in config_files:
        print(f"  - {f}")

    # Load eval config
    eval_config = {}
    if args.eval_config:
        eval_config = load_config(args.eval_config)
        print(f"\nUsing eval config: {args.eval_config}")

    if args.sample_n:
        print(f"Random sampling {args.sample_n} samples per dataset (seed={args.seed})")

    # Track results
    results = []
    failed = []

    # Evaluate each model
    for config_path in config_files:
        try:
            result = evaluate_model(config_path, eval_config, args)
            results.append(result)

            # Print quick summary
            overall = result["summary"]["overall"]
            print(f"\n{result['model_name']}: "
                  f"IFR={overall['instruction_following_rate']:.2%} | "
                  f"Acc|IF={overall['accuracy_if_followed']:.2%} | "
                  f"Accuracy={overall['accuracy']:.2%} "
                  f"({overall['correct_samples']}/{overall['total_samples']})")

        except Exception as e:
            print(f"\nError evaluating {config_path}: {e}")
            failed.append({"config": config_path, "error": str(e)})

            if not args.continue_on_error:
                raise

    # Print final summary
    print("\n" + "=" * 100)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 100)

    print(f"\nSuccessful: {len(results)}/{len(config_files)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  - {f['config']}: {f['error']}")

    # Print comparison table
    if results:
        print(f"\n{'Model':<25} {'IFR':<10} {'Acc|IF':<10} {'Accuracy':<10} {'Correct/Total':<15}")
        print("-" * 80)

        for r in sorted(results, key=lambda x: -x["summary"]["overall"]["accuracy_if_followed"]):
            name = r["model_name"][:23]
            overall = r["summary"]["overall"]
            ifr = f"{overall['instruction_following_rate']:.2%}"
            acc_if = f"{overall['accuracy_if_followed']:.2%}"
            acc = f"{overall['accuracy']:.2%}"
            ratio = f"{overall['correct_samples']}/{overall['total_samples']}"
            print(f"{name:<25} {ifr:<10} {acc_if:<10} {acc:<10} {ratio:<15}")

    # Save batch summary
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(args.output_dir, f"batch_summary_{timestamp}.json")

    batch_summary = {
        "timestamp": datetime.now().isoformat(),
        "config_files": config_files,
        "eval_config": args.eval_config,
        "sample_n": args.sample_n,
        "seed": args.seed,
        "results": results,
        "failed": failed
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2)

    print(f"\nBatch summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
