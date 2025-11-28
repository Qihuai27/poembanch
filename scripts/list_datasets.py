#!/usr/bin/env python3
"""
List available datasets and show statistics.

Usage:
    python scripts/list_datasets.py
    python scripts/list_datasets.py --data-dir ./data
    python scripts/list_datasets.py --corpus tang
    python scripts/list_datasets.py --task guess_author
"""
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.dataset import DatasetLoader


def parse_args():
    parser = argparse.ArgumentParser(description="List available datasets")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory")
    parser.add_argument("--corpus", type=str, help="Filter by corpus")
    parser.add_argument("--task", type=str, help="Filter by task")
    parser.add_argument("--variant", type=str, help="Filter by variant")
    parser.add_argument("--show-samples", type=int, default=0,
                        help="Show N sample entries from each dataset")
    return parser.parse_args()


def main():
    args = parse_args()

    loader = DatasetLoader(args.data_dir)

    # Get summary
    summary = loader.get_summary()

    print("=" * 60)
    print("PoemBench Dataset Summary")
    print("=" * 60)
    print(f"\nTotal datasets: {summary['total_datasets']}")
    print(f"Total samples: {summary['total_samples']:,}")

    # By corpus
    print("\nBy Corpus:")
    for corpus, info in sorted(summary["by_corpus"].items()):
        print(f"  {corpus}: {info['count']} datasets, {info['samples']:,} samples")

    # By task
    print("\nBy Task:")
    for task, info in sorted(summary["by_task"].items()):
        print(f"  {task}: {info['count']} datasets, {info['samples']:,} samples")

    # By variant
    print("\nBy Variant:")
    for variant, info in sorted(summary["by_variant"].items()):
        print(f"  {variant}: {info['count']} datasets, {info['samples']:,} samples")

    # List filtered datasets
    datasets = loader.filter_datasets(args.corpus, args.task, args.variant)

    if args.corpus or args.task or args.variant:
        print(f"\nFiltered datasets ({len(datasets)}):")
    else:
        print(f"\nAll datasets ({len(datasets)}):")

    for name in datasets:
        info = loader.get_dataset_info(name)
        print(f"  {name}: {info.sample_count:,} samples")

        if args.show_samples > 0:
            samples = loader.load_dataset(name, max_samples=args.show_samples)
            for i, sample in enumerate(samples):
                print(f"    [{i+1}] Type: {sample.task_type}")
                print(f"        Prompt: {sample.prompt[:50]}...")
                if sample.demo:
                    print(f"        Demo: {sample.demo[:50]}...")


if __name__ == "__main__":
    main()
