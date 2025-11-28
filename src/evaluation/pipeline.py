"""
Main evaluation pipeline.
"""
import json
import os
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from tqdm import tqdm

from .dataset import DatasetLoader, TaskSample, DEFAULT_SEED
from .prompt import PromptBuilder, ResponseParser
from .metrics import MetricsCalculator, DetailedResult, EvaluationMetrics
from ..models.base import BaseModel


class EvaluationPipeline:
    """Main evaluation pipeline."""

    def __init__(
        self,
        model: BaseModel,
        data_dir: str = "./data",
        output_dir: str = "./results"
    ):
        """
        Initialize evaluation pipeline.

        Args:
            model: Model to evaluate.
            data_dir: Directory containing datasets.
            output_dir: Directory for output results.
        """
        self.model = model
        self.data_dir = data_dir
        self.output_dir = output_dir

        self.dataset_loader = DatasetLoader(data_dir)
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()
        self.metrics_calculator = MetricsCalculator()

        # Create model-specific output directory
        self.model_output_dir = os.path.join(output_dir, model.name)
        os.makedirs(self.model_output_dir, exist_ok=True)

    def evaluate_sample(self, sample: TaskSample) -> DetailedResult:
        """
        Evaluate a single sample.

        Args:
            sample: TaskSample to evaluate.

        Returns:
            DetailedResult object.
        """
        # Build prompt
        prompt = self.prompt_builder.construct_prompt(sample)

        # Generate response
        gen_result = self.model.generate(prompt)

        # Parse and validate response
        parsed_answer, is_valid_format, is_correct = self.response_parser.parse_and_validate(
            gen_result.response, sample
        )

        return DetailedResult(
            task_id=sample.task_id,
            poem_id=sample.poem_id,
            prompt=prompt,
            response=gen_result.response,
            correct_answer=sample.correct_answer,
            parsed_answer=parsed_answer,
            is_valid_format=is_valid_format and gen_result.success,
            is_correct=is_correct and gen_result.success,
            latency=gen_result.latency,
            tokens_used=gen_result.tokens_used,
            success=gen_result.success,
            error_message=gen_result.error_message
        )

    def evaluate_dataset(
        self,
        dataset_name: str,
        max_samples: Optional[int] = None,
        sample_n: Optional[int] = None,
        seed: int = DEFAULT_SEED,
        show_progress: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluate a single dataset.

        Args:
            dataset_name: Name of the dataset.
            max_samples: Maximum samples to evaluate (sequential).
            sample_n: Number of samples to randomly sample.
            seed: Random seed for sampling (default: 1127).
            show_progress: Show progress bar.

        Returns:
            EvaluationMetrics for the dataset.
        """
        samples = self.dataset_loader.load_dataset(
            dataset_name,
            max_samples=max_samples,
            sample_n=sample_n,
            seed=seed
        )

        if show_progress:
            samples = tqdm(samples, desc=f"Evaluating {dataset_name}")

        for sample in samples:
            result = self.evaluate_sample(sample)
            self.metrics_calculator.add_result(dataset_name, result)

        self.metrics_calculator.finalize()
        return self.metrics_calculator.dataset_metrics[dataset_name]

    def evaluate_datasets(
        self,
        dataset_names: List[str],
        max_samples: Optional[int] = None,
        sample_n: Optional[int] = None,
        seed: int = DEFAULT_SEED,
        show_progress: bool = True
    ) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate multiple datasets.

        Args:
            dataset_names: List of dataset names.
            max_samples: Maximum samples per dataset (sequential).
            sample_n: Number of samples to randomly sample per dataset.
            seed: Random seed for sampling.
            show_progress: Show progress bar.

        Returns:
            Dictionary of dataset name to metrics.
        """
        results = {}

        for name in dataset_names:
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print(f"{'='*60}")

            metrics = self.evaluate_dataset(
                name,
                max_samples=max_samples,
                sample_n=sample_n,
                seed=seed,
                show_progress=show_progress
            )
            results[name] = metrics

            print(f"IFR: {metrics.instruction_following_rate:.2%} | "
                  f"Acc|IF: {metrics.accuracy_if_followed:.2%} | "
                  f"Accuracy: {metrics.accuracy:.2%} "
                  f"({metrics.correct_samples}/{metrics.total_samples})")

        return results

    def evaluate_all(
        self,
        max_samples: Optional[int] = None,
        sample_n: Optional[int] = None,
        seed: int = DEFAULT_SEED,
        corpus: Optional[str] = None,
        task: Optional[str] = None,
        variant: Optional[str] = None,
        show_progress: bool = True
    ) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate all datasets matching criteria.

        Args:
            max_samples: Maximum samples per dataset (sequential).
            sample_n: Number of samples to randomly sample per dataset.
            seed: Random seed for sampling.
            corpus: Filter by corpus.
            task: Filter by task.
            variant: Filter by variant.
            show_progress: Show progress bar.

        Returns:
            Dictionary of dataset name to metrics.
        """
        datasets = self.dataset_loader.filter_datasets(corpus, task, variant)

        if not datasets:
            datasets = self.dataset_loader.list_datasets()

        return self.evaluate_datasets(
            datasets,
            max_samples=max_samples,
            sample_n=sample_n,
            seed=seed,
            show_progress=show_progress
        )

    def save_results(
        self,
        filename: Optional[str] = None,
        include_details: bool = True
    ) -> str:
        """
        Save evaluation results to JSON.

        Results are saved in results/{model_name}/ directory.

        Args:
            filename: Output filename (auto-generated if None).
            include_details: Include detailed per-sample results.

        Returns:
            Path to saved file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_{timestamp}.json"

        output_path = os.path.join(self.model_output_dir, filename)

        summary = self.metrics_calculator.get_summary()

        output_data = {
            "model_name": self.model.name,
            "model_config": {
                "model_type": self.model.config.model_type,
                "model_path": self.model.config.model_path,
                "model_name": self.model.config.model_name,
                "temperature": self.model.config.temperature,
                "max_new_tokens": self.model.config.max_new_tokens,
            },
            "evaluation_time": datetime.now().isoformat(),
            "summary": summary,
        }

        if include_details:
            output_data["detailed_results"] = self.metrics_calculator.get_all_detailed_results()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {output_path}")

        # Also save a simple CSV-like summary for easy reading
        summary_path = os.path.join(self.model_output_dir, "summary.txt")
        self._save_summary_txt(summary_path, summary)

        return output_path

    def _save_summary_txt(self, path: str, summary: Dict[str, Any]) -> None:
        """Save a simple text summary."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {self.model.name}\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"{'Dataset':<35} {'IFR':<10} {'Acc|IF':<10} {'Accuracy':<10} {'Correct/Total':<15}\n")
            f.write("-" * 100 + "\n")

            for name, metrics in sorted(summary["by_dataset"].items()):
                ifr = f"{metrics['instruction_following_rate']:.2%}"
                acc_if = f"{metrics['accuracy_if_followed']:.2%}"
                acc = f"{metrics['accuracy']:.2%}"
                ratio = f"{metrics['correct_samples']}/{metrics['total_samples']}"
                f.write(f"{name:<35} {ifr:<10} {acc_if:<10} {acc:<10} {ratio:<15}\n")

            f.write("-" * 100 + "\n")
            overall = summary["overall"]
            ifr = f"{overall['instruction_following_rate']:.2%}"
            acc_if = f"{overall['accuracy_if_followed']:.2%}"
            acc = f"{overall['accuracy']:.2%}"
            ratio = f"{overall['correct_samples']}/{overall['total_samples']}"
            f.write(f"{'Overall':<35} {ifr:<10} {acc_if:<10} {acc:<10} {ratio:<15}\n")
            f.write("=" * 100 + "\n")
            f.write("\nIFR = Instruction Following Rate\n")
            f.write("Acc|IF = Accuracy if Instruction Followed\n")

    def print_summary(self) -> None:
        """Print evaluation summary."""
        self.metrics_calculator.print_summary()


def run_evaluation(
    model: BaseModel,
    datasets: Optional[List[str]] = None,
    data_dir: str = "./data",
    output_dir: str = "./results",
    max_samples: Optional[int] = None,
    sample_n: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run evaluation.

    Args:
        model: Model to evaluate.
        datasets: List of datasets (None for all).
        data_dir: Data directory.
        output_dir: Output directory.
        max_samples: Max samples per dataset (sequential).
        sample_n: Number of samples to randomly sample per dataset.
        seed: Random seed for sampling (default: 1127).
        save_results: Whether to save results.

    Returns:
        Evaluation summary.
    """
    # Load model if needed
    if not model.is_loaded:
        model.load()

    # Create pipeline
    pipeline = EvaluationPipeline(model, data_dir, output_dir)

    # Run evaluation
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

    # Print and save
    pipeline.print_summary()

    if save_results:
        pipeline.save_results()

    return pipeline.metrics_calculator.get_summary()
