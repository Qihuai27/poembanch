"""
Metrics calculation for evaluation.
"""
from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class EvaluationMetrics:
    """Metrics for a single dataset evaluation."""
    dataset_name: str
    total_samples: int = 0
    correct_samples: int = 0
    failed_samples: int = 0  # Generation failures (API errors, etc.)
    invalid_format_samples: int = 0  # Valid response but wrong format (instruction not followed)

    # Computed metrics
    accuracy: float = 0.0  # Overall accuracy (correct / total)
    instruction_following_rate: float = 0.0  # Rate of valid format responses
    accuracy_if_followed: float = 0.0  # Accuracy among valid format responses

    total_latency: float = 0.0
    avg_latency: float = 0.0
    total_tokens: int = 0

    def update(
        self,
        is_correct: bool,
        is_valid_format: bool,
        latency: float,
        tokens: int,
        success: bool
    ):
        """Update metrics with a sample result."""
        self.total_samples += 1

        if not success:
            # API/generation failure
            self.failed_samples += 1
        elif not is_valid_format:
            # Got response but format is invalid (instruction not followed)
            self.invalid_format_samples += 1
        elif is_correct:
            # Valid format and correct answer
            self.correct_samples += 1

        self.total_latency += latency
        self.total_tokens += tokens

    def finalize(self):
        """Compute final metrics."""
        # Samples that got a response (not API failures)
        responded_samples = self.total_samples - self.failed_samples

        # Samples with valid format (instruction followed)
        valid_format_samples = responded_samples - self.invalid_format_samples

        # Instruction following rate: valid_format / responded
        if responded_samples > 0:
            self.instruction_following_rate = valid_format_samples / responded_samples

        # Accuracy if instruction followed: correct / valid_format
        if valid_format_samples > 0:
            self.accuracy_if_followed = self.correct_samples / valid_format_samples

        # Overall accuracy: correct / total
        if self.total_samples > 0:
            self.accuracy = self.correct_samples / self.total_samples
            self.avg_latency = self.total_latency / self.total_samples

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "failed_samples": self.failed_samples,
            "invalid_format_samples": self.invalid_format_samples,
            "instruction_following_rate": self.instruction_following_rate,
            "accuracy_if_followed": self.accuracy_if_followed,
            "accuracy": self.accuracy,
            "total_latency": self.total_latency,
            "avg_latency": self.avg_latency,
            "total_tokens": self.total_tokens,
        }


@dataclass
class DetailedResult:
    """Detailed result for a single sample."""
    task_id: str
    poem_id: str
    prompt: str
    response: str
    correct_answer: str
    parsed_answer: str
    is_valid_format: bool  # Whether response follows instruction format
    is_correct: bool  # Whether answer is correct (only meaningful if is_valid_format)
    latency: float
    tokens_used: int
    success: bool  # Whether generation succeeded (no API errors)
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "poem_id": self.poem_id,
            "prompt": self.prompt,
            "response": self.response,
            "correct_answer": self.correct_answer,
            "parsed_answer": self.parsed_answer,
            "is_valid_format": self.is_valid_format,
            "is_correct": self.is_correct,
            "latency": self.latency,
            "tokens_used": self.tokens_used,
            "success": self.success,
            "error_message": self.error_message,
        }


class MetricsCalculator:
    """Calculate and aggregate metrics."""

    def __init__(self):
        self.dataset_metrics: Dict[str, EvaluationMetrics] = {}
        self.detailed_results: Dict[str, List[DetailedResult]] = {}

    def add_result(
        self,
        dataset_name: str,
        result: DetailedResult
    ) -> None:
        """Add a result for a dataset."""
        # Initialize if needed
        if dataset_name not in self.dataset_metrics:
            self.dataset_metrics[dataset_name] = EvaluationMetrics(
                dataset_name=dataset_name
            )
            self.detailed_results[dataset_name] = []

        # Update metrics
        self.dataset_metrics[dataset_name].update(
            is_correct=result.is_correct,
            is_valid_format=result.is_valid_format,
            latency=result.latency,
            tokens=result.tokens_used,
            success=result.success
        )

        # Store detailed result
        self.detailed_results[dataset_name].append(result)

    def finalize(self) -> None:
        """Finalize all metrics."""
        for metrics in self.dataset_metrics.values():
            metrics.finalize()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        self.finalize()

        total_samples = sum(m.total_samples for m in self.dataset_metrics.values())
        total_correct = sum(m.correct_samples for m in self.dataset_metrics.values())
        total_failed = sum(m.failed_samples for m in self.dataset_metrics.values())
        total_invalid = sum(m.invalid_format_samples for m in self.dataset_metrics.values())

        responded_samples = total_samples - total_failed
        valid_format_samples = responded_samples - total_invalid

        overall_ifr = valid_format_samples / responded_samples if responded_samples > 0 else 0.0
        overall_acc_if_followed = total_correct / valid_format_samples if valid_format_samples > 0 else 0.0
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {
            "overall": {
                "total_samples": total_samples,
                "correct_samples": total_correct,
                "failed_samples": total_failed,
                "invalid_format_samples": total_invalid,
                "instruction_following_rate": overall_ifr,
                "accuracy_if_followed": overall_acc_if_followed,
                "accuracy": overall_accuracy,
            },
            "by_dataset": {
                name: metrics.to_dict()
                for name, metrics in self.dataset_metrics.items()
            }
        }

    def get_detailed_results(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get detailed results for a dataset."""
        if dataset_name not in self.detailed_results:
            return []
        return [r.to_dict() for r in self.detailed_results[dataset_name]]

    def get_all_detailed_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all detailed results."""
        return {
            name: self.get_detailed_results(name)
            for name in self.detailed_results.keys()
        }

    def print_summary(self) -> None:
        """Print summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 100)
        print("Evaluation Summary")
        print("=" * 100)

        header = f"{'Dataset':<35} {'IFR':<10} {'Acc|IF':<10} {'Accuracy':<10} {'Correct/Total':<15}"
        print(f"\n{header}")
        print("-" * 100)

        for name, metrics in sorted(summary["by_dataset"].items()):
            ifr = f"{metrics['instruction_following_rate']:.2%}"
            acc_if = f"{metrics['accuracy_if_followed']:.2%}"
            acc = f"{metrics['accuracy']:.2%}"
            ratio = f"{metrics['correct_samples']}/{metrics['total_samples']}"
            print(f"{name:<35} {ifr:<10} {acc_if:<10} {acc:<10} {ratio:<15}")

        print("-" * 100)
        overall = summary["overall"]
        ifr = f"{overall['instruction_following_rate']:.2%}"
        acc_if = f"{overall['accuracy_if_followed']:.2%}"
        acc = f"{overall['accuracy']:.2%}"
        ratio = f"{overall['correct_samples']}/{overall['total_samples']}"
        print(f"{'Overall':<35} {ifr:<10} {acc_if:<10} {acc:<10} {ratio:<15}")
        print("=" * 100)
        print("\nIFR = Instruction Following Rate")
        print("Acc|IF = Accuracy if Instruction Followed")
