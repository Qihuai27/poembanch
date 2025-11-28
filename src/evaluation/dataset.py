"""
Dataset loading and management.
"""
import json
import glob
import os
import random
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass, field

# Default random seed for reproducible sampling
DEFAULT_SEED = 1127


@dataclass
class TaskSample:
    """Single task sample from dataset."""
    task_id: str
    poem_id: str
    task_type: str  # 'multiple_choice' or 'sorting'
    prompt: str
    demo: Optional[str]
    choices: List[str]
    goal: Any  # str for multiple_choice, List[str] for sorting
    hint: str = ""

    # Computed fields
    correct_answer: str = ""

    def __post_init__(self):
        """Compute correct answer after initialization."""
        self.correct_answer = self._compute_correct_answer()

    def _compute_correct_answer(self) -> str:
        """Compute the correct answer string."""
        if self.task_type == 'multiple_choice':
            try:
                idx = self.choices.index(self.goal) + 1
                return str(idx)
            except ValueError:
                return "Error: Goal not in choices"

        elif self.task_type == 'sorting':
            indices = []
            for sentence in self.goal:
                try:
                    idx = self.choices.index(sentence) + 1
                    indices.append(str(idx))
                except ValueError:
                    return "Error: Goal sentence not in choices"
            return "".join(indices)

        return ""


@dataclass
class DatasetInfo:
    """Dataset metadata."""
    name: str
    file_path: str
    corpus: str  # tang, song, tang300, tangsong, today
    task: str  # guess_author, guess_word, guess_ci_tone, match_sentence, sort_poem
    variant: str  # standard, fewshot1/3/10, cot, couplets, jue, lyu
    sample_count: int = 0


class DatasetLoader:
    """Load and manage datasets."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self._datasets: Dict[str, DatasetInfo] = {}
        self._scan_datasets()

    def _scan_datasets(self) -> None:
        """Scan data directory for datasets."""
        pattern = os.path.join(self.data_dir, "*.jsonl")
        files = glob.glob(pattern)

        for file_path in files:
            filename = os.path.basename(file_path)
            name = filename.replace(".jsonl", "")

            # Parse filename: corpus.task.variant.jsonl
            parts = name.split(".")
            if len(parts) >= 3:
                corpus = parts[0]
                task = parts[1]
                variant = ".".join(parts[2:])
            else:
                corpus = parts[0] if len(parts) > 0 else "unknown"
                task = parts[1] if len(parts) > 1 else "unknown"
                variant = "standard"

            # Count samples
            sample_count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for _ in f:
                    sample_count += 1

            self._datasets[name] = DatasetInfo(
                name=name,
                file_path=file_path,
                corpus=corpus,
                task=task,
                variant=variant,
                sample_count=sample_count
            )

    def list_datasets(self) -> List[str]:
        """List all available dataset names."""
        return sorted(self._datasets.keys())

    def get_dataset_info(self, name: str) -> Optional[DatasetInfo]:
        """Get dataset info by name."""
        return self._datasets.get(name)

    def filter_datasets(
        self,
        corpus: Optional[str] = None,
        task: Optional[str] = None,
        variant: Optional[str] = None
    ) -> List[str]:
        """Filter datasets by criteria."""
        results = []
        for name, info in self._datasets.items():
            if corpus and info.corpus != corpus:
                continue
            if task and info.task != task:
                continue
            if variant and info.variant != variant:
                continue
            results.append(name)
        return sorted(results)

    def load_dataset(
        self,
        name: str,
        max_samples: Optional[int] = None,
        sample_n: Optional[int] = None,
        seed: int = DEFAULT_SEED
    ) -> List[TaskSample]:
        """
        Load a dataset by name.

        Args:
            name: Dataset name (without .jsonl extension).
            max_samples: Maximum number of samples to load (sequential from start).
            sample_n: Number of samples to randomly sample (uses seed for reproducibility).
            seed: Random seed for sampling (default: 1127).

        Returns:
            List of TaskSample objects.

        Note:
            If both max_samples and sample_n are provided, sample_n takes precedence.
        """
        info = self._datasets.get(name)
        if info is None:
            raise ValueError(f"Dataset not found: {name}")

        # Load all samples first
        all_samples = []
        with open(info.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                sample = TaskSample(
                    task_id=data.get("task_id", f"{name}.{i}"),
                    poem_id=data.get("poem_id", ""),
                    task_type=data.get("type", "multiple_choice"),
                    prompt=data.get("prompt", ""),
                    demo=data.get("demo"),
                    choices=data.get("choices", []),
                    goal=data.get("goal"),
                    hint=data.get("hint", "")
                )
                all_samples.append(sample)

        # Apply sampling strategy
        if sample_n is not None and sample_n < len(all_samples):
            # Random sampling with fixed seed
            rng = random.Random(seed)
            samples = rng.sample(all_samples, sample_n)
        elif max_samples is not None and max_samples < len(all_samples):
            # Sequential sampling from start
            samples = all_samples[:max_samples]
        else:
            samples = all_samples

        return samples

    def iterate_dataset(
        self,
        name: str,
        max_samples: Optional[int] = None
    ) -> Iterator[TaskSample]:
        """
        Iterate over a dataset lazily.

        Args:
            name: Dataset name.
            max_samples: Maximum number of samples.

        Yields:
            TaskSample objects.
        """
        info = self._datasets.get(name)
        if info is None:
            raise ValueError(f"Dataset not found: {name}")

        with open(info.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break

                data = json.loads(line.strip())
                yield TaskSample(
                    task_id=data.get("task_id", f"{name}.{i}"),
                    poem_id=data.get("poem_id", ""),
                    task_type=data.get("type", "multiple_choice"),
                    prompt=data.get("prompt", ""),
                    demo=data.get("demo"),
                    choices=data.get("choices", []),
                    goal=data.get("goal"),
                    hint=data.get("hint", "")
                )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all datasets."""
        summary = {
            "total_datasets": len(self._datasets),
            "total_samples": sum(d.sample_count for d in self._datasets.values()),
            "by_corpus": {},
            "by_task": {},
            "by_variant": {},
        }

        for info in self._datasets.values():
            # By corpus
            if info.corpus not in summary["by_corpus"]:
                summary["by_corpus"][info.corpus] = {"count": 0, "samples": 0}
            summary["by_corpus"][info.corpus]["count"] += 1
            summary["by_corpus"][info.corpus]["samples"] += info.sample_count

            # By task
            if info.task not in summary["by_task"]:
                summary["by_task"][info.task] = {"count": 0, "samples": 0}
            summary["by_task"][info.task]["count"] += 1
            summary["by_task"][info.task]["samples"] += info.sample_count

            # By variant
            if info.variant not in summary["by_variant"]:
                summary["by_variant"][info.variant] = {"count": 0, "samples": 0}
            summary["by_variant"][info.variant]["count"] += 1
            summary["by_variant"][info.variant]["samples"] += info.sample_count

        return summary
