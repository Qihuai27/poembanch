"""
Base model interface for all model implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration dataclass."""
    name: str
    model_type: str  # 'local' or 'api'

    # Local model specific
    model_path: Optional[str] = None
    device: str = "auto"
    torch_dtype: str = "float16"

    # API model specific
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_name: Optional[str] = None

    # Generation parameters
    max_new_tokens: int = 128
    temperature: float = 0.1
    top_p: float = 0.9

    # Extra parameters
    extra: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResult:
    """Result of a single generation."""
    response: str
    raw_response: Any = None
    latency: float = 0.0
    tokens_used: int = 0
    success: bool = True
    error_message: str = ""


class BaseModel(ABC):
    """Abstract base class for all model implementations."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the model. Must be called before generate()."""
        pass

    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        """
        Generate a response for the given prompt.

        Args:
            prompt: The input prompt string.

        Returns:
            GenerationResult containing the response and metadata.
        """
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str]) -> List[GenerationResult]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts.

        Returns:
            List of GenerationResult objects.
        """
        pass

    def unload(self) -> None:
        """Unload the model to free resources."""
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def name(self) -> str:
        """Get model name."""
        return self.config.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name}, loaded={self._loaded})"
