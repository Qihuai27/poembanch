"""
Model registry for creating models from config.
"""
from typing import Dict, Type, Optional

from .base import BaseModel, ModelConfig
from .local_model import LocalModel
from .api_model import (
    APIModel,
    OpenAIModel,
    AnthropicModel,
    ZhipuModel,
    QwenAPIModel,
    DeepSeekModel,
)


class ModelRegistry:
    """Registry for model classes."""

    _models: Dict[str, Type[BaseModel]] = {
        # Local models
        "local": LocalModel,
        "huggingface": LocalModel,
        "transformers": LocalModel,

        # API models
        "openai": OpenAIModel,
        "gpt": OpenAIModel,
        "anthropic": AnthropicModel,
        "claude": AnthropicModel,
        "zhipu": ZhipuModel,
        "glm": ZhipuModel,
        "qwen_api": QwenAPIModel,
        "dashscope": QwenAPIModel,
        "deepseek": DeepSeekModel,
    }

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a new model class."""
        cls._models[name.lower()] = model_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseModel]]:
        """Get model class by name."""
        return cls._models.get(name.lower())

    @classmethod
    def list_models(cls) -> list:
        """List all registered model names."""
        return list(cls._models.keys())


def create_model(config: ModelConfig) -> BaseModel:
    """
    Create a model instance from config.

    Args:
        config: ModelConfig object.

    Returns:
        BaseModel instance.

    Raises:
        ValueError: If model type is not registered.
    """
    model_class = ModelRegistry.get(config.model_type)

    if model_class is None:
        available = ModelRegistry.list_models()
        raise ValueError(
            f"Unknown model type: {config.model_type}. "
            f"Available types: {available}"
        )

    return model_class(config)


def create_model_from_dict(config_dict: dict) -> BaseModel:
    """
    Create a model instance from a config dictionary.

    Args:
        config_dict: Dictionary with model configuration.

    Returns:
        BaseModel instance.
    """
    config = ModelConfig(
        name=config_dict.get("name", "unnamed"),
        model_type=config_dict.get("model_type", "local"),
        model_path=config_dict.get("model_path"),
        device=config_dict.get("device", "auto"),
        torch_dtype=config_dict.get("torch_dtype", "float16"),
        api_key=config_dict.get("api_key"),
        api_base=config_dict.get("api_base"),
        model_name=config_dict.get("model_name"),
        max_new_tokens=config_dict.get("max_new_tokens", 128),
        temperature=config_dict.get("temperature", 0.1),
        top_p=config_dict.get("top_p", 0.9),
        extra=config_dict.get("extra"),
    )

    return create_model(config)
