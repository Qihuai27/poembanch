"""
Configuration management utilities.
"""
import json
import yaml
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # Model settings
    model_name: str
    model_type: str
    model_path: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_model_name: Optional[str] = None

    # Generation settings
    max_new_tokens: int = 128
    temperature: float = 0.1
    top_p: float = 0.9
    device: str = "auto"
    torch_dtype: str = "float16"

    # Evaluation settings
    datasets: Optional[List[str]] = None
    max_samples: Optional[int] = None
    data_dir: str = "./data"
    output_dir: str = "./results"

    # Batch settings
    batch_size: int = 1
    save_every: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvalConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from file.

    Supports JSON and YAML formats.

    Args:
        path: Path to config file.

    Returns:
        Configuration dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith('.yaml') or path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary.
        path: Output path.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        if path.endswith('.yaml') or path.endswith('.yml'):
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        else:
            json.dump(config, f, ensure_ascii=False, indent=2)


def load_model_config(path: str) -> Dict[str, Any]:
    """
    Load model configuration.

    Args:
        path: Path to model config file.

    Returns:
        Model configuration dictionary.
    """
    config = load_config(path)

    # Handle environment variables for API keys
    if 'api_key' in config and config['api_key']:
        if config['api_key'].startswith('$'):
            env_var = config['api_key'][1:]
            config['api_key'] = os.getenv(env_var, '')

    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configurations.

    Later configs override earlier ones.

    Args:
        *configs: Configuration dictionaries.

    Returns:
        Merged configuration.
    """
    result = {}
    for config in configs:
        result.update(config)
    return result
