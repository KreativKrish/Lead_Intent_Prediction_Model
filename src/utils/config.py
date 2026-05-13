"""Configuration management."""

import os
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

_INTERP = re.compile(r"@\{(\w+)\|([^}]*)\}")


def _resolve(value: Any) -> Any:
    """Resolve @{VAR|default} interpolations recursively."""
    if isinstance(value, str):
        return _INTERP.sub(lambda m: os.environ.get(m.group(1), m.group(2)), value)
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(item) for item in value]
    return value


def load_config(env: str | None = None) -> dict[str, Any]:
    """Load and resolve configuration from YAML files with environment overrides."""
    env = env or os.getenv("ENV", "development")

    with open(CONFIG_DIR / "base.yaml") as f:
        config = yaml.safe_load(f)

    env_file = CONFIG_DIR / f"{env}.yaml"
    if env_file.exists():
        with open(env_file) as f:
            env_config = yaml.safe_load(f) or {}
            _deep_update(config, env_config)

    return _resolve(config)


def _deep_update(base: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


class Config:
    """Configuration singleton backed by a plain resolved dict."""

    _instance = None
    _data: dict = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if Config._data is None:
            Config._data = load_config()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value using dot-notation (e.g. 'mlflow.model_name')."""
        value = Config._data
        for k in key.split("."):
            if isinstance(value, dict):
                value = value.get(k, default)
                if value is default:
                    return default
            else:
                return default
        return value

    def to_dict(self) -> dict:
        return dict(Config._data)


def get_config() -> Config:
    """Get global configuration instance."""
    return Config()
