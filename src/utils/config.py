"""Configuration management using Dynaconf."""

import os
from pathlib import Path
from typing import Any

import yaml
from dynaconf import Dynaconf

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(env: str | None = None) -> dict[str, Any]:
    """Load configuration from YAML files with environment overrides.

    Args:
        env: Environment name (development, staging, production).
             Defaults to ENV env var or 'development'.

    Returns:
        Configuration dictionary.
    """
    env = env or os.getenv("ENV", "development")

    with open(CONFIG_DIR / "base.yaml") as f:
        config = yaml.safe_load(f)

    env_file = CONFIG_DIR / f"{env}.yaml"
    if env_file.exists():
        with open(env_file) as f:
            env_config = yaml.safe_load(f) or {}
            _deep_update(config, env_config)

    return config


def _deep_update(base: dict, update: dict) -> None:
    """Recursively update base dict with update dict."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


class Config:
    """Configuration singleton with environment interpolation."""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if Config._config is None:
            Config._config = Dynaconf(
                settings_files=[
                    str(CONFIG_DIR / "base.yaml"),
                    str(CONFIG_DIR / f"{os.getenv('ENV', 'development')}.yaml"),
                ],
                environments=True,
                env_prefix="APP",
            )

    def __getattr__(self, key: str):
        return getattr(Config._config, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split(".")
        value = Config._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def to_dict(self) -> dict:
        """Export entire config as dictionary."""
        return Config._config.to_dict()


def get_config() -> Config:
    """Get global configuration instance."""
    return Config()
