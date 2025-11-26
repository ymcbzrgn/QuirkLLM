"""
Configuration management for QuirkLLM.

Handles loading, saving, and validating user configuration from ~/.quirkllm/config.yaml.
Supports profile overrides, theme settings, logging levels, and backend preferences.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Configuration file locations
CONFIG_DIR = Path.home() / ".quirkllm"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
LOGS_DIR = CONFIG_DIR / "logs"
CACHE_DIR = CONFIG_DIR / "cache"
PLANS_DIR = CONFIG_DIR / "plans"
SESSIONS_DIR = CONFIG_DIR / "sessions"


@dataclass
class Config:
    """User configuration settings for QuirkLLM.

    Attributes:
        profile_override: Manual profile selection (survival/comfort/power/beast)
        theme: UI theme (dark/light/auto)
        log_level: Logging verbosity (debug/info/warning/error)
        model_path: Optional custom model path
        backend_type: Backend preference (llama-cpp/mlx/auto)
        context_length: Optional manual context window override
        quantization: Optional quantization override (Q4_K_M/Q8_0/etc)
        auto_save_sessions: Enable automatic session saving
        enable_rag: Enable RAG (Retrieval-Augmented Generation)
        rag_cache_size_mb: RAG cache size in megabytes
        enable_streaming: Enable streaming responses
        confirm_destructive: Require confirmation for destructive operations
        auto_install_deps: Automatically install missing dependencies
        gpu_offload: Enable GPU offload if available
        speculative_decoding: Enable speculative decoding optimization
        custom_settings: Additional user-defined settings
    """

    # Mode
    mode: str = "chat"  # chat, yami, plan, ghost

    # Profile & Performance
    profile_override: str | None = None
    context_length: int | None = None
    quantization: str | None = None

    # Backend
    backend_type: str = "auto"
    model_path: str | None = None
    gpu_offload: bool = True
    speculative_decoding: bool = False

    # RAG & Knowledge
    enable_rag: bool = True
    rag_cache_size_mb: int = 500

    # UI & UX
    theme: str = "dark"
    enable_streaming: bool = True
    log_level: str = "info"

    # Safety & Automation
    auto_save_sessions: bool = True
    confirm_destructive: bool = True
    auto_install_deps: bool = False

    # Custom settings
    custom_settings: dict[str, Any] = field(default_factory=dict)


def ensure_config_directories() -> None:
    """Create configuration directories if they don't exist."""
    for directory in [CONFIG_DIR, LOGS_DIR, CACHE_DIR, PLANS_DIR, SESSIONS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def generate_default_config() -> Config:
    """Generate a default configuration with sensible defaults.

    Returns:
        Config: Default configuration instance
    """
    return Config()


def _validate_enum_field(data: dict[str, Any], field: str, valid_values: set[str]) -> None:
    """Validate that a config field has one of the allowed values.

    Args:
        data: Configuration data dictionary
        field: Field name to validate
        valid_values: Set of valid values for the field

    Raises:
        ValueError: If field value is invalid
    """
    if field in data and data[field] is not None:
        if data[field].lower() not in valid_values:
            raise ValueError(
                f"Invalid {field}: {data[field]}. Must be one of: {', '.join(valid_values)}"
            )


def _validate_config_data(data: dict[str, Any]) -> None:
    """Validate configuration data against allowed values.

    Args:
        data: Configuration data dictionary to validate

    Raises:
        ValueError: If any field contains invalid values
    """
    _validate_enum_field(data, "profile_override", {"survival", "comfort", "power", "beast"})
    _validate_enum_field(data, "theme", {"dark", "light", "auto"})
    _validate_enum_field(data, "log_level", {"debug", "info", "warning", "error"})
    _validate_enum_field(data, "backend_type", {"llama-cpp", "mlx", "auto"})


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Optional custom config file path (default: ~/.quirkllm/config.yaml)

    Returns:
        Config: Configuration instance with values from file or defaults

    Raises:
        ValueError: If config file contains invalid values
    """
    ensure_config_directories()

    path = config_path or CONFIG_FILE

    # Return defaults if config file doesn't exist
    if not path.exists():
        return generate_default_config()

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Validate all configuration fields
        _validate_config_data(data)

        # Create Config instance with validated data
        return Config(
            **{k: v for k, v in data.items() if k != "custom_settings"},
            custom_settings=data.get("custom_settings", {}),
        )

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}") from e


def save_config(config: Config, config_path: Path | None = None) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration instance to save
        config_path: Optional custom config file path (default: ~/.quirkllm/config.yaml)
    """
    ensure_config_directories()

    path = config_path or CONFIG_FILE

    # Convert config to dict and write to YAML
    config_dict = asdict(config)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            config_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True, indent=2
        )


def merge_with_defaults(config: Config, **overrides: Any) -> Config:
    """Merge configuration with runtime overrides.

    Args:
        config: Base configuration
        **overrides: Keyword arguments to override config values

    Returns:
        Config: New configuration with overrides applied
    """
    config_dict = asdict(config)
    config_dict.update({k: v for k, v in overrides.items() if v is not None})
    return Config(
        **{k: v for k, v in config_dict.items() if k != "custom_settings"},
        custom_settings=config_dict.get("custom_settings", {}),
    )


def get_config_value(config: Config, key: str, default: Any = None) -> Any:
    """Get a configuration value by key, checking custom_settings if not found.

    Args:
        config: Configuration instance
        key: Configuration key to retrieve
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    # Check standard attributes first
    if hasattr(config, key):
        return getattr(config, key)

    # Fall back to custom_settings
    return config.custom_settings.get(key, default)


def set_config_value(config: Config, key: str, value: Any) -> Config:
    """Set a configuration value, updating custom_settings if not a standard attribute.

    Args:
        config: Configuration instance
        key: Configuration key to set
        value: Value to set

    Returns:
        Updated configuration instance
    """
    config_dict = asdict(config)

    if hasattr(config, key):
        # Update standard attribute
        config_dict[key] = value
    else:
        # Add to custom_settings
        config_dict["custom_settings"][key] = value

    return Config(
        **{k: v for k, v in config_dict.items() if k != "custom_settings"},
        custom_settings=config_dict["custom_settings"],
    )
