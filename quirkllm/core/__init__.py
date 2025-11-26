"""
QuirkLLM core modules.

This package contains the fundamental components of QuirkLLM:
- system_detector: Hardware detection and system profiling
- profile_manager: RAM-aware profile selection
- config: Configuration management
"""

from quirkllm.core.config import (
    CONFIG_DIR,
    CONFIG_FILE,
    Config,
    generate_default_config,
    get_config_value,
    load_config,
    merge_with_defaults,
    save_config,
    set_config_value,
)
from quirkllm.core.profile_manager import (
    PROFILES,
    ProfileConfig,
    ProfileType,
    select_profile,
)
from quirkllm.core.system_detector import (
    ProcessorType,
    SystemInfo,
    detect_gpu,
    detect_platform,
    detect_ram,
    detect_system,
)

__all__ = [
    # System detector
    "SystemInfo",
    "ProcessorType",
    "detect_system",
    "detect_ram",
    "detect_gpu",
    "detect_platform",
    # Profile manager
    "ProfileType",
    "ProfileConfig",
    "select_profile",
    "PROFILES",
    # Config
    "Config",
    "load_config",
    "save_config",
    "generate_default_config",
    "merge_with_defaults",
    "get_config_value",
    "set_config_value",
    "CONFIG_DIR",
    "CONFIG_FILE",
]
