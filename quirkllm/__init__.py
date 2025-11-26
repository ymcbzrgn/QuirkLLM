"""QuirkLLM - Local, free, GPU-optional AI coding assistant."""

__version__ = "0.1.0"
__author__ = "Yamac Bezirgan"
__license__ = "Apache-2.0"

from quirkllm.core.profile_manager import ProfileConfig, ProfileType
from quirkllm.core.system_detector import SystemInfo

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "ProfileConfig",
    "ProfileType",
    "SystemInfo",
]
