"""
QuirkLLM Modes Package

Four operational modes with different confirmation and automation levels:
- Chat: Safe mode with confirmations (default)
- YAMI: Auto-accept with safety validation
- Plan: Read-only planning mode
- Ghost: Background file watcher

Architecture:
- ModeBase: Abstract base class for all modes
- ModeRegistry: Global mode registry and factory
- SafetyChecker: Critical pattern detection and validation
"""

from quirkllm.modes.base import (
    ModeType,
    ModeConfig,
    ActionRequest,
    ActionResult,
    ModeBase,
)
from quirkllm.modes.registry import ModeRegistry, get_registry
from quirkllm.modes.chat_mode import ChatMode
from quirkllm.modes.safety_checker import SafetyChecker, ValidationResult
from quirkllm.modes.yami_mode import YAMIMode
from quirkllm.modes.plan_mode import PlanMode
from quirkllm.modes.ghost_mode import GhostMode

__all__ = [
    "ModeType",
    "ModeConfig",
    "ActionRequest",
    "ActionResult",
    "ModeBase",
    "ModeRegistry",
    "get_registry",
    "ChatMode",
    "SafetyChecker",
    "ValidationResult",
    "YAMIMode",
    "PlanMode",
    "GhostMode",
]


def _register_modes():
    """Auto-register all modes on import."""
    registry = get_registry()

    mode_classes = [
        (ModeType.CHAT, ChatMode),
        (ModeType.YAMI, YAMIMode),
        (ModeType.PLAN, PlanMode),
        (ModeType.GHOST, GhostMode),
    ]

    for mode_type, mode_class in mode_classes:
        if mode_type not in registry._modes:
            registry.register(mode_type, mode_class)


# Auto-register modes when package is imported
_register_modes()
