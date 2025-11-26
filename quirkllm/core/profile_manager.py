"""Profile management for RAM-adaptive behavior.

This module defines RAM-based performance profiles and selects the appropriate
profile based on available system memory.

Platform-aware selection:
- macOS: Uses total RAM (aggressive memory management)
- Linux/Windows: Uses available RAM (conservative approach)
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from quirkllm.core.system_detector import SystemInfo


class ProfileType(Enum):
    """RAM-based performance profiles."""

    SURVIVAL = "survival"
    COMFORT = "comfort"
    POWER = "power"
    BEAST = "beast"


CompactionMode = Literal["aggressive", "smart", "relaxed", "minimal"]


@dataclass
class ProfileConfig:
    """Profile configuration matching README specifications.

    Attributes:
        name: Human-readable profile name
        context_length: Maximum context window in tokens
        quantization: Model quantization format (Q4_K_M or Q8_0)
        batch_size: Inference batch size
        rag_cache_mb: RAG vector cache size in MB
        kv_cache_gb: KV-cache size in GB
        embedding_model: CodeBERT variant (small/base/large)
        concurrent_ops: Number of concurrent operations
        compaction_mode: History compaction strategy
        model_loading: Model loading strategy (lazy/hybrid/eager/full)
        expected_speed_toks: Expected inference speed (tokens/second)
    """

    name: str
    context_length: int
    quantization: str
    batch_size: int
    rag_cache_mb: int
    kv_cache_gb: int
    embedding_model: str
    concurrent_ops: int
    compaction_mode: CompactionMode
    model_loading: str
    expected_speed_toks: int


# Profile definitions matching README.md specifications
PROFILES: dict[ProfileType, ProfileConfig] = {
    ProfileType.SURVIVAL: ProfileConfig(
        name="Survival",
        context_length=16384,
        quantization="Q4_K_M",
        batch_size=1,
        rag_cache_mb=200,
        kv_cache_gb=2,
        embedding_model="small",
        concurrent_ops=1,
        compaction_mode="aggressive",
        model_loading="lazy",
        expected_speed_toks=3,
    ),
    ProfileType.COMFORT: ProfileConfig(
        name="Comfort",
        context_length=32768,
        quantization="Q4_K_M",
        batch_size=4,
        rag_cache_mb=500,
        kv_cache_gb=4,
        embedding_model="base",
        concurrent_ops=2,
        compaction_mode="smart",
        model_loading="hybrid",
        expected_speed_toks=5,
    ),
    ProfileType.POWER: ProfileConfig(
        name="Power",
        context_length=65536,
        quantization="Q8_0",
        batch_size=8,
        rag_cache_mb=2048,
        kv_cache_gb=8,
        embedding_model="large",
        concurrent_ops=4,
        compaction_mode="relaxed",
        model_loading="eager",
        expected_speed_toks=8,
    ),
    ProfileType.BEAST: ProfileConfig(
        name="Beast",
        context_length=131072,
        quantization="Q8_0",
        batch_size=16,
        rag_cache_mb=8192,
        kv_cache_gb=16,
        embedding_model="large",
        concurrent_ops=8,
        compaction_mode="minimal",
        model_loading="full",
        expected_speed_toks=12,
    ),
}


def select_profile(system_info: "SystemInfo", override: str | None = None) -> ProfileConfig:
    """Select profile based on platform-aware RAM detection.

    Platform-aware strategy:
    - macOS (darwin): Uses total RAM (aggressive memory compression/swap)
    - Linux/Windows: Uses adjusted available RAM (conservative approach)

    Profile thresholds (from README):
    - < 8GB: Survival
    - 8-24GB: Comfort
    - 24-48GB: Power
    - 48GB+: Beast

    Args:
        system_info: System detection results (from system_detector.detect_system())
        override: Manual profile override (e.g., "comfort", "power")

    Returns:
        ProfileConfig for the selected profile

    Raises:
        ValueError: If override is not a valid profile name
    """
    # Manual override
    if override:
        try:
            profile_type = ProfileType(override.lower())
            return PROFILES[profile_type]
        except ValueError as e:
            valid_profiles = [p.value for p in ProfileType]
            raise ValueError(
                f"Invalid profile '{override}'. Valid options: {valid_profiles}"
            ) from e

    # Platform-aware RAM decision
    if system_info.platform == "darwin":
        # macOS: Use total RAM (aggressive memory management)
        # Reason: macOS aggressively compresses memory, uses fast SSD swap,
        # and manages file cache smartly. "Available RAM" is misleading.
        decision_ram = system_info.total_ram_gb
    else:
        # Linux/Windows: Use adjusted available RAM (conservative)
        # Reason: Traditional swap mechanisms, varied hardware performance.
        decision_ram = system_info.adjusted_ram_gb

    # Auto-select based on RAM thresholds (from README specs)
    if decision_ram < 8:
        return PROFILES[ProfileType.SURVIVAL]
    elif decision_ram < 24:
        return PROFILES[ProfileType.COMFORT]
    elif decision_ram < 48:
        return PROFILES[ProfileType.POWER]
    else:
        return PROFILES[ProfileType.BEAST]
