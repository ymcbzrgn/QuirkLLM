"""System resource detection for QuirkLLM.

This module detects system resources (RAM, GPU, platform) to enable
RAM-aware adaptive behavior across different hardware configurations.
"""

import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

import psutil

PlatformType = Literal["darwin", "linux", "windows"]
ProcessorType = Literal["arm", "x86_64", "unknown"]


@dataclass
class SystemInfo:
    """System information for profile selection.

    Attributes:
        total_ram_gb: Total system RAM in GB
        available_ram_gb: Currently available RAM in GB (CRITICAL: used for profile selection)
        adjusted_ram_gb: Available RAM after platform-specific adjustments
        cpu_count: Number of logical CPU cores
        platform: Operating system (darwin/linux/windows)
        processor: Processor architecture (arm/x86_64/unknown)
        has_cuda: Whether CUDA GPU is available
        has_metal: Whether Metal (Apple Silicon) is available
    """

    total_ram_gb: float
    available_ram_gb: float
    adjusted_ram_gb: float
    cpu_count: int
    platform: PlatformType
    processor: ProcessorType
    has_cuda: bool
    has_metal: bool


def detect_ram() -> tuple[float, float, float]:
    """Detect RAM using psutil.

    Uses psutil.virtual_memory().available which accounts for:
    - Linux: free + buffers + cache (reclaimable)
    - macOS: free + inactive memory
    - Windows: free + standby (reclaimable cache)

    Platform-specific adjustments:
    - macOS Apple Silicon: Reserve 2GB for Metal buffer pool
    - Linux with active swap: Apply 20% penalty (system under pressure)

    Returns:
        Tuple of (total_gb, available_gb, adjusted_gb)
        adjusted_gb is guaranteed to be at least 1.0 GB
    """
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)

    # Start with available RAM
    adjusted = available_gb

    # macOS Apple Silicon: Reserve 2GB for Metal buffer pool
    if sys.platform == "darwin" and platform.processor() == "arm":
        adjusted -= 2.0

    # Linux: Penalize if swap is active (system under pressure)
    if sys.platform == "linux":
        swap = psutil.swap_memory()
        if swap.percent > 10:
            adjusted *= 0.8  # 20% penalty

    # Ensure minimum 1GB
    return total_gb, available_gb, max(adjusted, 1.0)


def detect_gpu() -> tuple[bool, bool]:
    """Detect GPU availability.

    Checks for:
    - CUDA: Uses nvidia-smi command (NVIDIA GPU)
    - Metal: Checks for macOS Apple Silicon

    Returns:
        Tuple of (has_cuda, has_metal)
    """
    has_cuda = False
    has_metal = False

    # CUDA detection (NVIDIA GPU)
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=2,
            check=False,
        )
        has_cuda = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Metal detection (macOS Apple Silicon)
    if sys.platform == "darwin" and platform.processor() == "arm":
        has_metal = True

    return has_cuda, has_metal


def detect_platform() -> tuple[PlatformType, ProcessorType]:
    """Detect OS platform and processor architecture.

    Returns:
        Tuple of (platform, processor)
    """
    # Map Python platform names to our types
    platform_map: dict[str, PlatformType] = {
        "darwin": "darwin",
        "linux": "linux",
        "win32": "windows",
    }

    # Detect processor architecture
    processor_type: ProcessorType
    proc = platform.processor().lower()
    if "arm" in proc or "aarch64" in proc:
        processor_type = "arm"
    elif "x86" in proc or "amd64" in proc or "intel" in proc or "i386" in proc or "i686" in proc:
        processor_type = "x86_64"
    else:
        processor_type = "unknown"

    return platform_map.get(sys.platform, "linux"), processor_type


def detect_system() -> SystemInfo:
    """Detect all system information.

    This is the main entry point for system detection.
    Combines RAM, GPU, and platform detection into a single SystemInfo object.

    Returns:
        SystemInfo with all detected system properties
    """
    total_ram, available_ram, adjusted_ram = detect_ram()
    has_cuda, has_metal = detect_gpu()
    platform_type, processor_type = detect_platform()

    return SystemInfo(
        total_ram_gb=total_ram,
        available_ram_gb=available_ram,
        adjusted_ram_gb=adjusted_ram,
        cpu_count=psutil.cpu_count(logical=True) or 4,
        platform=platform_type,
        processor=processor_type,
        has_cuda=has_cuda,
        has_metal=has_metal,
    )
