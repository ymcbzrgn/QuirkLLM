"""Tests for system_detector module."""

from unittest.mock import MagicMock, patch

import pytest

from quirkllm.core.system_detector import (
    detect_gpu,
    detect_platform,
    detect_ram,
    detect_system,
)


@pytest.fixture
def mock_psutil_8gb() -> MagicMock:
    """Mock psutil for 8GB total, 6GB available."""
    mock = MagicMock()
    mock.virtual_memory.return_value = MagicMock(
        total=8 * 1024**3,
        available=6 * 1024**3,
    )
    mock.swap_memory.return_value = MagicMock(percent=5.0)
    mock.cpu_count.return_value = 4
    return mock


@pytest.fixture
def mock_psutil_16gb() -> MagicMock:
    """Mock psutil for 16GB total, 12GB available."""
    mock = MagicMock()
    mock.virtual_memory.return_value = MagicMock(
        total=16 * 1024**3,
        available=12 * 1024**3,
    )
    mock.swap_memory.return_value = MagicMock(percent=3.0)
    mock.cpu_count.return_value = 8
    return mock


@pytest.fixture
def mock_psutil_32gb() -> MagicMock:
    """Mock psutil for 32GB total, 28GB available."""
    mock = MagicMock()
    mock.virtual_memory.return_value = MagicMock(
        total=32 * 1024**3,
        available=28 * 1024**3,
    )
    mock.swap_memory.return_value = MagicMock(percent=0.0)
    mock.cpu_count.return_value = 16
    return mock


@pytest.fixture
def mock_psutil_64gb() -> MagicMock:
    """Mock psutil for 64GB total, 60GB available."""
    mock = MagicMock()
    mock.virtual_memory.return_value = MagicMock(
        total=64 * 1024**3,
        available=60 * 1024**3,
    )
    mock.swap_memory.return_value = MagicMock(percent=0.0)
    mock.cpu_count.return_value = 32
    return mock


class TestDetectRam:
    """Tests for detect_ram function."""

    def test_basic_detection(self, mock_psutil_8gb: MagicMock) -> None:
        """Test basic RAM detection without adjustments."""
        with patch("quirkllm.core.system_detector.psutil", mock_psutil_8gb), patch(
            "quirkllm.core.system_detector.sys.platform", "linux"
        ):
            total, available, adjusted = detect_ram()

            assert total == pytest.approx(8.0, abs=0.1)
            assert available == pytest.approx(6.0, abs=0.1)
            # No adjustments on Linux with low swap
            assert adjusted == pytest.approx(6.0, abs=0.1)

    def test_macos_metal_adjustment(self, mock_psutil_16gb: MagicMock) -> None:
        """Test macOS Apple Silicon reserves 2GB for Metal."""
        with patch("quirkllm.core.system_detector.psutil", mock_psutil_16gb), patch(
            "quirkllm.core.system_detector.sys.platform", "darwin"
        ), patch("quirkllm.core.system_detector.platform.processor", return_value="arm"):
            total, available, adjusted = detect_ram()

            assert total == pytest.approx(16.0, abs=0.1)
            assert available == pytest.approx(12.0, abs=0.1)
            # 12GB - 2GB Metal reserve = 10GB
            assert adjusted == pytest.approx(10.0, abs=0.1)

    def test_linux_swap_penalty(self, mock_psutil_16gb: MagicMock) -> None:
        """Test Linux applies 20% penalty when swap is active."""
        # Override swap to be high
        mock_psutil_16gb.swap_memory.return_value = MagicMock(percent=15.0)

        with patch("quirkllm.core.system_detector.psutil", mock_psutil_16gb), patch(
            "quirkllm.core.system_detector.sys.platform", "linux"
        ):
            total, available, adjusted = detect_ram()

            assert total == pytest.approx(16.0, abs=0.1)
            assert available == pytest.approx(12.0, abs=0.1)
            # 12GB * 0.8 (20% penalty) = 9.6GB
            assert adjusted == pytest.approx(9.6, abs=0.1)

    def test_minimum_1gb_floor(self, mock_psutil_8gb: MagicMock) -> None:
        """Test adjusted RAM never goes below 1GB."""
        # Set very low available RAM
        mock_psutil_8gb.virtual_memory.return_value = MagicMock(
            total=4 * 1024**3,
            available=0.5 * 1024**3,  # Only 512MB available
        )

        with patch("quirkllm.core.system_detector.psutil", mock_psutil_8gb), patch(
            "quirkllm.core.system_detector.sys.platform", "darwin"
        ), patch("quirkllm.core.system_detector.platform.processor", return_value="arm"):
            total, available, adjusted = detect_ram()

            assert total == pytest.approx(4.0, abs=0.1)
            assert available == pytest.approx(0.5, abs=0.1)
            # Would be 0.5 - 2.0 = -1.5, but floored to 1.0
            assert adjusted == pytest.approx(1.0, abs=0.1)


class TestDetectGpu:
    """Tests for detect_gpu function."""

    def test_cuda_available(self) -> None:
        """Test CUDA detection when nvidia-smi succeeds."""
        mock_result = MagicMock(returncode=0)

        with patch("quirkllm.core.system_detector.subprocess.run", return_value=mock_result):
            has_cuda, has_metal = detect_gpu()

            assert has_cuda is True
            # Metal depends on platform, not testing here
            assert isinstance(has_metal, bool)

    def test_cuda_not_available(self) -> None:
        """Test CUDA detection when nvidia-smi fails."""
        mock_result = MagicMock(returncode=1)

        with patch("quirkllm.core.system_detector.subprocess.run", return_value=mock_result):
            has_cuda, has_metal = detect_gpu()

            assert has_cuda is False

    def test_cuda_not_installed(self) -> None:
        """Test CUDA detection when nvidia-smi not found."""
        with patch("quirkllm.core.system_detector.subprocess.run", side_effect=FileNotFoundError):
            has_cuda, has_metal = detect_gpu()

            assert has_cuda is False

    def test_metal_on_apple_silicon(self) -> None:
        """Test Metal detection on macOS Apple Silicon."""
        with patch("quirkllm.core.system_detector.sys.platform", "darwin"), patch(
            "quirkllm.core.system_detector.platform.processor", return_value="arm"
        ), patch("quirkllm.core.system_detector.subprocess.run", side_effect=FileNotFoundError):
            has_cuda, has_metal = detect_gpu()

            assert has_cuda is False
            assert has_metal is True

    def test_no_metal_on_intel_mac(self) -> None:
        """Test no Metal on Intel Mac."""
        with patch("quirkllm.core.system_detector.sys.platform", "darwin"), patch(
            "quirkllm.core.system_detector.platform.processor", return_value="x86_64"
        ), patch("quirkllm.core.system_detector.subprocess.run", side_effect=FileNotFoundError):
            has_cuda, has_metal = detect_gpu()

            assert has_cuda is False
            assert has_metal is False


class TestDetectPlatform:
    """Tests for detect_platform function."""

    def test_darwin_arm(self) -> None:
        """Test macOS Apple Silicon detection."""
        with patch("quirkllm.core.system_detector.sys.platform", "darwin"), patch(
            "quirkllm.core.system_detector.platform.processor", return_value="arm64"
        ):
            platform_type, processor_type = detect_platform()

            assert platform_type == "darwin"
            assert processor_type == "arm"

    def test_darwin_x86(self) -> None:
        """Test macOS Intel detection."""
        with patch("quirkllm.core.system_detector.sys.platform", "darwin"), patch(
            "quirkllm.core.system_detector.platform.processor", return_value="i386"
        ):
            platform_type, processor_type = detect_platform()

            assert platform_type == "darwin"
            assert processor_type == "x86_64"

    def test_linux_x86(self) -> None:
        """Test Linux x86_64 detection."""
        with patch("quirkllm.core.system_detector.sys.platform", "linux"), patch(
            "quirkllm.core.system_detector.platform.processor", return_value="x86_64"
        ):
            platform_type, processor_type = detect_platform()

            assert platform_type == "linux"
            assert processor_type == "x86_64"

    def test_windows(self) -> None:
        """Test Windows detection."""
        with patch("quirkllm.core.system_detector.sys.platform", "win32"), patch(
            "quirkllm.core.system_detector.platform.processor", return_value="AMD64"
        ):
            platform_type, processor_type = detect_platform()

            assert platform_type == "windows"
            assert processor_type == "x86_64"

    def test_unknown_processor(self) -> None:
        """Test unknown processor architecture."""
        with patch("quirkllm.core.system_detector.sys.platform", "linux"), patch(
            "quirkllm.core.system_detector.platform.processor", return_value="riscv"
        ):
            platform_type, processor_type = detect_platform()

            assert platform_type == "linux"
            assert processor_type == "unknown"


class TestDetectSystem:
    """Tests for detect_system integration."""

    def test_detect_system_integration(self, mock_psutil_16gb: MagicMock) -> None:
        """Test full system detection integration."""
        with patch("quirkllm.core.system_detector.psutil", mock_psutil_16gb), patch(
            "quirkllm.core.system_detector.sys.platform", "linux"
        ), patch("quirkllm.core.system_detector.platform.processor", return_value="x86_64"), patch(
            "quirkllm.core.system_detector.subprocess.run", side_effect=FileNotFoundError
        ):
            system_info = detect_system()

            # RAM
            assert system_info.total_ram_gb == pytest.approx(16.0, abs=0.1)
            assert system_info.available_ram_gb == pytest.approx(12.0, abs=0.1)
            assert system_info.adjusted_ram_gb == pytest.approx(12.0, abs=0.1)

            # CPU
            assert system_info.cpu_count == 8

            # Platform
            assert system_info.platform == "linux"
            assert system_info.processor == "x86_64"

            # GPU
            assert system_info.has_cuda is False
            assert system_info.has_metal is False
