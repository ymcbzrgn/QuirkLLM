"""Tests for profile_manager module."""

import pytest

from quirkllm.core.profile_manager import PROFILES, ProfileType, select_profile
from quirkllm.core.system_detector import SystemInfo


class TestProfileSelection:
    """Tests for profile selection logic."""

    @pytest.mark.parametrize(
        "total_ram,available_ram,platform,expected_profile",
        [
            # macOS (darwin): Uses total RAM
            (16.0, 5.0, "darwin", ProfileType.COMFORT),  # 16 GB total → Comfort
            (32.0, 8.0, "darwin", ProfileType.POWER),  # 32 GB total → Power
            (64.0, 12.0, "darwin", ProfileType.BEAST),  # 64 GB total → Beast
            (6.0, 4.0, "darwin", ProfileType.SURVIVAL),  # 6 GB total → Survival
            # Linux: Uses available RAM
            (16.0, 5.0, "linux", ProfileType.SURVIVAL),  # 5 GB available → Survival
            (16.0, 12.0, "linux", ProfileType.COMFORT),  # 12 GB available → Comfort
            (32.0, 28.0, "linux", ProfileType.POWER),  # 28 GB available → Power
            (64.0, 52.0, "linux", ProfileType.BEAST),  # 52 GB available → Beast
            # Windows: Uses available RAM
            (16.0, 6.0, "windows", ProfileType.SURVIVAL),  # 6 GB available → Survival
            (16.0, 10.0, "windows", ProfileType.COMFORT),  # 10 GB available → Comfort
            (32.0, 30.0, "windows", ProfileType.POWER),  # 30 GB available → Power
            (64.0, 50.0, "windows", ProfileType.BEAST),  # 50 GB available → Beast
        ],
    )
    def test_platform_aware_selection(
        self,
        total_ram: float,
        available_ram: float,
        platform: str,
        expected_profile: ProfileType,
    ) -> None:
        """Test platform-aware profile selection."""
        system_info = SystemInfo(
            total_ram_gb=total_ram,
            available_ram_gb=available_ram,
            adjusted_ram_gb=available_ram,  # Simplified for test
            platform=platform,
            processor="arm",
            has_cuda=False,
            has_metal=(platform == "darwin"),
            cpu_count=8,
        )

        config = select_profile(system_info)
        assert ProfileType[config.name.upper()] == expected_profile

    def test_macos_16gb_comfort_mode(self) -> None:
        """Test real-world macOS scenario: 16 GB total, 5 GB available → Comfort."""
        system_info = SystemInfo(
            total_ram_gb=16.0,
            available_ram_gb=5.0,
            adjusted_ram_gb=3.0,  # After Metal reserve
            platform="darwin",
            processor="arm",
            has_cuda=False,
            has_metal=True,
            cpu_count=8,
        )

        config = select_profile(system_info)
        assert config.name == "Comfort"
        assert config.context_length == 32768

    def test_linux_16gb_low_available_survival(self) -> None:
        """Test Linux conservative approach: 16 GB total, 5 GB available → Survival."""
        system_info = SystemInfo(
            total_ram_gb=16.0,
            available_ram_gb=5.0,
            adjusted_ram_gb=5.0,
            platform="linux",
            processor="x86_64",
            has_cuda=False,
            has_metal=False,
            cpu_count=8,
        )

        config = select_profile(system_info)
        assert config.name == "Survival"

    def test_manual_override_survival(self) -> None:
        """Test manual override to Survival mode."""
        system_info = SystemInfo(
            total_ram_gb=64.0,
            available_ram_gb=32.0,
            adjusted_ram_gb=32.0,
            platform="linux",
            processor="x86_64",
            has_cuda=False,
            has_metal=False,
            cpu_count=16,
        )

        config = select_profile(system_info, override="survival")
        assert config.name == "Survival"
        assert config.context_length == 16384

    def test_manual_override_comfort(self) -> None:
        """Test manual override to Comfort mode."""
        system_info = SystemInfo(
            total_ram_gb=6.0,
            available_ram_gb=4.0,
            adjusted_ram_gb=4.0,
            platform="darwin",
            processor="arm",
            has_cuda=False,
            has_metal=True,
            cpu_count=4,
        )

        config = select_profile(system_info, override="comfort")
        assert config.name == "Comfort"
        assert config.context_length == 32768

    def test_manual_override_case_insensitive(self) -> None:
        """Test manual override is case-insensitive."""
        system_info = SystemInfo(
            total_ram_gb=16.0,
            available_ram_gb=10.0,
            adjusted_ram_gb=10.0,
            platform="linux",
            processor="x86_64",
            has_cuda=False,
            has_metal=False,
            cpu_count=8,
        )

        config1 = select_profile(system_info, override="POWER")
        config2 = select_profile(system_info, override="power")
        config3 = select_profile(system_info, override="PoWeR")

        assert config1.name == "Power"
        assert config2.name == "Power"
        assert config3.name == "Power"

    def test_invalid_override_raises_error(self) -> None:
        """Test invalid profile override raises ValueError."""
        system_info = SystemInfo(
            total_ram_gb=16.0,
            available_ram_gb=10.0,
            adjusted_ram_gb=10.0,
            platform="linux",
            processor="x86_64",
            has_cuda=False,
            has_metal=False,
            cpu_count=8,
        )

        with pytest.raises(ValueError, match="Invalid profile 'invalid'"):
            select_profile(system_info, override="invalid")

    def test_invalid_override_error_message(self) -> None:
        """Test error message contains valid options."""
        system_info = SystemInfo(
            total_ram_gb=16.0,
            available_ram_gb=10.0,
            adjusted_ram_gb=10.0,
            platform="linux",
            processor="x86_64",
            has_cuda=False,
            has_metal=False,
            cpu_count=8,
        )

        with pytest.raises(ValueError) as exc_info:
            select_profile(system_info, override="turbo")

        error_msg = str(exc_info.value)
        assert "survival" in error_msg.lower()
        assert "comfort" in error_msg.lower()
        assert "power" in error_msg.lower()
        assert "beast" in error_msg.lower()


class TestProfileConfigurations:
    """Tests for profile configurations matching README specs."""

    def test_survival_profile_specs(self) -> None:
        """Test Survival profile matches README specifications."""
        profile = PROFILES[ProfileType.SURVIVAL]

        assert profile.name == "Survival"
        assert profile.context_length == 16384
        assert profile.quantization == "Q4_K_M"
        assert profile.batch_size == 1
        assert profile.rag_cache_mb == 200
        assert profile.kv_cache_gb == 2
        assert profile.embedding_model == "small"
        assert profile.concurrent_ops == 1
        assert profile.compaction_mode == "aggressive"
        assert profile.model_loading == "lazy"
        assert profile.expected_speed_toks == 3

    def test_comfort_profile_specs(self) -> None:
        """Test Comfort profile matches README specifications."""
        profile = PROFILES[ProfileType.COMFORT]

        assert profile.name == "Comfort"
        assert profile.context_length == 32768
        assert profile.quantization == "Q4_K_M"
        assert profile.batch_size == 4
        assert profile.rag_cache_mb == 500
        assert profile.kv_cache_gb == 4
        assert profile.embedding_model == "base"
        assert profile.concurrent_ops == 2
        assert profile.compaction_mode == "smart"
        assert profile.model_loading == "hybrid"
        assert profile.expected_speed_toks == 5

    def test_power_profile_specs(self) -> None:
        """Test Power profile matches README specifications."""
        profile = PROFILES[ProfileType.POWER]

        assert profile.name == "Power"
        assert profile.context_length == 65536
        assert profile.quantization == "Q8_0"
        assert profile.batch_size == 8
        assert profile.rag_cache_mb == 2048
        assert profile.kv_cache_gb == 8
        assert profile.embedding_model == "large"
        assert profile.concurrent_ops == 4
        assert profile.compaction_mode == "relaxed"
        assert profile.model_loading == "eager"
        assert profile.expected_speed_toks == 8

    def test_beast_profile_specs(self) -> None:
        """Test Beast profile matches README specifications."""
        profile = PROFILES[ProfileType.BEAST]

        assert profile.name == "Beast"
        assert profile.context_length == 131072
        assert profile.quantization == "Q8_0"
        assert profile.batch_size == 16
        assert profile.rag_cache_mb == 8192
        assert profile.kv_cache_gb == 16
        assert profile.embedding_model == "large"
        assert profile.concurrent_ops == 8
        assert profile.compaction_mode == "minimal"
        assert profile.model_loading == "full"
        assert profile.expected_speed_toks == 12

    def test_all_profiles_present(self) -> None:
        """Test all four profiles are defined."""
        assert len(PROFILES) == 4
        assert ProfileType.SURVIVAL in PROFILES
        assert ProfileType.COMFORT in PROFILES
        assert ProfileType.POWER in PROFILES
        assert ProfileType.BEAST in PROFILES

    def test_profile_progression(self) -> None:
        """Test profiles progress in capability."""
        survival = PROFILES[ProfileType.SURVIVAL]
        comfort = PROFILES[ProfileType.COMFORT]
        power = PROFILES[ProfileType.POWER]
        beast = PROFILES[ProfileType.BEAST]

        # Context length progression
        assert survival.context_length < comfort.context_length
        assert comfort.context_length < power.context_length
        assert power.context_length < beast.context_length

        # Batch size progression
        assert survival.batch_size < comfort.batch_size
        assert comfort.batch_size < power.batch_size
        assert power.batch_size < beast.batch_size

        # RAM usage progression
        assert survival.kv_cache_gb < comfort.kv_cache_gb
        assert comfort.kv_cache_gb < power.kv_cache_gb
        assert power.kv_cache_gb < beast.kv_cache_gb
