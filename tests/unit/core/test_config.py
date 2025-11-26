"""
Tests for configuration management (quirkllm/core/config.py).
"""

import pytest
import yaml

from quirkllm.core.config import (
    Config,
    ensure_config_directories,
    generate_default_config,
    get_config_value,
    load_config,
    merge_with_defaults,
    save_config,
    set_config_value,
)


class TestConfigDataclass:
    """Test Config dataclass initialization and defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()

        assert config.profile_override is None
        assert config.context_length is None
        assert config.quantization is None
        assert config.backend_type == "auto"
        assert config.model_path is None
        assert config.gpu_offload is True
        assert config.speculative_decoding is False
        assert config.enable_rag is True
        assert config.rag_cache_size_mb == 500
        assert config.theme == "dark"
        assert config.enable_streaming is True
        assert config.log_level == "info"
        assert config.auto_save_sessions is True
        assert config.confirm_destructive is True
        assert config.auto_install_deps is False
        assert config.custom_settings == {}

    def test_custom_config(self):
        """Test custom configuration values."""
        config = Config(
            profile_override="beast",
            theme="light",
            log_level="debug",
            backend_type="llama-cpp",
            model_path="/custom/model.gguf",
            enable_rag=False,
            custom_settings={"custom_key": "custom_value"},
        )

        assert config.profile_override == "beast"
        assert config.theme == "light"
        assert config.log_level == "debug"
        assert config.backend_type == "llama-cpp"
        assert config.model_path == "/custom/model.gguf"
        assert config.enable_rag is False
        assert config.custom_settings == {"custom_key": "custom_value"}


class TestConfigDirectories:
    """Test configuration directory creation."""

    def test_ensure_config_directories(self, tmp_path, monkeypatch):
        """Test that all required directories are created."""
        # Use temporary directory
        test_config_dir = tmp_path / ".quirkllm"
        monkeypatch.setattr("quirkllm.core.config.CONFIG_DIR", test_config_dir)
        monkeypatch.setattr("quirkllm.core.config.LOGS_DIR", test_config_dir / "logs")
        monkeypatch.setattr("quirkllm.core.config.CACHE_DIR", test_config_dir / "cache")
        monkeypatch.setattr("quirkllm.core.config.PLANS_DIR", test_config_dir / "plans")
        monkeypatch.setattr("quirkllm.core.config.SESSIONS_DIR", test_config_dir / "sessions")

        ensure_config_directories()

        assert test_config_dir.exists()
        assert (test_config_dir / "logs").exists()
        assert (test_config_dir / "cache").exists()
        assert (test_config_dir / "plans").exists()
        assert (test_config_dir / "sessions").exists()


class TestGenerateDefaultConfig:
    """Test default configuration generation."""

    def test_generate_default_config(self):
        """Test that generated config matches Config() defaults."""
        config = generate_default_config()
        default = Config()

        assert config.profile_override == default.profile_override
        assert config.theme == default.theme
        assert config.log_level == default.log_level
        assert config.backend_type == default.backend_type
        assert config.enable_rag == default.enable_rag


class TestLoadConfig:
    """Test configuration loading from YAML."""

    def test_load_nonexistent_config(self, tmp_path):
        """Test loading config when file doesn't exist returns defaults."""
        config_file = tmp_path / "nonexistent.yaml"
        config = load_config(config_file)

        default = generate_default_config()
        assert config.theme == default.theme
        assert config.log_level == default.log_level

    def test_load_empty_config(self, tmp_path):
        """Test loading empty config file returns defaults."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        config = load_config(config_file)
        default = generate_default_config()
        assert config.theme == default.theme

    def test_load_valid_config(self, tmp_path):
        """Test loading valid configuration."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "profile_override": "power",
            "theme": "light",
            "log_level": "debug",
            "backend_type": "mlx",
            "enable_rag": False,
            "rag_cache_size_mb": 1000,
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)

        assert config.profile_override == "power"
        assert config.theme == "light"
        assert config.log_level == "debug"
        assert config.backend_type == "mlx"
        assert config.enable_rag is False
        assert config.rag_cache_size_mb == 1000

    def test_load_config_with_custom_settings(self, tmp_path):
        """Test loading config with custom settings."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "theme": "dark",
            "custom_settings": {"my_custom_key": "my_custom_value", "another_key": 42},
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)

        assert config.theme == "dark"
        assert config.custom_settings["my_custom_key"] == "my_custom_value"
        assert config.custom_settings["another_key"] == 42

    def test_load_invalid_profile(self, tmp_path):
        """Test loading config with invalid profile raises error."""
        config_file = tmp_path / "config.yaml"
        config_data = {"profile_override": "invalid_profile"}

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="Invalid profile_override"):
            load_config(config_file)

    def test_load_invalid_theme(self, tmp_path):
        """Test loading config with invalid theme raises error."""
        config_file = tmp_path / "config.yaml"
        config_data = {"theme": "invalid_theme"}

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="Invalid theme"):
            load_config(config_file)

    def test_load_invalid_log_level(self, tmp_path):
        """Test loading config with invalid log level raises error."""
        config_file = tmp_path / "config.yaml"
        config_data = {"log_level": "invalid_level"}

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="Invalid log_level"):
            load_config(config_file)

    def test_load_invalid_backend(self, tmp_path):
        """Test loading config with invalid backend raises error."""
        config_file = tmp_path / "config.yaml"
        config_data = {"backend_type": "invalid_backend"}

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="Invalid backend_type"):
            load_config(config_file)

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading config with invalid YAML raises error."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content: [[[")

        with pytest.raises(ValueError, match="Invalid YAML"):
            load_config(config_file)


class TestSaveConfig:
    """Test configuration saving to YAML."""

    def test_save_default_config(self, tmp_path):
        """Test saving default configuration."""
        config_file = tmp_path / "config.yaml"
        config = generate_default_config()

        save_config(config, config_file)

        assert config_file.exists()
        loaded = load_config(config_file)
        assert loaded.theme == config.theme
        assert loaded.log_level == config.log_level

    def test_save_custom_config(self, tmp_path):
        """Test saving custom configuration."""
        config_file = tmp_path / "config.yaml"
        config = Config(
            profile_override="beast",
            theme="light",
            log_level="debug",
            backend_type="llama-cpp",
            model_path="/custom/path",
            custom_settings={"custom_key": "custom_value"},
        )

        save_config(config, config_file)

        loaded = load_config(config_file)
        assert loaded.profile_override == "beast"
        assert loaded.theme == "light"
        assert loaded.log_level == "debug"
        assert loaded.backend_type == "llama-cpp"
        assert loaded.model_path == "/custom/path"
        assert loaded.custom_settings["custom_key"] == "custom_value"

    def test_save_load_roundtrip(self, tmp_path):
        """Test that save/load preserves all values."""
        config_file = tmp_path / "config.yaml"
        original = Config(
            profile_override="power",
            context_length=65536,
            quantization="Q8_0",
            backend_type="mlx",
            gpu_offload=False,
            enable_rag=False,
            rag_cache_size_mb=2000,
            theme="auto",
            log_level="warning",
            auto_install_deps=True,
            custom_settings={"key1": "value1", "key2": 123},
        )

        save_config(original, config_file)
        loaded = load_config(config_file)

        assert loaded.profile_override == original.profile_override
        assert loaded.context_length == original.context_length
        assert loaded.quantization == original.quantization
        assert loaded.backend_type == original.backend_type
        assert loaded.gpu_offload == original.gpu_offload
        assert loaded.enable_rag == original.enable_rag
        assert loaded.rag_cache_size_mb == original.rag_cache_size_mb
        assert loaded.theme == original.theme
        assert loaded.log_level == original.log_level
        assert loaded.auto_install_deps == original.auto_install_deps
        assert loaded.custom_settings == original.custom_settings


class TestMergeWithDefaults:
    """Test configuration merging with overrides."""

    def test_merge_no_overrides(self):
        """Test merge with no overrides returns same config."""
        config = Config(theme="dark", log_level="info")
        merged = merge_with_defaults(config)

        assert merged.theme == "dark"
        assert merged.log_level == "info"

    def test_merge_with_overrides(self):
        """Test merge with overrides updates values."""
        config = Config(theme="dark", log_level="info")
        merged = merge_with_defaults(config, theme="light", profile_override="beast")

        assert merged.theme == "light"
        assert merged.log_level == "info"  # Unchanged
        assert merged.profile_override == "beast"

    def test_merge_none_overrides_ignored(self):
        """Test that None overrides are ignored."""
        config = Config(theme="dark", log_level="info")
        merged = merge_with_defaults(config, theme=None, profile_override=None)

        assert merged.theme == "dark"
        assert merged.profile_override is None


class TestGetConfigValue:
    """Test getting configuration values."""

    def test_get_standard_attribute(self):
        """Test getting standard configuration attribute."""
        config = Config(theme="light", log_level="debug")

        assert get_config_value(config, "theme") == "light"
        assert get_config_value(config, "log_level") == "debug"

    def test_get_custom_setting(self):
        """Test getting custom configuration setting."""
        config = Config(custom_settings={"custom_key": "custom_value"})

        assert get_config_value(config, "custom_key") == "custom_value"

    def test_get_nonexistent_with_default(self):
        """Test getting nonexistent key returns default."""
        config = Config()

        assert get_config_value(config, "nonexistent", "default") == "default"
        assert get_config_value(config, "nonexistent") is None


class TestSetConfigValue:
    """Test setting configuration values."""

    def test_set_standard_attribute(self):
        """Test setting standard configuration attribute."""
        config = Config(theme="dark")
        updated = set_config_value(config, "theme", "light")

        assert updated.theme == "light"
        assert config.theme == "dark"  # Original unchanged

    def test_set_custom_setting(self):
        """Test setting custom configuration setting."""
        config = Config()
        updated = set_config_value(config, "custom_key", "custom_value")

        assert updated.custom_settings["custom_key"] == "custom_value"
        assert "custom_key" not in config.custom_settings  # Original unchanged

    def test_set_preserves_other_values(self):
        """Test that setting one value preserves others."""
        config = Config(theme="dark", log_level="info", custom_settings={"key1": "value1"})
        updated = set_config_value(config, "theme", "light")

        assert updated.theme == "light"
        assert updated.log_level == "info"
        assert updated.custom_settings["key1"] == "value1"
