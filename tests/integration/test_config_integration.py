"""
Integration tests for configuration persistence and management.

Tests config save/load cycles, override scenarios, and validation
to ensure configuration persists correctly across sessions.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from quirkllm.core.config import (
    Config,
    generate_default_config,
    load_config,
    merge_with_defaults,
    save_config,
)


class TestConfigSaveLoadCycle:
    """Test configuration save and load operations."""

    def test_save_and_load_default_config(self, tmp_path):
        """Test that default config can be saved and loaded."""
        config_file = tmp_path / "config.yaml"
        config = generate_default_config()

        # Save
        save_config(config, config_file)

        # Verify file exists
        assert config_file.exists()

        # Load
        loaded = load_config(config_file)

        # Verify all fields match
        assert loaded.theme == config.theme
        assert loaded.log_level == config.log_level
        assert loaded.backend_type == config.backend_type
        assert loaded.enable_rag == config.enable_rag

    def test_save_and_load_custom_config(self, tmp_path):
        """Test that custom config values persist."""
        config_file = tmp_path / "config.yaml"
        config = Config(
            profile_override="beast",
            theme="light",
            log_level="debug",
            backend_type="mlx",
            model_path="/custom/model.gguf",
            gpu_offload=False,
            enable_rag=False,
            rag_cache_size_mb=2000,
            custom_settings={"my_key": "my_value", "number": 42},
        )

        # Save
        save_config(config, config_file)

        # Load
        loaded = load_config(config_file)

        # Verify all custom fields match
        assert loaded.profile_override == "beast"
        assert loaded.theme == "light"
        assert loaded.log_level == "debug"
        assert loaded.backend_type == "mlx"
        assert loaded.model_path == "/custom/model.gguf"
        assert loaded.gpu_offload is False
        assert loaded.enable_rag is False
        assert loaded.rag_cache_size_mb == 2000
        assert loaded.custom_settings["my_key"] == "my_value"
        assert loaded.custom_settings["number"] == 42

    def test_multiple_save_load_cycles(self, tmp_path):
        """Test that config survives multiple save/load cycles."""
        config_file = tmp_path / "config.yaml"

        # Cycle 1
        config1 = Config(theme="dark", log_level="info")
        save_config(config1, config_file)
        loaded1 = load_config(config_file)
        assert loaded1.theme == "dark"
        assert loaded1.log_level == "info"

        # Cycle 2 - modify and save again
        config2 = Config(theme="light", log_level="debug")
        save_config(config2, config_file)
        loaded2 = load_config(config_file)
        assert loaded2.theme == "light"
        assert loaded2.log_level == "debug"

        # Cycle 3 - back to original
        save_config(config1, config_file)
        loaded3 = load_config(config_file)
        assert loaded3.theme == "dark"
        assert loaded3.log_level == "info"

    def test_load_nonexistent_file_returns_defaults(self, tmp_path):
        """Test that loading nonexistent file returns defaults."""
        nonexistent = tmp_path / "nonexistent.yaml"

        loaded = load_config(nonexistent)

        default = generate_default_config()
        assert loaded.theme == default.theme
        assert loaded.log_level == default.log_level
        assert loaded.backend_type == default.backend_type


class TestConfigValidation:
    """Test configuration validation during load."""

    def test_load_invalid_profile(self, tmp_path):
        """Test that invalid profile raises error."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"profile_override": "invalid_profile"}, f)

        with pytest.raises(ValueError, match="Invalid profile_override"):
            load_config(config_file)

    def test_load_invalid_theme(self, tmp_path):
        """Test that invalid theme raises error."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"theme": "invalid_theme"}, f)

        with pytest.raises(ValueError, match="Invalid theme"):
            load_config(config_file)

    def test_load_invalid_log_level(self, tmp_path):
        """Test that invalid log level raises error."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"log_level": "invalid_level"}, f)

        with pytest.raises(ValueError, match="Invalid log_level"):
            load_config(config_file)

    def test_load_invalid_backend(self, tmp_path):
        """Test that invalid backend raises error."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"backend_type": "invalid_backend"}, f)

        with pytest.raises(ValueError, match="Invalid backend_type"):
            load_config(config_file)

    def test_load_corrupted_yaml(self, tmp_path):
        """Test that corrupted YAML raises error."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content: [[[")

        with pytest.raises(ValueError, match="Invalid YAML"):
            load_config(config_file)

    def test_load_empty_file(self, tmp_path):
        """Test that empty config file loads defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        loaded = load_config(config_file)

        default = generate_default_config()
        assert loaded.theme == default.theme


class TestConfigOverrides:
    """Test configuration override mechanisms."""

    def test_merge_with_runtime_overrides(self):
        """Test merging config with runtime overrides."""
        base_config = Config(theme="dark", log_level="info")

        merged = merge_with_defaults(base_config, theme="light", log_level="debug")

        assert merged.theme == "light"
        assert merged.log_level == "debug"

    def test_merge_preserves_unmodified_values(self):
        """Test that merge preserves values not in overrides."""
        base_config = Config(
            theme="dark",
            log_level="info",
            backend_type="auto",
            enable_rag=True,
            rag_cache_size_mb=500,
        )

        merged = merge_with_defaults(base_config, theme="light")

        assert merged.theme == "light"  # Changed
        assert merged.log_level == "info"  # Preserved
        assert merged.backend_type == "auto"  # Preserved
        assert merged.enable_rag is True  # Preserved
        assert merged.rag_cache_size_mb == 500  # Preserved

    def test_merge_none_values_ignored(self):
        """Test that None override values are ignored."""
        base_config = Config(theme="dark", log_level="info")

        merged = merge_with_defaults(base_config, theme=None, log_level=None)

        assert merged.theme == "dark"  # Unchanged
        assert merged.log_level == "info"  # Unchanged

    def test_profile_override_persistence(self, tmp_path):
        """Test that profile override persists across save/load."""
        config_file = tmp_path / "config.yaml"

        # Set profile override
        config = Config(profile_override="power")
        save_config(config, config_file)

        # Load and verify
        loaded = load_config(config_file)
        assert loaded.profile_override == "power"

        # Merge with runtime override
        merged = merge_with_defaults(loaded, profile_override="beast")
        assert merged.profile_override == "beast"


class TestConfigCustomSettings:
    """Test custom settings functionality."""

    def test_custom_settings_persist(self, tmp_path):
        """Test that custom settings persist across save/load."""
        config_file = tmp_path / "config.yaml"

        config = Config(
            custom_settings={
                "my_plugin_enabled": True,
                "my_plugin_timeout": 30,
                "my_plugin_url": "https://example.com",
            }
        )

        save_config(config, config_file)
        loaded = load_config(config_file)

        assert loaded.custom_settings["my_plugin_enabled"] is True
        assert loaded.custom_settings["my_plugin_timeout"] == 30
        assert loaded.custom_settings["my_plugin_url"] == "https://example.com"

    def test_empty_custom_settings(self, tmp_path):
        """Test that empty custom settings work correctly."""
        config_file = tmp_path / "config.yaml"

        config = Config(custom_settings={})

        save_config(config, config_file)
        loaded = load_config(config_file)

        assert loaded.custom_settings == {}

    def test_custom_settings_with_nested_values(self, tmp_path):
        """Test custom settings with nested dictionaries."""
        config_file = tmp_path / "config.yaml"

        config = Config(
            custom_settings={
                "plugin": {
                    "enabled": True,
                    "settings": {"timeout": 30, "retries": 3},
                }
            }
        )

        save_config(config, config_file)
        loaded = load_config(config_file)

        assert loaded.custom_settings["plugin"]["enabled"] is True
        assert loaded.custom_settings["plugin"]["settings"]["timeout"] == 30


class TestConfigIntegrationScenarios:
    """Test real-world configuration scenarios."""

    def test_first_run_scenario(self, tmp_path):
        """Test first-run scenario: no config exists."""
        config_file = tmp_path / "first_run.yaml"

        # First run: no config exists
        loaded = load_config(config_file)

        # Should get defaults
        assert loaded.theme == "dark"
        assert loaded.log_level == "info"

        # User modifies and saves
        loaded_with_changes = Config(
            theme=loaded.theme,
            log_level="debug",  # User wants debug logging
            profile_override="power",  # User has powerful machine
        )
        save_config(loaded_with_changes, config_file)

        # Second run: config exists
        loaded_again = load_config(config_file)
        assert loaded_again.log_level == "debug"
        assert loaded_again.profile_override == "power"

    def test_config_migration_scenario(self, tmp_path):
        """Test config migration: old config + new defaults."""
        config_file = tmp_path / "old_config.yaml"

        # Simulate old config with fewer fields
        with open(config_file, "w") as f:
            yaml.dump(
                {
                    "theme": "light",
                    "log_level": "debug",
                    # Missing new fields
                },
                f,
            )

        # Load should fill in missing fields with defaults
        loaded = load_config(config_file)

        assert loaded.theme == "light"  # From old config
        assert loaded.log_level == "debug"  # From old config
        assert loaded.backend_type == "auto"  # Default for missing field
        assert loaded.enable_rag is True  # Default for missing field

    def test_user_override_workflow(self, tmp_path):
        """Test typical user override workflow."""
        config_file = tmp_path / "user_config.yaml"

        # Load config (first time, gets defaults)
        config = load_config(config_file)

        # User wants to override for this session
        session_config = merge_with_defaults(
            config, profile_override="beast", log_level="debug"
        )

        assert session_config.profile_override == "beast"
        assert session_config.log_level == "debug"

        # User decides to save these preferences
        save_config(session_config, config_file)

        # Next session loads saved preferences
        next_session = load_config(config_file)
        assert next_session.profile_override == "beast"
        assert next_session.log_level == "debug"

    def test_partial_config_update(self, tmp_path):
        """Test updating only some config values."""
        config_file = tmp_path / "partial.yaml"

        # Start with some config
        original = Config(
            theme="dark",
            log_level="info",
            backend_type="auto",
            enable_rag=True,
        )
        save_config(original, config_file)

        # Load and update only theme
        loaded = load_config(config_file)
        updated = merge_with_defaults(loaded, theme="light")

        # Verify only theme changed
        assert updated.theme == "light"  # Changed
        assert updated.log_level == "info"  # Same
        assert updated.backend_type == "auto"  # Same
        assert updated.enable_rag is True  # Same

        # Save updated config
        save_config(updated, config_file)

        # Verify persistence
        reloaded = load_config(config_file)
        assert reloaded.theme == "light"
        assert reloaded.log_level == "info"
