"""
Integration tests for error scenarios and edge cases.

Focus on integration-level error handling that spans multiple components.
Unit-level edge cases are covered in unit tests.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml


class TestCLIErrorHandling:
    """Test CLI error handling in real usage scenarios."""

    def test_cli_with_invalid_profile_argument(self):
        """CLI should handle invalid --profile argument gracefully."""
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm", "--profile", "invalid_profile_name"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should exit with error
        assert result.returncode != 0
        # Error should be helpful
        output = result.stdout + result.stderr
        assert "invalid" in output.lower() or "profile" in output.lower()

    def test_cli_with_missing_config_file(self):
        """CLI should handle missing config file path."""
        nonexistent_path = "/tmp/definitely_does_not_exist_12345.yaml"

        result = subprocess.run(
            [sys.executable, "-m", "quirkllm", "--config", nonexistent_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should either create default or exit gracefully
        assert result.returncode in [0, 1]

    def test_cli_with_multiple_conflicting_options(self):
        """CLI should handle conflicting options sensibly."""
        # Create temp config with survival profile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"theme": "dark"}, f)
            config_path = f.name

        try:
            # CLI should handle conflicts
            result = subprocess.run(
                [sys.executable, "-m", "quirkllm", "--profile", "beast", "--config", config_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Should succeed with CLI flag taking precedence
            assert result.returncode == 0
        finally:
            os.unlink(config_path)

    def test_cli_with_invalid_command_sequence(self):
        """CLI should reject invalid command sequences."""
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm", "--invalid-flag", "--another-bad-flag"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should exit with error
        assert result.returncode != 0


class TestConfigFileErrorRecovery:
    """Test config file error handling and recovery."""

    def test_corrupted_yaml_file_produces_helpful_error(self, tmp_path: Path):
        """Corrupted YAML should produce helpful error message."""
        from quirkllm.core.config import load_config

        corrupted_file = tmp_path / "corrupted.yaml"
        corrupted_file.write_text("invalid: yaml: [[[unclosed")

        with pytest.raises(ValueError, match="Invalid YAML"):
            load_config(corrupted_file)

    def test_empty_config_file_uses_defaults(self, tmp_path: Path):
        """Empty config file should load with all defaults."""
        from quirkllm.core.config import load_config

        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        config = load_config(empty_file)

        # Should have reasonable defaults
        assert config is not None
        assert hasattr(config, "theme")
        assert hasattr(config, "log_level")

    def test_config_with_partial_fields_uses_defaults(self, tmp_path: Path):
        """Config with only some fields should use defaults for missing ones."""
        from quirkllm.core.config import load_config

        config_file = tmp_path / "partial.yaml"
        config_data = {
            "theme": "light",  # Custom value
            "enable_rag": False,  # Custom value
            # Other fields should use defaults
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Should load successfully, using defaults for missing fields
        config = load_config(config_file)
        assert config.theme == "light"  # Custom
        assert config.enable_rag is False  # Custom
        assert config.log_level == "info"  # Default

    def test_readonly_directory_write_fails_gracefully(self, tmp_path: Path):
        """Attempting to write to read-only directory should fail cleanly."""
        from quirkllm.core.config import Config, save_config

        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        config_file = readonly_dir / "config.yaml"
        config = Config()

        try:
            with pytest.raises(PermissionError):
                save_config(config, config_file)
        finally:
            readonly_dir.chmod(0o755)


class TestBackendErrorRecovery:
    """Test backend error handling (integration-level)."""

    def test_backend_generate_before_load_raises_helpful_error(self):
        """Generating before loading should provide clear error."""
        from quirkllm.backends.base import BackendType, GenerationParams, create_backend

        backend = create_backend(BackendType.MOCK)

        assert not backend.is_loaded()

        params = GenerationParams(prompt="test", max_tokens=50)

        with pytest.raises(RuntimeError, match="not loaded"):
            backend.generate(params)

    def test_backend_load_unload_multiple_times(self):
        """Backend should handle multiple load/unload cycles."""
        from quirkllm.backends.base import BackendType, create_backend

        backend = create_backend(BackendType.MOCK)

        backend.load_model("model1.bin")
        assert backend.is_loaded()
        backend.unload_model()
        assert not backend.is_loaded()

        backend.load_model("model2.bin")
        assert backend.is_loaded()
        backend.unload_model()
        assert not backend.is_loaded()

    def test_backend_types_are_implemented(self):
        """Backend types LLAMACPP and MLX are now implemented."""
        from quirkllm.backends.base import Backend, BackendType, create_backend

        # LLAMACPP backend is implemented
        llamacpp = create_backend(BackendType.LLAMACPP)
        assert isinstance(llamacpp, Backend)

        # MLX backend requires MLX to be installed
        try:
            mlx = create_backend(BackendType.MLX)
            assert isinstance(mlx, Backend)
        except RuntimeError as e:
            if "MLX not installed" in str(e):
                pytest.skip("MLX not installed")


class TestFileSystemEdgeCases:
    """Test file system edge cases."""

    def test_broken_symlink_handling(self, tmp_path: Path):
        """Accessing broken symlink should raise FileNotFoundError."""
        symlink = tmp_path / "broken_link"
        target = tmp_path / "nonexistent_target"

        symlink.symlink_to(target)

        with pytest.raises(FileNotFoundError):
            symlink.read_text()

    def test_directory_creation_in_readonly_parent(self, tmp_path: Path):
        """Creating directory in read-only parent should fail gracefully."""
        readonly_dir = tmp_path / "readonly_parent"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        try:
            with pytest.raises(PermissionError):
                (readonly_dir / "new_subdir").mkdir()
        finally:
            readonly_dir.chmod(0o755)


class TestEndToEndErrorScenarios:
    """Test complete workflows with errors."""

    def test_cli_startup_with_corrupted_config_directory(self):
        """CLI should handle corrupted config directory gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            bad_config_dir = f.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "quirkllm", "--config", bad_config_dir, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert result.returncode in [0, 1]
        finally:
            if os.path.exists(bad_config_dir):
                os.unlink(bad_config_dir)

    def test_cli_with_extremely_long_command(self):
        """CLI should handle extremely long input gracefully."""
        long_name = "a" * 10000

        result = subprocess.run(
            [sys.executable, "-m", "quirkllm", "--profile", long_name],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
