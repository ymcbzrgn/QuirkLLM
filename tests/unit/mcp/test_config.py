"""
Tests for MCP Configuration.

Test Categories:
1. Config Path Tests (2)
2. Config Generation Tests (2)
3. Installation Tests (1)

Total: 5 tests
"""

import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import pytest

from quirkllm.mcp.config import (
    get_claude_config_path,
    generate_mcp_config,
    install_config,
    uninstall_config,
    check_installation,
    load_existing_config,
    get_config_info,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# 1. Config Path Tests (2)
# =============================================================================


class TestConfigPath:
    """Tests for config path detection."""

    def test_get_claude_config_path_macos(self):
        """Test config path on macOS."""
        with patch("quirkllm.mcp.config.sys.platform", "darwin"):
            path = get_claude_config_path()

        assert "Library" in str(path)
        assert "Application Support" in str(path)
        assert "Claude" in str(path)
        assert path.name == "claude_desktop_config.json"

    def test_get_claude_config_path_windows(self):
        """Test config path on Windows."""
        with patch("quirkllm.mcp.config.sys.platform", "win32"):
            path = get_claude_config_path()

        assert "AppData" in str(path)
        assert "Roaming" in str(path)
        assert "Claude" in str(path)

    def test_get_claude_config_path_linux(self):
        """Test config path on Linux."""
        with patch("quirkllm.mcp.config.sys.platform", "linux"):
            path = get_claude_config_path()

        assert ".config" in str(path)
        assert "Claude" in str(path)


# =============================================================================
# 2. Config Generation Tests (2)
# =============================================================================


class TestConfigGeneration:
    """Tests for config generation."""

    def test_generate_mcp_config(self):
        """Test generating MCP config."""
        config = generate_mcp_config()

        assert "mcpServers" in config
        assert "quirkllm" in config["mcpServers"]
        assert "command" in config["mcpServers"]["quirkllm"]
        assert "args" in config["mcpServers"]["quirkllm"]
        assert "--mcp" in config["mcpServers"]["quirkllm"]["args"]

    def test_generate_mcp_config_has_env(self):
        """Test generated config has empty env dict."""
        config = generate_mcp_config()

        assert "env" in config["mcpServers"]["quirkllm"]
        assert isinstance(config["mcpServers"]["quirkllm"]["env"], dict)


# =============================================================================
# 3. Installation Tests (3)
# =============================================================================


class TestInstallation:
    """Tests for config installation."""

    def test_install_config_new(self, temp_config_dir):
        """Test installing config to new location."""
        config_path = temp_config_dir / "claude_desktop_config.json"

        with patch("quirkllm.mcp.config.get_claude_config_path", return_value=config_path):
            result = install_config()

        assert result == config_path
        assert config_path.exists()

        with open(config_path) as f:
            config = json.load(f)

        assert "quirkllm" in config["mcpServers"]

    def test_install_config_merge(self, temp_config_dir):
        """Test merging config with existing."""
        config_path = temp_config_dir / "claude_desktop_config.json"

        # Create existing config
        existing = {
            "mcpServers": {
                "other-server": {"command": "other"},
            }
        }
        with open(config_path, "w") as f:
            json.dump(existing, f)

        with patch("quirkllm.mcp.config.get_claude_config_path", return_value=config_path):
            install_config(merge=True)

        with open(config_path) as f:
            config = json.load(f)

        # Both servers should exist
        assert "quirkllm" in config["mcpServers"]
        assert "other-server" in config["mcpServers"]

    def test_uninstall_config(self, temp_config_dir):
        """Test uninstalling config."""
        config_path = temp_config_dir / "claude_desktop_config.json"

        # Create config with quirkllm
        config = {
            "mcpServers": {
                "quirkllm": {"command": "quirkllm", "args": ["--mcp"]},
                "other": {"command": "other"},
            }
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        with patch("quirkllm.mcp.config.get_claude_config_path", return_value=config_path):
            result = uninstall_config()

        assert result is True

        with open(config_path) as f:
            updated = json.load(f)

        assert "quirkllm" not in updated["mcpServers"]
        assert "other" in updated["mcpServers"]


# =============================================================================
# Additional Tests
# =============================================================================


class TestCheckInstallation:
    """Tests for installation checking."""

    def test_check_installation_not_installed(self, temp_config_dir):
        """Test check when not installed."""
        config_path = temp_config_dir / "claude_desktop_config.json"

        with patch("quirkllm.mcp.config.get_claude_config_path", return_value=config_path):
            result = check_installation()

        assert result["installed"] is False
        assert len(result["errors"]) > 0

    def test_check_installation_installed(self, temp_config_dir):
        """Test check when installed."""
        config_path = temp_config_dir / "claude_desktop_config.json"

        # Create config
        config = {
            "mcpServers": {
                "quirkllm": {"command": "quirkllm", "args": ["--mcp"]},
            }
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f)

        with patch("quirkllm.mcp.config.get_claude_config_path", return_value=config_path):
            result = check_installation()

        assert result["installed"] is True
        assert len(result["errors"]) == 0


class TestLoadExistingConfig:
    """Tests for loading existing config."""

    def test_load_existing_valid(self, temp_config_dir):
        """Test loading valid existing config."""
        config_path = temp_config_dir / "config.json"
        config_path.write_text('{"key": "value"}')

        result = load_existing_config(config_path)

        assert result == {"key": "value"}

    def test_load_existing_invalid_json(self, temp_config_dir):
        """Test loading invalid JSON returns empty dict."""
        config_path = temp_config_dir / "config.json"
        config_path.write_text("not valid json")

        result = load_existing_config(config_path)

        assert result == {}

    def test_load_existing_not_found(self, temp_config_dir):
        """Test loading non-existent file returns empty dict."""
        config_path = temp_config_dir / "nonexistent.json"

        result = load_existing_config(config_path)

        assert result == {}


class TestGetConfigInfo:
    """Tests for config info retrieval."""

    def test_get_config_info(self, temp_config_dir):
        """Test getting config info."""
        config_path = temp_config_dir / "claude_desktop_config.json"

        with patch("quirkllm.mcp.config.get_claude_config_path", return_value=config_path):
            info = get_config_info()

        assert "config_path" in info
        assert "platform" in info
        assert "quirkllm_path" in info
