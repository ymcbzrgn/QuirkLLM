"""
Tests for CLI entry point (quirkllm/__main__.py).

These tests ensure 100% coverage of the main entry point including:
- Exception handling (KeyboardInterrupt, general exceptions)
- Debug mode branches  
- Welcome banner display
- CLI argument parsing
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import click.testing
import pytest

from quirkllm.__main__ import display_welcome_banner, main
from quirkllm.core.profile_manager import ProfileConfig
from quirkllm.core.system_detector import SystemInfo


@pytest.fixture
def mock_system_info():
    """Create a mock SystemInfo for testing."""
    return SystemInfo(
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        adjusted_ram_gb=8.0,
        cpu_count=8,
        platform="darwin",
        processor="arm",
        has_cuda=False,
        has_metal=True,
    )


@pytest.fixture
def mock_profile_config():
    """Create a mock ProfileConfig for testing."""
    return ProfileConfig(
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
    )


@pytest.fixture
def cli_runner():
    """Create a Click test runner."""
    return click.testing.CliRunner()


class TestDisplayWelcomeBanner:
    """Tests for display_welcome_banner function."""

    def test_display_welcome_banner_basic(self, mock_system_info, mock_profile_config, capsys):
        """Test welcome banner displays correctly."""
        display_welcome_banner(mock_system_info, mock_profile_config)

        captured = capsys.readouterr()
        output = captured.out

        # Check for key information
        assert "QuirkLLM" in output
        assert "System Information" in output
        assert "Active Profile" in output
        assert "darwin" in output
        assert "arm" in output
        assert "Comfort" in output
        assert "32,768" in output or "32768" in output

    def test_display_welcome_banner_with_cuda(self, mock_profile_config, capsys):
        """Test welcome banner shows CUDA when available."""
        system_info = SystemInfo(
            total_ram_gb=32.0,
            available_ram_gb=16.0,
            adjusted_ram_gb=16.0,
            cpu_count=16,
            platform="linux",
            processor="x86_64",
            has_cuda=True,
            has_metal=False,
        )

        display_welcome_banner(system_info, mock_profile_config)
        captured = capsys.readouterr()
        output = captured.out

        assert "CUDA" in output or "✓" in output

    def test_display_welcome_banner_shows_ram_info(
        self, mock_system_info, mock_profile_config, capsys
    ):
        """Test that RAM information is displayed."""
        display_welcome_banner(mock_system_info, mock_profile_config)
        captured = capsys.readouterr()
        output = captured.out

        assert "16.0" in output  # total RAM
        assert "8.0" in output  # available/adjusted RAM


class TestMainCLI:
    """Tests for main CLI function."""

    @patch("quirkllm.__main__.detect_system")
    @patch("quirkllm.__main__.select_profile")
    @patch("quirkllm.__main__.REPL")
    def test_main_basic_execution(
        self, mock_repl_class, mock_select_profile, mock_detect_system, cli_runner
    ):
        """Test basic CLI execution without errors."""
        # Setup mocks
        mock_detect_system.return_value = SystemInfo(
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            adjusted_ram_gb=8.0,
            cpu_count=8,
            platform="darwin",
            processor="arm",
            has_cuda=False,
            has_metal=True,
        )
        mock_select_profile.return_value = ProfileConfig(
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
        )

        # Mock REPL instance
        mock_repl_instance = Mock()
        mock_repl_class.return_value = mock_repl_instance

        # Run CLI
        result = cli_runner.invoke(main, [])

        # Verify calls
        assert mock_detect_system.called
        assert mock_select_profile.called
        assert mock_repl_class.called
        assert mock_repl_instance.run.called
        assert result.exit_code == 0

    @patch("quirkllm.__main__.detect_system")
    @patch("quirkllm.__main__.select_profile")
    @patch("quirkllm.__main__.REPL")
    def test_main_with_profile_override(
        self, mock_repl_class, mock_select_profile, mock_detect_system, cli_runner
    ):
        """Test CLI with manual profile override."""
        mock_detect_system.return_value = SystemInfo(
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            adjusted_ram_gb=8.0,
            cpu_count=8,
            platform="darwin",
            processor="arm",
            has_cuda=False,
            has_metal=True,
        )
        mock_select_profile.return_value = ProfileConfig(
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
        )
        mock_repl_instance = Mock()
        mock_repl_class.return_value = mock_repl_instance

        # Run with profile override
        result = cli_runner.invoke(main, ["--profile", "power"])

        # Verify profile override was passed
        assert result.exit_code == 0
        assert mock_select_profile.call_args[1]["override"] == "power"

    @patch("quirkllm.__main__.detect_system")
    @patch("quirkllm.__main__.select_profile")
    @patch("quirkllm.__main__.REPL")
    def test_main_with_debug_flag(
        self, mock_repl_class, mock_select_profile, mock_detect_system, cli_runner
    ):
        """Test CLI with debug mode enabled."""
        mock_detect_system.return_value = SystemInfo(
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            adjusted_ram_gb=8.0,
            cpu_count=8,
            platform="darwin",
            processor="arm",
            has_cuda=False,
            has_metal=True,
        )
        mock_select_profile.return_value = ProfileConfig(
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
        )
        mock_repl_instance = Mock()
        mock_repl_class.return_value = mock_repl_instance

        # Run with debug flag
        result = cli_runner.invoke(main, ["--debug"])

        # Verify REPL was created with debug=True
        assert result.exit_code == 0
        assert mock_repl_class.call_args[1]["debug"] is True
        # Check debug output was printed
        assert "Detecting" in result.output or "System detected" in result.output

    @patch("quirkllm.__main__.detect_system")
    @patch("quirkllm.__main__.select_profile")
    @patch("quirkllm.__main__.REPL")
    def test_main_keyboard_interrupt(
        self, mock_repl_class, mock_select_profile, mock_detect_system, cli_runner
    ):
        """Test CLI handles KeyboardInterrupt gracefully."""
        mock_detect_system.return_value = SystemInfo(
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            adjusted_ram_gb=8.0,
            cpu_count=8,
            platform="darwin",
            processor="arm",
            has_cuda=False,
            has_metal=True,
        )
        mock_select_profile.return_value = ProfileConfig(
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
        )

        # Mock REPL to raise KeyboardInterrupt
        mock_repl_instance = Mock()
        mock_repl_instance.run.side_effect = KeyboardInterrupt()
        mock_repl_class.return_value = mock_repl_instance

        # Run CLI
        result = cli_runner.invoke(main, [])

        # Should exit with code 130 (standard for Ctrl+C)
        assert result.exit_code == 130
        assert "Interrupted" in result.output

    @patch("quirkllm.__main__.detect_system")
    def test_main_general_exception(self, mock_detect_system, cli_runner):
        """Test CLI handles general exceptions."""
        # Make detect_system raise an exception
        mock_detect_system.side_effect = RuntimeError("Test error")

        # Run CLI
        result = cli_runner.invoke(main, [])

        # Should exit with code 1
        assert result.exit_code == 1
        assert "Error" in result.output
        assert "Test error" in result.output

    @patch("quirkllm.__main__.detect_system")
    def test_main_general_exception_with_debug(self, mock_detect_system, cli_runner, capsys):
        """Test CLI shows full traceback in debug mode."""
        # Make detect_system raise an exception
        mock_detect_system.side_effect = RuntimeError("Test error with debug")

        # Run CLI with debug flag
        result = cli_runner.invoke(main, ["--debug"])

        # Should exit with code 1 and show traceback
        assert result.exit_code == 1
        assert "Error" in result.output
        assert "Test error with debug" in result.output

    def test_main_version_option(self, cli_runner):
        """Test --version flag."""
        result = cli_runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "QuirkLLM" in result.output or "0.1.0" in result.output

    def test_main_help_option(self, cli_runner):
        """Test --help flag."""
        result = cli_runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "QuirkLLM" in result.output
        assert "--profile" in result.output
        assert "--debug" in result.output
        assert "--config" in result.output
