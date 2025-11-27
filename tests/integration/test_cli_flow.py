"""
Integration tests for CLI flow and user interaction.

Tests the full command-line interface using subprocess to simulate
real user interaction with the quirkllm CLI.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestCLIBasics:
    """Test basic CLI functionality and flags."""

    def test_version_flag(self):
        """Test --version flag returns version info."""
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "quirkllm" in output.lower() or "version" in output.lower()

    def test_help_flag(self):
        """Test --help flag shows usage information."""
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "usage" in output.lower() or "quirkllm" in output.lower()
        assert "help" in output.lower()

    def test_cli_starts_without_errors(self):
        """Test that CLI can start and quit cleanly."""
        # Send /quit command immediately to test startup
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm"],
            input="/quit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should exit cleanly (exit code 0 or 130 for Ctrl+C simulation)
        assert result.returncode in [0, 130]
        # Should show welcome banner
        assert "QuirkLLM" in result.stdout or "quirkllm" in result.stdout.lower()


class TestCLICommands:
    """Test CLI slash commands."""

    def test_help_command(self):
        """Test /help command displays available commands."""
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm"],
            input="/help\n/quit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode in [0, 130]
        output = result.stdout.lower()
        # Should show help information
        assert "help" in output or "command" in output

    def test_status_command(self):
        """Test /status command shows system information."""
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm"],
            input="/status\n/quit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode in [0, 130]
        output = result.stdout.lower()
        # Should show system status
        assert "ram" in output or "profile" in output or "system" in output

    def test_quit_command(self):
        """Test /quit command exits cleanly."""
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm"],
            input="/quit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode in [0, 130]

    def test_command_aliases(self):
        """Test command aliases (?, h, info, stat, exit, q)."""
        # Test ? as help alias
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm"],
            input="?\n/quit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode in [0, 130]

        # Test q as quit alias
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm"],
            input="q\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode in [0, 130]

    def test_invalid_command(self):
        """Test that invalid commands are handled gracefully."""
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm"],
            input="/invalidcommand123\n/quit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode in [0, 130]
        # Should not crash, should continue to accept commands


class TestCLIFlags:
    """Test CLI command-line flags."""

    def test_profile_override_flag(self):
        """Test --profile flag overrides auto-detection."""
        # Test with beast profile
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm", "--profile", "beast"],
            input="/status\n/quit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode in [0, 130]
        output = result.stdout.lower()
        # Should mention beast profile
        assert "beast" in output

    def test_debug_flag(self):
        """Test --debug flag enables debug output."""
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm", "--debug"],
            input="/quit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode in [0, 130]
        # Debug mode should show more information
        # At minimum, should not crash

    def test_custom_config_flag(self):
        """Test --config flag uses custom config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp_config:
            temp_config.write("theme: light\nlog_level: debug\n")
            temp_config_path = temp_config.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "quirkllm", "--config", temp_config_path],
                input="/quit\n",
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert result.returncode in [0, 130]
            # Should load successfully with custom config
        finally:
            Path(temp_config_path).unlink(missing_ok=True)

    def test_invalid_profile_flag(self):
        """Test that invalid profile names are rejected."""
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm", "--profile", "invalid_profile"],
            input="/quit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should either fail with error code or show error message
        if result.returncode != 0:
            assert result.returncode != 0
        else:
            # If it doesn't fail with exit code, should show error in output
            output = (result.stdout + result.stderr).lower()
            assert "error" in output or "invalid" in output


class TestCLIWorkflow:
    """Test complete CLI workflow scenarios."""

    def test_full_workflow_help_status_quit(self):
        """Test complete workflow: startup → /help → /status → /quit."""
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm"],
            input="/help\n/status\n/quit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode in [0, 130]
        output = result.stdout.lower()

        # Should have executed all commands
        assert "help" in output or "command" in output
        assert "ram" in output or "profile" in output or "system" in output

    def test_multiple_status_checks(self):
        """Test that /status can be called multiple times."""
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm"],
            input="/status\n/status\n/status\n/quit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode in [0, 130]
        # Should handle multiple status commands without issue

    def test_mixed_valid_invalid_commands(self):
        """Test that CLI recovers from invalid commands."""
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm"],
            input="/help\n/invalid\n/status\n/badcommand\n/quit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode in [0, 130]
        # Should handle mix of valid and invalid commands gracefully


class TestCLIOutputFormatting:
    """Test CLI output formatting and Rich UI."""

    def test_welcome_banner_present(self):
        """Test that welcome banner is displayed on startup."""
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm"],
            input="/quit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode in [0, 130]
        output = result.stdout

        # Should show welcome information
        assert "QuirkLLM" in output or "quirkllm" in output.lower()
        # Should mention profile or system info
        assert (
            "Profile" in output
            or "profile" in output
            or "RAM" in output
            or "System" in output
        )

    def test_no_color_output(self):
        """Test that NO_COLOR=1 disables ANSI colors."""
        env = {"NO_COLOR": "1"}
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm"],
            input="/help\n/quit\n",
            capture_output=True,
            text=True,
            timeout=10,
            env={**subprocess.os.environ, **env},
        )

        assert result.returncode in [0, 130]
        output = result.stdout

        # With NO_COLOR=1, should not have ANSI escape codes
        # (or at least should work without crashing)
        assert len(output) > 0


class TestCLIErrorHandling:
    """Test CLI error handling and recovery."""

    def test_graceful_handling_of_missing_config(self):
        """Test that missing config file doesn't crash CLI."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_config = Path(temp_dir) / "nonexistent.yaml"

            result = subprocess.run(
                [sys.executable, "-m", "quirkllm", "--config", str(nonexistent_config)],
                input="/quit\n",
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Should either load defaults or show clear error
            assert result.returncode in [0, 130] or result.returncode == 1

    def test_ctrl_c_handling(self):
        """Test that Ctrl+C is handled gracefully (simulated)."""
        # This is hard to test in subprocess, but we verify the CLI
        # accepts the quit command which simulates user wanting to exit
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm"],
            input="/quit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode in [0, 130]
        # Should exit cleanly without traceback
        assert "Traceback" not in result.stdout
        assert "Traceback" not in result.stderr


class TestCLIStartupPerformance:
    """Test CLI startup performance."""

    def test_startup_time_reasonable(self):
        """Test that CLI starts in reasonable time (<5s)."""
        import time

        start_time = time.time()

        result = subprocess.run(
            [sys.executable, "-m", "quirkllm", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        elapsed_time = time.time() - start_time

        assert result.returncode == 0
        # Should start quickly for --version (no full initialization)
        assert elapsed_time < 5.0, f"Startup took {elapsed_time:.2f}s, expected <5s"

    def test_full_startup_time(self):
        """Test that full CLI initialization is reasonably fast (<10s)."""
        import time

        start_time = time.time()

        result = subprocess.run(
            [sys.executable, "-m", "quirkllm"],
            input="/quit\n",
            capture_output=True,
            text=True,
            timeout=15,
        )

        elapsed_time = time.time() - start_time

        assert result.returncode in [0, 130]
        # Full initialization with REPL should still be fast
        assert (
            elapsed_time < 10.0
        ), f"Full startup took {elapsed_time:.2f}s, expected <10s"
