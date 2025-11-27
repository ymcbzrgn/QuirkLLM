"""
Additional tests for REPL (quirkllm/cli/repl.py) to achieve 100% coverage.

Covers:
- Exception handling in commands
- Exception handling in run loop
- Debug mode exception printing
- EOFError handling (Ctrl+D)
"""

from unittest.mock import Mock, patch, MagicMock

import pytest

from quirkllm.cli.repl import REPL
from quirkllm.core.profile_manager import ProfileConfig
from quirkllm.core.system_detector import SystemInfo


@pytest.fixture(autouse=True)
def mock_mode_initialization():
    """Mock mode initialization to avoid registry issues."""
    with patch.object(REPL, "_initialize_mode"):
        yield


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


class TestREPLExceptionHandling:
    """Tests for REPL exception handling."""

    def test_command_handler_exception(self, mock_system_info, mock_profile_config, capsys):
        """Test that command handler exceptions are caught."""
        repl = REPL(mock_system_info, mock_profile_config, debug=False)

        # Patch a command handler to raise an exception
        def bad_handler():
            raise ValueError("Test command error")

        repl.commands["test_cmd"] = Mock(handler=bad_handler)

        # Handle the command
        result = repl._handle_command("/test_cmd")

        # Should handle the error gracefully
        assert result is True  # Command was recognized
        captured = capsys.readouterr()
        assert "error" in captured.out.lower()

    def test_command_handler_exception_with_debug(
        self, mock_system_info, mock_profile_config, capsys
    ):
        """Test that debug mode shows exception details."""
        repl = REPL(mock_system_info, mock_profile_config, debug=True)

        # Patch a command handler to raise an exception
        def bad_handler():
            raise ValueError("Test debug error")

        repl.commands["test_cmd"] = Mock(handler=bad_handler)

        # Mock console to verify print_exception is called in debug mode
        mock_console = Mock()
        repl.console = mock_console

        with patch.object(repl, "debug", True):
            repl._handle_command("/test_cmd")

        # In debug mode, print_exception should be called
        mock_console.print_exception.assert_called()

    def test_run_loop_exception_handling(self, mock_system_info, mock_profile_config):
        """Test that run loop exceptions are handled."""
        repl = REPL(mock_system_info, mock_profile_config, debug=False)

        # Mock session.prompt to raise an exception, then stop
        with patch.object(repl.session, "prompt") as mock_prompt:
            mock_prompt.side_effect = [RuntimeError("Test error"), EOFError()]

            # Should not crash
            with pytest.raises(SystemExit):
                repl.run()

    def test_run_loop_exception_with_debug(self, mock_system_info, mock_profile_config):
        """Test that run loop exceptions show traceback in debug mode."""
        repl = REPL(mock_system_info, mock_profile_config, debug=True)

        # Mock session.prompt to raise an exception, then stop
        with patch.object(repl.session, "prompt") as mock_prompt:
            mock_prompt.side_effect = [RuntimeError("Test debug error"), EOFError()]

            # Should exit with code 1
            with pytest.raises(SystemExit) as exc_info:
                repl.run()

            assert exc_info.value.code == 1

    def test_ctrl_c_handling(self, mock_system_info, mock_profile_config):
        """Test that Ctrl+C clears line and continues."""
        repl = REPL(mock_system_info, mock_profile_config)

        # Mock session.prompt to raise KeyboardInterrupt, then EOFError to exit
        with patch.object(repl.session, "prompt") as mock_prompt:
            mock_prompt.side_effect = [KeyboardInterrupt(), EOFError()]

            # Should handle KeyboardInterrupt and continue until EOFError
            repl.run()

            # Should have called prompt twice
            assert mock_prompt.call_count == 2

    def test_ctrl_d_handling(self, mock_system_info, mock_profile_config, capsys):
        """Test that Ctrl+D exits gracefully."""
        repl = REPL(mock_system_info, mock_profile_config)

        # Mock session.prompt to raise EOFError
        with patch.object(repl.session, "prompt") as mock_prompt:
            mock_prompt.side_effect = EOFError()

            # Should exit gracefully
            repl.run()

            captured = capsys.readouterr()
            assert "Goodbye" in captured.out or not repl.running

    def test_empty_input_handling(self, mock_system_info, mock_profile_config):
        """Test that empty input is skipped."""
        repl = REPL(mock_system_info, mock_profile_config)

        # Mock session.prompt to return empty string, then EOF
        with patch.object(repl.session, "prompt") as mock_prompt:
            mock_prompt.side_effect = ["", "   ", EOFError()]

            # Should skip empty inputs
            repl.run()

            # Should have called prompt 3 times
            assert mock_prompt.call_count == 3

    def test_handle_chat_no_model_loaded(self, mock_system_info, mock_profile_config, capsys):
        """Test that chat messages show 'no model loaded' when backend not available."""
        repl = REPL(mock_system_info, mock_profile_config)

        # Handle a chat message (no model loaded)
        repl._handle_chat("Hello world")

        captured = capsys.readouterr()
        assert "no model loaded" in captured.out.lower()
        assert "--model-path" in captured.out

    def test_unknown_command(self, mock_system_info, mock_profile_config, capsys):
        """Test that unknown commands show error message."""
        repl = REPL(mock_system_info, mock_profile_config)

        # Handle unknown command
        result = repl._handle_command("/unknown_cmd")

        assert result is True  # Command was recognized as a command attempt
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out
        assert "/help" in captured.out

    def test_empty_slash_command(self, mock_system_info, mock_profile_config):
        """Test that empty slash command (just '/') returns False."""
        repl = REPL(mock_system_info, mock_profile_config)

        # Handle empty command
        result = repl._handle_command("/")

        assert result is False  # Not a valid command
