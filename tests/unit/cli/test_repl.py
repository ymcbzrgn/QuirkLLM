"""
Tests for REPL mode integration.

Tests verify:
- Mode initialization from config
- Mode switching via /mode command
- Mode cycling via Shift+Tab
- Prompt indicator updates
- Config persistence
- Command handling
- Error scenarios
"""

from unittest.mock import Mock, patch, MagicMock, call
import pytest

from quirkllm.cli.repl import REPL, Command
from quirkllm.core.system_detector import SystemInfo
from quirkllm.core.profile_manager import ProfileConfig
from quirkllm.core.config import Config
from quirkllm.modes import ModeType


@pytest.fixture
def system_info():
    """Create test system info."""
    return SystemInfo(
        platform="darwin",
        processor="arm64",
        cpu_count=8,
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        adjusted_ram_gb=11.0,
        has_cuda=False,
        has_metal=True,
    )


@pytest.fixture
def profile_config():
    """Create test profile config."""
    return ProfileConfig(
        name="Comfort",
        context_length=32000,
        quantization="Q4_K_M",
        batch_size=4,
        rag_cache_mb=500,
        kv_cache_gb=1.5,
        embedding_model="nomic-embed-text",
        concurrent_ops=2,
        compaction_mode="Smart",
        model_loading="Optimized",
        expected_speed_toks=25,
    )


@pytest.fixture
def config():
    """Create test config."""
    return Config(
        mode="chat",
        profile_override="comfort",
    )


class TestREPLInitialization:
    """Test REPL initialization and setup."""
    
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_creates_repl(self, mock_load_config, mock_get_registry, system_info, profile_config, config):
        """Test basic REPL creation."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        
        assert repl is not None
        assert repl.system_info == system_info
        assert repl.profile_config == profile_config
        assert repl.config == config
    
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_initializes_mode_from_config(self, mock_load_config, mock_get_registry, system_info, profile_config, config):
        """Test mode is initialized from config."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        mock_mode = Mock()
        mock_registry.create_mode.return_value = mock_mode
        mock_get_registry.return_value = mock_registry
        
        config.mode = "yami"
        repl = REPL(system_info, profile_config, config=config)
        
        # Verify mode was created and activated
        mock_registry.create_mode.assert_called_once_with(ModeType.YAMI, config)
        mock_mode.activate.assert_called_once()
        assert repl.current_mode == mock_mode
    
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_initializes_key_bindings(self, mock_load_config, mock_get_registry, system_info, profile_config, config):
        """Test key bindings are created."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        
        assert repl.key_bindings is not None
    
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_registers_mode_command(self, mock_load_config, mock_get_registry, system_info, profile_config, config):
        """Test /mode command is registered."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        
        assert "mode" in repl.commands
        assert repl.commands["mode"].name == "mode"


class TestModeSwitching:
    """Test mode switching functionality."""
    
    @patch("quirkllm.cli.repl.save_config")
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_switches_mode(self, mock_load_config, mock_get_registry, mock_save_config, system_info, profile_config, config):
        """Test switching to a different mode."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        
        # Create mock modes
        mock_chat_mode = Mock()
        mock_chat_mode.mode_type = ModeType.CHAT
        mock_yami_mode = Mock()
        mock_yami_mode.mode_type = ModeType.YAMI
        
        # Configure registry to return different modes
        mock_registry.create_mode.side_effect = [mock_chat_mode, mock_yami_mode]
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        
        # Switch to YAMI mode
        result = repl.switch_mode(ModeType.YAMI)
        
        assert result is True
        assert repl.current_mode == mock_yami_mode
        mock_chat_mode.deactivate.assert_called_once()
        mock_yami_mode.activate.assert_called_once()
        mock_save_config.assert_called()
    
    @patch("quirkllm.cli.repl.save_config")
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_persists_mode_to_config(self, mock_load_config, mock_get_registry, mock_save_config, system_info, profile_config, config):
        """Test mode is persisted to config."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        
        mock_chat_mode = Mock()
        mock_plan_mode = Mock()
        mock_registry.create_mode.side_effect = [mock_chat_mode, mock_plan_mode]
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        repl.switch_mode(ModeType.PLAN)
        
        # Verify config was updated and saved
        assert repl.config.mode == "plan"
        mock_save_config.assert_called_with(config)
    
    @patch("quirkllm.cli.repl.Console")
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_cycles_through_modes(self, mock_load_config, mock_get_registry, mock_console_class, system_info, profile_config, config):
        """Test mode cycling: chat -> yami -> plan -> ghost -> chat."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        
        # Create mock modes for cycle
        modes = []
        for mode_type in [ModeType.CHAT, ModeType.YAMI, ModeType.PLAN, ModeType.GHOST, ModeType.CHAT]:
            mock_mode = Mock()
            mock_mode.mode_type = mode_type
            mock_mode.get_prompt_indicator.return_value = "🔄"
            modes.append(mock_mode)
        
        mock_registry.create_mode.side_effect = modes
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        
        # Cycle through modes
        repl._cycle_mode()  # chat -> yami
        assert repl.current_mode.mode_type == ModeType.YAMI
        
        repl._cycle_mode()  # yami -> plan
        assert repl.current_mode.mode_type == ModeType.PLAN
        
        repl._cycle_mode()  # plan -> ghost
        assert repl.current_mode.mode_type == ModeType.GHOST
        
        repl._cycle_mode()  # ghost -> chat
        assert repl.current_mode.mode_type == ModeType.CHAT


class TestPromptIndicator:
    """Test prompt indicator updates."""
    
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_prompt_shows_mode_indicator(self, mock_load_config, mock_get_registry, system_info, profile_config, config):
        """Test prompt includes mode emoji."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        mock_mode = Mock()
        mock_mode.get_prompt_indicator.return_value = "🚀"
        mock_registry.create_mode.return_value = mock_mode
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        prompt = repl._get_prompt_text()
        
        assert "🚀" in prompt
        assert "quirk>" in prompt
    
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_prompt_updates_on_mode_change(self, mock_load_config, mock_get_registry, system_info, profile_config, config):
        """Test prompt indicator changes when mode changes."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        
        mock_chat_mode = Mock()
        mock_chat_mode.get_prompt_indicator.return_value = "🔄"
        mock_yami_mode = Mock()
        mock_yami_mode.get_prompt_indicator.return_value = "🚀"
        
        mock_registry.create_mode.side_effect = [mock_chat_mode, mock_yami_mode]
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        
        # Initial prompt
        prompt1 = repl._get_prompt_text()
        assert "🔄" in prompt1
        
        # Switch mode
        repl.switch_mode(ModeType.YAMI)
        
        # Updated prompt
        prompt2 = repl._get_prompt_text()
        assert "🚀" in prompt2


class TestModeCommand:
    """Test /mode command handling."""
    
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_mode_command_without_args_shows_current(self, mock_load_config, mock_get_registry, system_info, profile_config, config):
        """Test /mode without args shows current mode."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        mock_mode = Mock()
        mock_mode.mode_type = ModeType.CHAT
        mock_mode.get_prompt_indicator.return_value = "🔄"
        mock_registry.create_mode.return_value = mock_mode
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        
        # Mock console AFTER REPL creation
        mock_console = Mock()
        repl.console = mock_console
        
        repl._current_command_args = ""
        repl._cmd_mode()
        
        # Verify console.print was called (showing current mode)
        assert mock_console.print.called
    
    @patch("quirkllm.cli.repl.Console")
    @patch("quirkllm.cli.repl.save_config")
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_mode_command_with_valid_name_switches(self, mock_load_config, mock_get_registry, mock_save_config, mock_console_class, system_info, profile_config, config):
        """Test /mode <name> switches to that mode."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        
        mock_chat_mode = Mock()
        mock_plan_mode = Mock()
        mock_registry.create_mode.side_effect = [mock_chat_mode, mock_plan_mode]
        mock_get_registry.return_value = mock_registry
        
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        repl = REPL(system_info, profile_config, config=config)
        repl._current_command_args = "plan"
        repl._cmd_mode()
        
        # Verify mode was switched
        assert repl.current_mode == mock_plan_mode
        mock_plan_mode.activate.assert_called_once()
    
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_mode_command_with_invalid_name_shows_error(self, mock_load_config, mock_get_registry, system_info, profile_config, config):
        """Test /mode <invalid> shows error."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        mock_mode = Mock()
        mock_registry.create_mode.return_value = mock_mode
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        
        # Mock console AFTER REPL creation
        mock_console = Mock()
        repl.console = mock_console
        
        repl._current_command_args = "invalid_mode"
        repl._cmd_mode()
        
        # Verify error message was printed
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Invalid mode" in str(call) or "invalid" in str(call).lower() for call in print_calls)
    
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_mode_command_already_in_mode_warns(self, mock_load_config, mock_get_registry, system_info, profile_config, config):
        """Test switching to current mode shows warning."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        mock_mode = Mock()
        mock_mode.mode_type = ModeType.CHAT
        mock_registry.create_mode.return_value = mock_mode
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        
        # Mock console AFTER REPL creation
        mock_console = Mock()
        repl.console = mock_console
        
        repl._current_command_args = "chat"
        repl._cmd_mode()
        
        # Verify warning was printed
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Already in" in str(call) for call in print_calls)


class TestCommandHandling:
    """Test command parsing and handling."""
    
    @patch("quirkllm.cli.repl.save_config")
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_handles_mode_command_with_args(self, mock_load_config, mock_get_registry, mock_save_config, system_info, profile_config, config):
        """Test command args are passed correctly."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        
        mock_chat_mode = Mock()
        mock_chat_mode.mode_type = ModeType.CHAT
        
        mock_yami_mode = Mock()
        mock_yami_mode.mode_type = ModeType.YAMI
        mock_yami_mode.get_prompt_indicator.return_value = "🚀"
        
        mock_registry.create_mode.side_effect = [mock_chat_mode, mock_yami_mode]
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        
        # Mock console AFTER REPL creation
        mock_console = Mock()
        repl.console = mock_console
        
        # Handle command with args (should switch to YAMI mode)
        result = repl._handle_command("/mode yami")
        
        assert result is True
        # Mode should have been switched (check current mode type)
        assert repl.current_mode.mode_type == ModeType.YAMI
    
    @patch("quirkllm.cli.repl.Console")
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_non_commands_return_false(self, mock_load_config, mock_get_registry, mock_console_class, system_info, profile_config, config):
        """Test non-command input returns False."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        mock_mode = Mock()
        mock_registry.create_mode.return_value = mock_mode
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        
        result = repl._handle_command("hello world")
        
        assert result is False


class TestModeActivationDeactivation:
    """Test mode activation and deactivation."""
    
    @patch("quirkllm.cli.repl.save_config")
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_deactivates_old_mode_on_switch(self, mock_load_config, mock_get_registry, mock_save_config, system_info, profile_config, config):
        """Test old mode is deactivated when switching."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        
        mock_chat_mode = Mock()
        mock_yami_mode = Mock()
        mock_registry.create_mode.side_effect = [mock_chat_mode, mock_yami_mode]
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        repl.switch_mode(ModeType.YAMI)
        
        # Verify old mode was deactivated
        mock_chat_mode.deactivate.assert_called_once()
    
    @patch("quirkllm.cli.repl.save_config")
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_activates_new_mode_on_switch(self, mock_load_config, mock_get_registry, mock_save_config, system_info, profile_config, config):
        """Test new mode is activated when switching."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        
        mock_chat_mode = Mock()
        mock_plan_mode = Mock()
        mock_registry.create_mode.side_effect = [mock_chat_mode, mock_plan_mode]
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        repl.switch_mode(ModeType.PLAN)
        
        # Verify new mode was activated
        mock_plan_mode.activate.assert_called_once()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_handles_mode_creation_failure(self, mock_load_config, mock_get_registry, system_info, profile_config, config):
        """Test handling when mode creation fails."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        
        mock_chat_mode = Mock()
        mock_registry.create_mode.side_effect = [mock_chat_mode, Exception("Mode creation failed")]
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        
        # Mock console AFTER REPL creation
        mock_console = Mock()
        repl.console = mock_console
        
        result = repl.switch_mode(ModeType.YAMI)
        
        assert result is False
        # Verify error was printed
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("Failed" in str(call) or "failed" in str(call).lower() for call in print_calls)
    
    @patch("quirkllm.cli.repl.get_registry")
    @patch("quirkllm.cli.repl.load_config")
    def test_handles_no_current_mode(self, mock_load_config, mock_get_registry, system_info, profile_config, config):
        """Test handling when no mode is active."""
        mock_load_config.return_value = config
        mock_registry = Mock()
        mock_registry.create_mode.return_value = None
        mock_get_registry.return_value = mock_registry
        
        repl = REPL(system_info, profile_config, config=config)
        
        # Should handle gracefully
        prompt = repl._get_prompt_text()
        assert "quirk>" in prompt
