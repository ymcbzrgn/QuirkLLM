"""
Tests for Chat Mode.

Tests cover:
- Mode activation and deactivation
- Action confirmation flow
- Diff display for file edits
- Critical operation blocking
- "Always allow" functionality
- Session statistics
- Prompt indicator
- Edge cases and error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from quirkllm.modes.base import ModeType, ModeConfig, ActionRequest, ActionResult
from quirkllm.modes.chat_mode import ChatMode


@pytest.fixture
def chat_mode():
    """Create a ChatMode instance for testing."""
    mode = ChatMode(ModeType.CHAT)
    return mode


@pytest.fixture
def mock_console():
    """Create a mock Rich console."""
    with patch('quirkllm.modes.chat_mode.Console') as mock:
        console_instance = MagicMock()
        mock.return_value = console_instance
        yield console_instance


class TestChatModeInitialization:
    """Test ChatMode initialization."""
    
    def test_chat_mode_creation(self):
        """Test that ChatMode can be created."""
        mode = ChatMode(ModeType.CHAT)
        
        assert mode.mode_type == ModeType.CHAT
        assert not mode.is_active
        assert mode.always_allow == set()
        assert mode.session_stats == {
            "actions_confirmed": 0,
            "actions_rejected": 0,
            "actions_blocked": 0,
        }
    
    def test_chat_mode_default_config(self):
        """Test ChatMode uses safe default config."""
        mode = ChatMode(ModeType.CHAT)
        
        assert mode.config.auto_confirm is False
        assert mode.config.allow_file_edits is True
        assert mode.config.allow_destructive is False
        assert mode.config.confirm_high_risk is True
        assert mode.config.diff_display is True
    
    def test_chat_mode_custom_config(self):
        """Test ChatMode with custom config."""
        config = ModeConfig(diff_display=False)
        mode = ChatMode(ModeType.CHAT, config)
        
        assert mode.config.diff_display is False


class TestChatModeActivation:
    """Test Chat mode activation and deactivation."""
    
    def test_activate_sets_active_flag(self, chat_mode):
        """Test that activate sets is_active flag."""
        assert not chat_mode.is_active
        
        chat_mode.activate()
        
        assert chat_mode.is_active
    
    def test_activate_resets_session_state(self, chat_mode):
        """Test that activate resets session state."""
        # Pollute state
        chat_mode.always_allow.add("edit_file")
        chat_mode.session_stats["actions_confirmed"] = 5
        
        chat_mode.activate()
        
        assert chat_mode.always_allow == set()
        assert chat_mode.session_stats["actions_confirmed"] == 0
    
    def test_activate_displays_welcome_message(self, chat_mode):
        """Test that activate displays welcome panel."""
        with patch.object(chat_mode.console, 'print') as mock_print:
            chat_mode.activate()
            
            # Should print welcome panel
            mock_print.assert_called_once()
    
    def test_deactivate_clears_active_flag(self, chat_mode):
        """Test that deactivate clears is_active flag."""
        chat_mode.activate()
        assert chat_mode.is_active
        
        chat_mode.deactivate()
        
        assert not chat_mode.is_active
    
    def test_deactivate_displays_stats(self, chat_mode):
        """Test that deactivate displays session stats."""
        with patch.object(chat_mode.console, 'print') as mock_print:
            chat_mode.activate()
            chat_mode.session_stats["actions_confirmed"] = 3
            chat_mode.session_stats["actions_rejected"] = 1
            
            mock_print.reset_mock()
            chat_mode.deactivate()
            
            # Should print stats when there are actions
            mock_print.assert_called_once()


class TestChatModePromptIndicator:
    """Test Chat mode prompt indicator."""
    
    def test_get_prompt_indicator(self, chat_mode):
        """Test that prompt indicator is chat emoji."""
        assert chat_mode.get_prompt_indicator() == "💬"


class TestChatModeCriticalBlocking:
    """Test critical operation blocking."""
    
    def test_critical_operation_blocked(self, chat_mode):
        """Test that critical operations are blocked."""
        action = ActionRequest(
            action_type="delete_file",
            target="/important/file.py",
            risk_level="critical",
        )
        
        result = chat_mode.handle_action(action)
        
        assert not result.success
        # Either blocked or not allowed - both mean rejected
        assert not result.success
        assert chat_mode.session_stats["actions_blocked"] == 1
    
    def test_critical_operation_returns_error(self, chat_mode):
        """Test that critical operations return error result."""
        action = ActionRequest(
            action_type="run_command",
            target="rm -rf /",
            risk_level="critical",
        )
        
        result = chat_mode.handle_action(action)
        
        # Should return failed result
        assert not result.success
        assert len(result.errors) > 0


class TestChatModeConfirmationFlow:
    """Test action confirmation flow."""
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_action_confirmed_with_yes(self, mock_prompt, chat_mode):
        """Test action execution when user says yes."""
        mock_prompt.return_value = "y"
        
        action = ActionRequest(
            action_type="edit_file",
            target="test.py",
            risk_level="low",
        )
        
        result = chat_mode.handle_action(action)
        
        assert result.success
        assert chat_mode.session_stats["actions_confirmed"] == 1
        assert "edit_file" not in chat_mode.always_allow
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_action_rejected_with_no(self, mock_prompt, chat_mode):
        """Test action rejection when user says no."""
        mock_prompt.return_value = "n"
        
        action = ActionRequest(
            action_type="edit_file",
            target="test.py",
            risk_level="low",
        )
        
        result = chat_mode.handle_action(action)
        
        assert not result.success
        assert "rejected by user" in result.message
        assert chat_mode.session_stats["actions_rejected"] == 1
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_action_confirmed_with_always(self, mock_prompt, chat_mode):
        """Test action with "always allow" option."""
        mock_prompt.return_value = "a"
        
        action = ActionRequest(
            action_type="edit_file",
            target="test.py",
            risk_level="low",
        )
        
        result = chat_mode.handle_action(action)
        
        assert result.success
        assert "edit_file" in chat_mode.always_allow
        assert chat_mode.session_stats["actions_confirmed"] == 1
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_always_allow_skips_confirmation(self, mock_prompt, chat_mode):
        """Test that "always allow" skips future confirmations."""
        # First action with "always"
        mock_prompt.return_value = "a"
        action1 = ActionRequest(
            action_type="edit_file",
            target="test1.py",
            risk_level="low",
        )
        chat_mode.handle_action(action1)
        
        # Second action should not prompt
        mock_prompt.reset_mock()
        action2 = ActionRequest(
            action_type="edit_file",
            target="test2.py",
            risk_level="low",
        )
        result = chat_mode.handle_action(action2)
        
        assert result.success
        mock_prompt.assert_not_called()  # No prompt for second action
        assert chat_mode.session_stats["actions_confirmed"] == 2
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_view_option_redisplays_details(self, mock_prompt, chat_mode, mock_console):
        """Test that 'view' option redisplays action details."""
        chat_mode.console = mock_console
        # First return 'v' (view), then 'y' (yes)
        mock_prompt.side_effect = ["v", "y"]
        
        action = ActionRequest(
            action_type="edit_file",
            target="test.py",
            risk_level="low",
        )
        
        result = chat_mode.handle_action(action)
        
        assert result.success
        assert mock_prompt.call_count == 2
        # Should display details at least twice (initial + view)
        assert mock_console.print.call_count >= 2
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_quit_option_rejects_action(self, mock_prompt, chat_mode):
        """Test that 'quit' option rejects action."""
        mock_prompt.return_value = "q"
        
        action = ActionRequest(
            action_type="edit_file",
            target="test.py",
            risk_level="low",
        )
        
        result = chat_mode.handle_action(action)
        
        assert not result.success
        assert chat_mode.session_stats["actions_rejected"] == 1


class TestChatModeDiffDisplay:
    """Test diff display functionality."""
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_diff_displayed_for_file_edits(self, mock_prompt, chat_mode, mock_console):
        """Test that diff is displayed for file edit actions."""
        chat_mode.console = mock_console
        mock_prompt.return_value = "y"
        
        action = ActionRequest(
            action_type="edit_file",
            target="test.py",
            details={
                "old_content": "def foo():\n    pass\n",
                "new_content": "def foo():\n    return 42\n",
            },
            risk_level="low",
        )
        
        chat_mode.handle_action(action)
        
        # Should print diff
        assert mock_console.print.call_count >= 2  # Details + diff
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_no_diff_when_disabled(self, mock_prompt, chat_mode, mock_console):
        """Test that diff is not shown when disabled in config."""
        config = ModeConfig(diff_display=False)
        chat_mode = ChatMode(ModeType.CHAT, config)
        chat_mode.console = mock_console
        mock_prompt.return_value = "y"
        
        action = ActionRequest(
            action_type="edit_file",
            target="test.py",
            details={
                "old_content": "old",
                "new_content": "new",
            },
            risk_level="low",
        )
        
        chat_mode.handle_action(action)
        
        # Should only print action details, not diff
        assert mock_console.print.call_count == 1
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_no_diff_for_non_edit_actions(self, mock_prompt, chat_mode, mock_console):
        """Test that diff is not shown for non-edit actions."""
        chat_mode.console = mock_console
        mock_prompt.return_value = "y"
        
        action = ActionRequest(
            action_type="run_command",
            target="pytest",
            risk_level="safe",
        )
        
        chat_mode.handle_action(action)
        
        # Should only print action details
        assert mock_console.print.call_count == 1


class TestChatModeValidation:
    """Test action validation."""
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_high_risk_action_prompts(self, mock_prompt, chat_mode):
        """Test that high-risk (non-critical) actions still prompt."""
        mock_prompt.return_value = "y"
        
        # High risk but not critical - should prompt
        action = ActionRequest(
            action_type="delete_file",
            target="important.py",
            risk_level="high",  # High but not critical
        )
        
        result = chat_mode.handle_action(action)
        
        # Should prompt and execute if confirmed
        assert result.success
        mock_prompt.assert_called_once()


class TestChatModeSessionStats:
    """Test session statistics tracking."""
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_session_stats_tracking(self, mock_prompt, chat_mode):
        """Test that session stats are tracked correctly."""
        # Confirm one action
        mock_prompt.return_value = "y"
        action1 = ActionRequest(action_type="edit_file", target="test1.py")
        chat_mode.handle_action(action1)
        
        # Reject one action
        mock_prompt.return_value = "n"
        action2 = ActionRequest(action_type="edit_file", target="test2.py")
        chat_mode.handle_action(action2)
        
        # Block one critical action
        action3 = ActionRequest(
            action_type="delete_file",
            target="test3.py",
            risk_level="critical",
        )
        chat_mode.handle_action(action3)
        
        stats = chat_mode.get_session_stats()
        
        assert stats["actions_confirmed"] == 1
        assert stats["actions_rejected"] == 1
        assert stats["actions_blocked"] == 1
    
    def test_get_session_stats_returns_copy(self, chat_mode):
        """Test that get_session_stats returns a copy."""
        stats1 = chat_mode.get_session_stats()
        stats2 = chat_mode.get_session_stats()
        
        assert stats1 == stats2
        assert stats1 is not stats2
    
    def test_clear_always_allow(self, chat_mode):
        """Test clearing always allow set."""
        chat_mode.always_allow.add("edit_file")
        chat_mode.always_allow.add("create_file")
        
        assert len(chat_mode.always_allow) == 2
        
        chat_mode.clear_always_allow()
        
        assert len(chat_mode.always_allow) == 0


class TestChatModeActionExecution:
    """Test action execution (placeholder)."""
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_successful_action_execution(self, mock_prompt, chat_mode):
        """Test successful action execution."""
        mock_prompt.return_value = "y"
        
        action = ActionRequest(
            action_type="edit_file",
            target="test.py",
            risk_level="low",
        )
        
        result = chat_mode.handle_action(action)
        
        assert result.success
        assert "executed successfully" in result.message
        assert result.modified_files == ["test.py"]
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_action_result_contains_details(self, mock_prompt, chat_mode):
        """Test that action result contains execution details."""
        mock_prompt.return_value = "y"
        
        action = ActionRequest(
            action_type="run_command",
            target="pytest",
            risk_level="safe",
        )
        
        result = chat_mode.handle_action(action)
        
        assert result.success
        assert result.details["action_type"] == "run_command"
        assert result.details["target"] == "pytest"


class TestChatModeEdgeCases:
    """Test edge cases and error handling."""
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_empty_action_type(self, mock_prompt, chat_mode):
        """Test handling of action with empty type."""
        mock_prompt.return_value = "y"
        
        action = ActionRequest(
            action_type="",
            target="test.py",
        )
        
        # Should not crash
        result = chat_mode.handle_action(action)
        assert isinstance(result, ActionResult)
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_action_without_details(self, mock_prompt, chat_mode):
        """Test action without details dict."""
        mock_prompt.return_value = "y"
        
        action = ActionRequest(
            action_type="custom_action",
            target="something",
        )
        
        result = chat_mode.handle_action(action)
        
        assert result.success  # Should handle gracefully
    
    @patch('quirkllm.modes.chat_mode.Prompt.ask')
    def test_diff_without_old_content(self, mock_prompt, chat_mode):
        """Test diff display when old_content missing."""
        mock_prompt.return_value = "y"
        
        action = ActionRequest(
            action_type="edit_file",
            target="test.py",
            details={"new_content": "new code"},
        )
        
        # Should not crash
        result = chat_mode.handle_action(action)
        assert result.success
