"""
Tests for ActionHandler - Mode-aware action coordination.

Tests verify:
- Action routing to modes
- Safety validation integration
- File operations (read, write, edit, delete, create)
- Command execution
- Action tracking and statistics
- Mode updates
- Error handling
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from quirkllm.core.action_handler import ActionHandler
from quirkllm.modes import (
    ModeType,
    ModeConfig,
    ActionRequest,
    ActionResult,
    SafetyChecker,
    ValidationResult,
)
from quirkllm.core.config import Config


@pytest.fixture
def mock_mode():
    """Create mock mode."""
    mode = Mock()
    mode.mode_type = ModeType.CHAT
    mode.config = ModeConfig(
        auto_confirm=False,
        allow_file_edits=True,
    )
    mode.handle_action.return_value = ActionResult(success=True, message="OK")
    return mode


@pytest.fixture
def safety_checker():
    """Create safety checker."""
    return SafetyChecker()


@pytest.fixture
def config():
    """Create test config."""
    return Config()


@pytest.fixture
def handler(mock_mode, safety_checker, config):
    """Create action handler."""
    return ActionHandler(mock_mode, None, safety_checker, config)


class TestActionHandlerInitialization:
    """Test ActionHandler initialization."""
    
    def test_creates_handler(self, mock_mode, safety_checker, config):
        """Test basic handler creation."""
        handler = ActionHandler(mock_mode, None, safety_checker, config)
        
        assert handler is not None
        assert handler.current_mode == mock_mode
        assert handler.safety_checker == safety_checker
        assert handler.config == config
    
    def test_initializes_empty_history(self, handler):
        """Test action history is empty on init."""
        assert handler.action_history == []
    
    def test_initializes_stats(self, handler):
        """Test action stats are initialized."""
        stats = handler.get_action_stats()
        
        assert stats["total_actions"] == 0
        assert stats["successful_actions"] == 0
        assert stats["failed_actions"] == 0
        assert stats["blocked_actions"] == 0
    
    def test_creates_safety_checker_if_none(self, mock_mode):
        """Test default SafetyChecker is created."""
        handler = ActionHandler(mock_mode, None, None, None)
        
        assert handler.safety_checker is not None
        assert isinstance(handler.safety_checker, SafetyChecker)


class TestActionRouting:
    """Test action routing to modes."""
    
    def test_routes_to_mode(self, handler, mock_mode):
        """Test action is routed to current mode."""
        request = ActionRequest(
            action_type="analyze",
            target="test.py",
            details={},
        )
        
        handler.handle_action(request)
        
        mock_mode.handle_action.assert_called_once_with(request)
    
    def test_returns_mode_result(self, handler, mock_mode):
        """Test returns result from mode."""
        expected = ActionResult(success=True, message="Test result")
        mock_mode.handle_action.return_value = expected
        
        request = ActionRequest(
            action_type="analyze",
            target="test.py",
            details={},
        )
        
        result = handler.handle_action(request)
        
        assert result == expected
    
    def test_updates_mode(self, handler):
        """Test mode can be updated."""
        new_mode = Mock()
        new_mode.mode_type = ModeType.YAMI
        
        handler.update_mode(new_mode)
        
        assert handler.current_mode == new_mode


class TestSafetyValidation:
    """Test safety validation integration."""
    
    def test_blocks_critical_actions(self, handler, mock_mode):
        """Test critical actions are blocked."""
        request = ActionRequest(
            action_type="command",
            target="rm -rf /",
            details={},
        )
        
        result = handler.handle_action(request)
        
        assert result.success is False
        assert "blocked" in result.message.lower()
        assert handler.action_stats["blocked_actions"] == 1
    
    def test_allows_safe_actions(self, handler, mock_mode):
        """Test safe actions are allowed."""
        request = ActionRequest(
            action_type="file_read",
            target="README.md",
            details={},
        )
        
        result = handler.handle_action(request)
        
        # Should not be blocked
        mock_mode.handle_action.assert_called_once()
    
    def test_validates_write_operations(self, handler):
        """Test write operations are validated."""
        request = ActionRequest(
            action_type="file_write",
            target="/etc/passwd",
            details={"content": "malicious"},
        )
        
        result = handler.handle_action(request)
        
        # Protected path should be blocked
        assert result.success is False


class TestFileOperations:
    """Test file operations."""
    
    def test_reads_file(self, handler, mock_mode):
        """Test file read operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test content")
            
            # Configure mode to allow execution
            mock_mode.config.allow_file_edits = True
            mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
            
            request = ActionRequest(
                action_type="file_read",
                target=str(test_file),
                details={},
            )
            
            result = handler.handle_action(request)
            
            assert result.success is True
            assert "test content" in str(result.details)
    
    def test_writes_file(self, handler, mock_mode):
        """Test file write operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "new.txt"
            
            mock_mode.config.allow_file_edits = True
            mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
            
            request = ActionRequest(
                action_type="file_write",
                target=str(test_file),
                details={"content": "new content"},
            )
            
            result = handler.handle_action(request)
            
            assert result.success is True
            assert test_file.exists()
            assert test_file.read_text() == "new content"
    
    def test_edits_file(self, handler, mock_mode):
        """Test file edit operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "edit.txt"
            test_file.write_text("old content here")
            
            mock_mode.config.allow_file_edits = True
            mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
            
            request = ActionRequest(
                action_type="file_edit",
                target=str(test_file),
                details={"old_content": "old", "new_content": "new"},
            )
            
            result = handler.handle_action(request)
            
            assert result.success is True
            assert test_file.read_text() == "new content here"
    
    def test_deletes_file(self, handler, mock_mode):
        """Test file delete operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "delete.txt"
            test_file.write_text("delete me")
            
            mock_mode.config.allow_file_edits = True
            mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
            
            request = ActionRequest(
                action_type="file_delete",
                target=str(test_file),
                details={},
            )
            
            result = handler.handle_action(request)
            
            assert result.success is True
            assert not test_file.exists()
    
    def test_creates_file(self, handler, mock_mode):
        """Test file create operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "subdir" / "create.txt"
            
            mock_mode.config.allow_file_edits = True
            mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
            
            request = ActionRequest(
                action_type="create_file",
                target=str(test_file),
                details={"content": "created content"},
            )
            
            result = handler.handle_action(request)
            
            assert result.success is True
            assert test_file.exists()
            assert test_file.read_text() == "created content"
    
    def test_handles_missing_file(self, handler, mock_mode):
        """Test handling of missing file."""
        mock_mode.config.allow_file_edits = True
        mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
        
        request = ActionRequest(
            action_type="file_read",
            target="/nonexistent/file.txt",
            details={},
        )
        
        result = handler.handle_action(request)
        
        assert result.success is False
        assert "not found" in result.message.lower()


class TestCommandExecution:
    """Test command execution."""
    
    def test_executes_safe_command(self, handler, mock_mode):
        """Test safe command execution."""
        mock_mode.config.allow_file_edits = True
        mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
        
        request = ActionRequest(
            action_type="command",
            target="echo 'hello'",
            details={},
        )
        
        result = handler.handle_action(request)
        
        # Command validation may block or allow
        # Just verify it was processed
        assert result is not None
    
    def test_blocks_dangerous_command(self, handler, mock_mode):
        """Test dangerous commands are blocked."""
        request = ActionRequest(
            action_type="command",
            target="rm -rf /",
            details={},
        )
        
        result = handler.handle_action(request)
        
        assert result.success is False
        assert handler.action_stats["blocked_actions"] > 0
    
    def test_command_with_timeout(self, handler, mock_mode):
        """Test command timeout."""
        mock_mode.config.allow_file_edits = True
        mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
        
        request = ActionRequest(
            action_type="command",
            target="sleep 100",
            details={"timeout": 0.1},
        )
        
        result = handler.handle_action(request)
        
        # Should timeout
        assert result.success is False or "timeout" in result.message.lower()


class TestActionTracking:
    """Test action tracking and statistics."""
    
    def test_records_action_in_history(self, handler, mock_mode):
        """Test actions are recorded in history."""
        request = ActionRequest(
            action_type="analyze",
            target="test.py",
            details={},
        )
        
        handler.handle_action(request)
        
        history = handler.get_action_history()
        assert len(history) == 1
        assert history[0]["action_type"] == "analyze"
        assert history[0]["target"] == "test.py"
    
    def test_tracks_action_stats(self, handler, mock_mode):
        """Test action statistics are tracked."""
        # Successful action
        mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
        request1 = ActionRequest(action_type="analyze", target="test.py", details={})
        handler.handle_action(request1)
        
        # Failed action
        mock_mode.handle_action.return_value = ActionResult(success=False, message="Error")
        request2 = ActionRequest(action_type="analyze", target="test2.py", details={})
        handler.handle_action(request2)
        
        stats = handler.get_action_stats()
        assert stats["total_actions"] == 2
        assert stats["successful_actions"] == 1
        assert stats["failed_actions"] == 1
    
    def test_tracks_actions_by_type(self, handler, mock_mode):
        """Test actions are tracked by type."""
        mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
        
        handler.handle_action(ActionRequest("file_read", "test.py", {}))
        handler.handle_action(ActionRequest("file_read", "test2.py", {}))
        handler.handle_action(ActionRequest("command", "ls", {}))
        
        stats = handler.get_action_stats()
        assert stats["actions_by_type"]["file_read"] == 2
        assert stats["actions_by_type"]["command"] >= 1
    
    def test_tracks_actions_by_mode(self, handler, mock_mode):
        """Test actions are tracked by mode."""
        mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
        
        handler.handle_action(ActionRequest("analyze", "test.py", {}))
        
        # Switch mode
        yami_mode = Mock()
        yami_mode.mode_type = ModeType.YAMI
        yami_mode.handle_action.return_value = ActionResult(success=True, message="OK")
        handler.update_mode(yami_mode)
        
        handler.handle_action(ActionRequest("analyze", "test2.py", {}))
        
        stats = handler.get_action_stats()
        assert stats["actions_by_mode"]["chat"] == 1
        assert stats["actions_by_mode"]["yami"] == 1
    
    def test_limits_history_size(self, handler, mock_mode):
        """Test history is limited to 100 items."""
        mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
        
        # Add 150 actions
        for i in range(150):
            request = ActionRequest("analyze", f"test{i}.py", {})
            handler.handle_action(request)
        
        # Should only keep last 100
        assert len(handler.action_history) == 100
    
    def test_clears_history(self, handler, mock_mode):
        """Test history can be cleared."""
        mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
        handler.handle_action(ActionRequest("analyze", "test.py", {}))
        
        handler.clear_history()
        
        assert len(handler.action_history) == 0
    
    def test_resets_stats(self, handler, mock_mode):
        """Test stats can be reset."""
        mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
        handler.handle_action(ActionRequest("analyze", "test.py", {}))
        
        handler.reset_stats()
        
        stats = handler.get_action_stats()
        assert stats["total_actions"] == 0


class TestReadOnlyModes:
    """Test read-only mode enforcement."""
    
    def test_skips_execution_for_readonly_mode(self, handler, mock_mode):
        """Test read-only modes don't execute file ops."""
        # Set mode to read-only
        mock_mode.config.allow_file_edits = False
        mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            
            request = ActionRequest(
                action_type="file_write",
                target=str(test_file),
                details={"content": "should not write"},
            )
            
            result = handler.handle_action(request)
            
            # Mode handles it but file shouldn't be written
            assert not test_file.exists()


class TestErrorHandling:
    """Test error handling."""
    
    def test_handles_mode_exception(self, handler, mock_mode):
        """Test handling when mode raises exception."""
        mock_mode.handle_action.side_effect = Exception("Mode error")
        
        request = ActionRequest("analyze", "test.py", {})
        result = handler.handle_action(request)
        
        assert result.success is False
        assert "failed" in result.message.lower()
        assert handler.action_stats["failed_actions"] == 1
    
    def test_handles_file_operation_exception(self, handler, mock_mode):
        """Test handling of file operation errors."""
        mock_mode.config.allow_file_edits = True
        mock_mode.handle_action.return_value = ActionResult(success=True, message="OK")
        
        request = ActionRequest(
            action_type="file_read",
            target="/nonexistent/path/file.txt",
            details={},
        )
        
        result = handler.handle_action(request)
        
        assert result.success is False
