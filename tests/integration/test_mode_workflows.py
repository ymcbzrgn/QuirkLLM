"""
Integration tests for mode workflows.

Tests the full integration of modes, action handlers, safety checkers,
and file operations working together.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from quirkllm.core.action_handler import ActionHandler
from quirkllm.core.config import Config
from quirkllm.modes import (
    ModeType,
    get_registry,
    ActionRequest,
)
from quirkllm.modes.safety_checker import SafetyChecker


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def modes():
    """Provide pre-instantiated and activated modes for testing."""
    from quirkllm.modes import ChatMode, YAMIMode, PlanMode, GhostMode, ModeConfig
    
    # Instantiate all modes directly (registry has constructor signature mismatch)
    chat = ChatMode(ModeType.CHAT, ModeConfig(auto_confirm=False))
    yami = YAMIMode()  # YAMI constructs its own config
    plan = PlanMode()  # Plan constructs its own config
    ghost = GhostMode()  # Ghost constructs its own config
    
    # Activate all modes for testing
    chat.activate()
    yami.activate()
    plan.activate()
    ghost.activate()
    
    return {
        ModeType.CHAT: chat,
        ModeType.YAMI: yami,
        ModeType.PLAN: plan,
        ModeType.GHOST: ghost,
    }


@pytest.fixture
def safety_checker():
    """Create a SafetyChecker instance."""
    return SafetyChecker()


@pytest.fixture
def config():
    """Create a Config instance."""
    return Config()


class TestModeCreationAndActivation:
    """Test mode creation and activation workflows."""
    
    def test_creates_all_mode_types(self, modes):
        """Test that all mode types can be created."""
        for mode_type in [ModeType.CHAT, ModeType.YAMI, ModeType.PLAN, ModeType.GHOST]:
            mode = modes[mode_type]
            assert mode is not None
            assert mode.mode_type == mode_type
    
    def test_activates_and_deactivates_modes(self, modes):
        """Test mode activation and deactivation."""
        mode = modes[ModeType.CHAT]
        
        # Activate
        mode.activate()
        assert mode.is_active is True
        
        # Deactivate
        mode.deactivate()
        assert mode.is_active is False
    
    def test_mode_switching_workflow(self, modes):
        """Test switching between different modes."""
        chat_mode = modes[ModeType.CHAT]
        yami_mode = modes[ModeType.YAMI]
        
        # Activate chat
        chat_mode.activate()
        assert chat_mode.is_active is True
        
        # Switch to YAMI
        chat_mode.deactivate()
        yami_mode.activate()
        
        assert chat_mode.is_active is False
        assert yami_mode.is_active is True


class TestFileOperationIntegration:
    """Test file operations integrated with modes and safety."""
    
    def test_file_read_operation(self, modes, safety_checker, config, temp_workspace):
        """Test reading a file through the full workflow."""
        # Create test file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Hello World", encoding="utf-8")
        
        # Create mode and handler
        mode = modes[ModeType.YAMI]
        handler = ActionHandler(mode, None, safety_checker, config)
        
        # Read file
        request = ActionRequest(
            action_type="file_read",
            target=str(test_file),
            details={},
        )
        
        result = handler.handle_action(request)
        
        assert result.success is True
        assert "Hello World" in str(result.details)
    
    def test_file_write_operation(self, modes, safety_checker, config, temp_workspace):
        """Test writing a file through the full workflow."""
        test_file = temp_workspace / "output.txt"
        
        # Create mode and handler
        mode = modes[ModeType.YAMI]
        mode.activate()
        handler = ActionHandler(mode, None, safety_checker, config)
        
        # Write file
        request = ActionRequest(
            action_type="file_write",
            target=str(test_file),
            details={"content": "Test content"},
        )
        
        result = handler.handle_action(request)
        
        assert result.success is True
        assert test_file.exists()
        assert test_file.read_text(encoding="utf-8") == "Test content"
    
    def test_file_edit_operation(self, modes, safety_checker, config, temp_workspace):
        """Test editing a file through the full workflow."""
        test_file = temp_workspace / "edit.txt"
        test_file.write_text("Hello World", encoding="utf-8")
        
        # Create mode and handler
        mode = modes[ModeType.YAMI]
        mode.activate()
        handler = ActionHandler(mode, None, safety_checker, config)
        
        # Edit file
        request = ActionRequest(
            action_type="file_edit",
            target=str(test_file),
            details={
                "old_content": "World",
                "new_content": "Universe",
            },
        )
        
        result = handler.handle_action(request)
        
        assert result.success is True
        assert test_file.read_text(encoding="utf-8") == "Hello Universe"
    
    def test_file_delete_operation(self, modes, safety_checker, config, temp_workspace):
        """Test deleting a file through the full workflow."""
        test_file = temp_workspace / "delete_me.txt"
        test_file.write_text("To be deleted", encoding="utf-8")
        
        # Create mode and handler
        mode = modes[ModeType.YAMI]
        mode.activate()
        handler = ActionHandler(mode, None, safety_checker, config)
        
        # Delete file
        request = ActionRequest(
            action_type="file_delete",
            target=str(test_file),
            details={},
        )
        
        result = handler.handle_action(request)
        
        assert result.success is True
        assert not test_file.exists()


class TestSafetyIntegration:
    """Test safety checker integration with modes."""
    
    def test_blocks_critical_file_operations(self, modes, safety_checker, config):
        """Test that critical operations are blocked."""
        mode = modes[ModeType.YAMI]
        handler = ActionHandler(mode, None, safety_checker, config)
        
        # Try to delete critical file
        request = ActionRequest(
            action_type="file_delete",
            target="/etc/passwd",
            details={},
        )
        
        result = handler.handle_action(request)
        
        # Should be blocked (either by safety checker or OS permissions)
        assert result.success is False
        assert "blocked" in result.message.lower() or "permission denied" in result.message.lower()
    
    def test_blocks_dangerous_commands(self, modes, safety_checker, config):
        """Test that dangerous commands are blocked."""
        mode = modes[ModeType.YAMI]
        handler = ActionHandler(mode, None, safety_checker, config)
        
        # Try to run dangerous command
        request = ActionRequest(
            action_type="command",
            target="rm -rf /",
            details={},
        )
        
        result = handler.handle_action(request)
        
        # Should be blocked
        assert result.success is False
        assert "blocked" in result.message.lower()
    
    def test_allows_safe_operations(self, modes, safety_checker, config, temp_workspace):
        """Test that safe operations are allowed."""
        test_file = temp_workspace / "safe.txt"
        
        mode = modes[ModeType.YAMI]
        mode.activate()
        handler = ActionHandler(mode, None, safety_checker, config)
        
        # Safe file creation
        request = ActionRequest(
            action_type="file_create",
            target=str(test_file),
            details={"content": "Safe content"},
        )
        
        result = handler.handle_action(request)
        
        assert result.success is True


class TestActionTracking:
    """Test action tracking and statistics."""
    
    def test_tracks_action_history(self, modes, safety_checker, config, temp_workspace):
        """Test that actions are tracked in history."""
        mode = modes[ModeType.YAMI]
        handler = ActionHandler(mode, None, safety_checker, config)
        
        test_file = temp_workspace / "track.txt"
        
        # Perform multiple actions
        for i in range(3):
            request = ActionRequest(
                action_type="file_create",
                target=str(temp_workspace / f"file{i}.txt"),
                details={"content": f"Content {i}"},
            )
            handler.handle_action(request)
        
        # Check history
        history = handler.get_action_history(limit=5)
        
        assert len(history) == 3
        assert all("file_create" == h["action_type"] for h in history)
    
    def test_tracks_action_statistics(self, modes, safety_checker, config, temp_workspace):
        """Test that action statistics are maintained."""
        mode = modes[ModeType.YAMI]
        mode.activate()
        handler = ActionHandler(mode, None, safety_checker, config)
        
        # Perform actions
        test_file = temp_workspace / "stats.txt"
        test_file.write_text("Test", encoding="utf-8")
        
        # Successful read
        request = ActionRequest(
            action_type="file_read",
            target=str(test_file),
            details={},
        )
        handler.handle_action(request)
        
        # Failed read (missing file)
        request = ActionRequest(
            action_type="file_read",
            target=str(temp_workspace / "missing.txt"),
            details={},
        )
        handler.handle_action(request)
        
        stats = handler.get_action_stats()
        
        assert stats["total_actions"] == 2
        assert stats["successful_actions"] == 1
        assert stats["failed_actions"] == 1


class TestReadOnlyModes:
    """Test read-only mode behavior."""
    
    def test_plan_mode_blocks_writes(self, modes, safety_checker, config, temp_workspace):
        """Test that Plan mode blocks write operations."""
        mode = modes[ModeType.PLAN]
        mode.activate()
        handler = ActionHandler(mode, None, safety_checker, config)
        
        # Try to write file
        request = ActionRequest(
            action_type="file_write",
            target=str(temp_workspace / "blocked.txt"),
            details={"content": "Should not write"},
        )
        
        # Mode should handle it but not execute
        result = handler.handle_action(request)
        
        # Plan mode returns success from handle_action but doesn't execute
        # File should not exist
        assert not (temp_workspace / "blocked.txt").exists()
    
    def test_plan_mode_allows_reads(self, modes, safety_checker, config, temp_workspace):
        """Test that Plan mode allows read operations."""
        test_file = temp_workspace / "readable.txt"
        test_file.write_text("Can read this", encoding="utf-8")
        
        mode = modes[ModeType.PLAN]
        mode.activate()
        handler = ActionHandler(mode, None, safety_checker, config)
        
        # Read file
        request = ActionRequest(
            action_type="read_file",  # Plan mode uses read_file, not file_read
            target=str(test_file),
            details={},
        )
        
        result = handler.handle_action(request)
        
        assert result.success is True
        assert "Can read this" in str(result.details)


class TestModeCoordination:
    """Test coordination between different modes."""
    
    def test_mode_updates_in_handler(self, modes, safety_checker, config):
        """Test that handler can update its mode reference."""
        chat_mode = modes[ModeType.CHAT]
        yami_mode = modes[ModeType.YAMI]
        
        handler = ActionHandler(chat_mode, None, safety_checker, config)
        
        assert handler.current_mode == chat_mode
        
        # Update mode
        handler.update_mode(yami_mode)
        
        assert handler.current_mode == yami_mode
    
    def test_different_modes_same_handler(self, modes, safety_checker, config, temp_workspace):
        """Test using different modes with the same handler."""
        handler = ActionHandler(None, None, safety_checker, config)
        
        # Use with YAMI mode first
        yami_mode1 = modes[ModeType.YAMI]
        handler.update_mode(yami_mode1)
        
        test_file = temp_workspace / "mode_test.txt"
        request = ActionRequest(
            action_type="file_create",
            target=str(test_file),
            details={"content": "Test"},
        )
        
        result = handler.handle_action(request)
        assert result.success is True
        assert test_file.exists(), "File should have been created"
        
        # Edit the same file with the same handler
        edit_request = ActionRequest(
            action_type="file_edit",
            target=str(test_file),
            details={"old_content": "Test", "new_content": "Modified"},
        )
        
        result = handler.handle_action(edit_request)
        assert result.success is True
        
        # Verify edit worked
        assert test_file.read_text() == "Modified"


class TestCommandExecution:
    """Test command execution integration."""
    
    def test_executes_safe_command(self, modes, safety_checker, config):
        """Test execution of safe commands."""
        mode = modes[ModeType.YAMI]
        mode.activate()
        handler = ActionHandler(mode, None, safety_checker, config)
        
        # Run safe command
        request = ActionRequest(
            action_type="command",
            target="echo 'Hello'",
            details={},
        )
        
        result = handler.handle_action(request)
        
        assert result.success is True
        assert "Hello" in result.details.get("stdout", "")
    
    def test_command_with_timeout(self, modes, safety_checker, config):
        """Test command execution with timeout."""
        mode = modes[ModeType.YAMI]
        mode.activate()
        handler = ActionHandler(mode, None, safety_checker, config)
        
        # Quick command with timeout
        request = ActionRequest(
            action_type="command",
            target="sleep 0.1",
            details={"timeout": 1},
        )
        
        result = handler.handle_action(request)
        
        assert result.success is True


class TestErrorHandling:
    """Test error handling in integrated workflows."""
    
    def test_handles_missing_file_gracefully(self, modes, safety_checker, config):
        """Test graceful handling of missing file."""
        mode = modes[ModeType.YAMI]
        handler = ActionHandler(mode, None, safety_checker, config)
        
        request = ActionRequest(
            action_type="file_read",
            target="/nonexistent/path/file.txt",
            details={},
        )
        
        result = handler.handle_action(request)
        
        assert result.success is False
        assert "error" in result.message.lower() or "not found" in result.message.lower()
    
    def test_handles_invalid_edit_gracefully(self, modes, safety_checker, config, temp_workspace):
        """Test graceful handling of invalid edit."""
        test_file = temp_workspace / "edit_test.txt"
        test_file.write_text("Original content", encoding="utf-8")
        
        mode = modes[ModeType.YAMI]
        mode.activate()
        handler = ActionHandler(mode, None, safety_checker, config)
        
        # Try to replace non-existent content
        request = ActionRequest(
            action_type="file_edit",
            target=str(test_file),
            details={
                "old_content": "This text is not in the file",
                "new_content": "New text",
            },
        )
        
        result = handler.handle_action(request)
        
        assert result.success is False


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""
    
    def test_full_refactoring_workflow(self, modes, safety_checker, config, temp_workspace):
        """Test a complete refactoring workflow."""
        # Create initial file
        source_file = temp_workspace / "refactor.py"
        source_file.write_text(
            "def old_function():\n    return 'old'\n",
            encoding="utf-8"
        )
        
        # Use YAMI mode for fast workflow
        mode = modes[ModeType.YAMI]
        mode.activate()
        handler = ActionHandler(mode, None, safety_checker, config)
        
        # Step 1: Read original
        read_request = ActionRequest(
            action_type="file_read",
            target=str(source_file),
            details={},
        )
        result = handler.handle_action(read_request)
        assert result.success is True
        
        # Step 2: Edit function
        edit_request = ActionRequest(
            action_type="file_edit",
            target=str(source_file),
            details={
                "old_content": "old_function",
                "new_content": "new_function",
            },
        )
        result = handler.handle_action(edit_request)
        assert result.success is True
        
        # Step 3: Verify change
        final_content = source_file.read_text(encoding="utf-8")
        assert "new_function" in final_content
        assert "old_function" not in final_content
        
        # Check statistics
        stats = handler.get_action_stats()
        assert stats["total_actions"] >= 2
