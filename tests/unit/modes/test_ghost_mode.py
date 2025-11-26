"""
Tests for Ghost Mode - Background file watcher and impact analyzer.

Ghost mode tests verify that:
- File watcher starts and stops correctly
- Events are detected and queued
- Debouncing works properly
- Pattern matching filters files correctly
- Threading is safe and clean
- Read-only enforcement works
- Session statistics track correctly
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from quirkllm.modes.ghost_mode import GhostMode, CodeChangeHandler
from quirkllm.modes.base import (
    ModeType,
    ModeConfig,
    ActionRequest,
    ActionResult,
)


class TestGhostModeInitialization:
    """Test Ghost mode creation and configuration."""
    
    def test_creates_ghost_mode(self):
        """Test basic Ghost mode creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            
            assert mode is not None
            assert mode.mode_type == ModeType.GHOST
            assert mode.config.auto_confirm is True
            assert mode.config.allow_file_edits is False
            assert mode.config.allow_destructive is False
            assert mode.config.background_watch is True
    
    def test_initializes_with_default_patterns(self):
        """Test default watch patterns are set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            
            assert len(mode.config.watch_patterns) > 0
            assert "**/*.py" in mode.config.watch_patterns
    
    def test_initializes_with_custom_patterns(self):
        """Test custom watch patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_patterns = ["*.txt", "*.md"]
            mode = GhostMode(watch_dir=tmpdir, patterns=custom_patterns)
            
            assert mode.config.watch_patterns == custom_patterns
    
    def test_initializes_session_stats(self):
        """Test session statistics are initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            
            assert mode.session_stats == {
                "changes_detected": 0,
                "files_analyzed": 0,
                "watcher_active": False,
            }
    
    def test_initializes_empty_queue(self):
        """Test analysis queue is empty on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            
            assert mode.analysis_queue == []


class TestGhostModeActivationDeactivation:
    """Test Ghost mode activation and deactivation."""
    
    @patch("quirkllm.modes.ghost_mode.Observer")
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_activate_sets_flag(self, mock_console_class, mock_observer_class):
        """Test activation sets active flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            mode.activate()
            
            assert mode.active is True
    
    @patch("quirkllm.modes.ghost_mode.Observer")
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_activate_starts_watcher(self, mock_console_class, mock_observer_class):
        """Test activation starts file watcher."""
        mock_observer = Mock()
        mock_observer_class.return_value = mock_observer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            mode.activate()
            
            # Verify observer was started
            assert mock_observer.start.called
            assert mode.session_stats["watcher_active"] is True
    
    @patch("quirkllm.modes.ghost_mode.Observer")
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_activate_displays_welcome_panel(self, mock_console_class, mock_observer_class):
        """Test activation displays welcome message."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            mode.activate()
            
            # Verify console.print was called
            assert mock_console.print.called
    
    @patch("quirkllm.modes.ghost_mode.Observer")
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_deactivate_stops_watcher(self, mock_console_class, mock_observer_class):
        """Test deactivation stops file watcher."""
        mock_observer = Mock()
        mock_observer_class.return_value = mock_observer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            mode.activate()
            mode.deactivate()
            
            # Verify observer was stopped
            assert mock_observer.stop.called
            assert mode.active is False
    
    @patch("quirkllm.modes.ghost_mode.Observer")
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_deactivate_displays_summary(self, mock_console_class, mock_observer_class):
        """Test deactivation displays session summary."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            mode.session_stats = {
                "changes_detected": 5,
                "files_analyzed": 3,
                "watcher_active": True,
            }
            
            mode.activate()
            mode.deactivate()
            
            # Verify console.print was called (summary)
            assert mock_console.print.called


class TestGhostModePromptIndicator:
    """Test Ghost mode prompt indicator."""
    
    def test_returns_ghost_emoji(self):
        """Test prompt indicator is ghost emoji."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            
            assert mode.get_prompt_indicator() == "👻"


class TestCodeChangeHandler:
    """Test file system event handler."""
    
    def test_creates_handler(self):
        """Test handler creation."""
        callback = Mock()
        patterns = ["*.py"]
        
        handler = CodeChangeHandler(callback, patterns)
        
        assert handler.callback == callback
        assert handler.patterns == patterns
    
    def test_should_watch_matching_pattern(self):
        """Test pattern matching for watched files."""
        callback = Mock()
        patterns = ["*.py", "*.js"]
        
        handler = CodeChangeHandler(callback, patterns)
        
        assert handler._should_watch("/path/to/file.py") is True
        assert handler._should_watch("/path/to/file.js") is True
        assert handler._should_watch("/path/to/file.txt") is False
    
    def test_debouncing_prevents_duplicates(self):
        """Test debouncing prevents duplicate events."""
        callback = Mock()
        patterns = ["*.py"]
        
        handler = CodeChangeHandler(callback, patterns, debounce_time=0.5)
        
        filepath = "/path/to/test.py"
        
        # First event should process
        assert handler._should_process(filepath) is True
        
        # Immediate second event should be debounced
        assert handler._should_process(filepath) is False
        
        # After debounce time, should process again
        time.sleep(0.6)
        assert handler._should_process(filepath) is True
    
    def test_on_modified_calls_callback(self):
        """Test modified event calls callback."""
        callback = Mock()
        patterns = ["*.py"]
        
        handler = CodeChangeHandler(callback, patterns, debounce_time=0.1)
        
        # Create mock event
        event = Mock()
        event.is_directory = False
        event.src_path = "/path/to/test.py"
        
        handler.on_modified(event)
        
        # Verify callback was called
        assert callback.called
        callback.assert_called_with("/path/to/test.py", "modified")
    
    def test_ignores_directory_events(self):
        """Test directory events are ignored."""
        callback = Mock()
        patterns = ["*.py"]
        
        handler = CodeChangeHandler(callback, patterns)
        
        # Create mock directory event
        event = Mock()
        event.is_directory = True
        event.src_path = "/path/to/dir"
        
        handler.on_modified(event)
        
        # Callback should not be called
        assert not callback.called


class TestGhostModeFileWatching:
    """Test Ghost mode file watching functionality."""
    
    @patch("quirkllm.modes.ghost_mode.Observer")
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_queues_file_changes(self, mock_console_class, mock_observer_class):
        """Test file changes are added to queue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            mode.activate()
            
            # Simulate file change
            mode._on_file_changed("/path/to/test.py", "modified")
            
            assert len(mode.analysis_queue) == 1
            assert mode.analysis_queue[0]["filepath"] == "/path/to/test.py"
            assert mode.analysis_queue[0]["event_type"] == "modified"
    
    @patch("quirkllm.modes.ghost_mode.Observer")
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_increments_change_counter(self, mock_console_class, mock_observer_class):
        """Test change counter increments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            mode.activate()
            
            mode._on_file_changed("/path/to/test1.py", "modified")
            mode._on_file_changed("/path/to/test2.py", "created")
            
            assert mode.session_stats["changes_detected"] == 2


class TestGhostModeReadOnlyEnforcement:
    """Test Ghost mode blocks write operations."""
    
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_blocks_file_edit(self, mock_console_class):
        """Test file edit is blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            
            request = ActionRequest(
                action_type="file_edit",
                target="test.py",
                details={},
            )
            
            result = mode.handle_action(request)
            
            assert result.success is False
            assert "read-only" in result.message.lower()
    
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_blocks_delete(self, mock_console_class):
        """Test delete is blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            
            request = ActionRequest(
                action_type="delete",
                target="test.py",
                details={},
            )
            
            result = mode.handle_action(request)
            
            assert result.success is False
    
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_blocks_command(self, mock_console_class):
        """Test command execution is blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            
            request = ActionRequest(
                action_type="command",
                target="echo test",
                details={},
            )
            
            result = mode.handle_action(request)
            
            assert result.success is False
    
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_allows_read_file(self, mock_console_class):
        """Test read_file is allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            
            request = ActionRequest(
                action_type="read_file",
                target="README.md",
                details={},
            )
            
            result = mode.handle_action(request)
            
            assert result.success is True


class TestGhostModeAnalysis:
    """Test Ghost mode analysis functionality."""
    
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_analyzes_existing_file(self, mock_console_class):
        """Test analysis of existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("print('hello')")
            
            mode = GhostMode(watch_dir=tmpdir)
            
            request = ActionRequest(
                action_type="analyze_change",
                target=str(test_file),
                details={},
            )
            
            result = mode.handle_action(request)
            
            assert result.success is True
            assert "test.py" in result.message
    
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_analysis_fails_for_missing_file(self, mock_console_class):
        """Test analysis fails for non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            
            request = ActionRequest(
                action_type="analyze_change",
                target="/nonexistent/file.py",
                details={},
            )
            
            result = mode.handle_action(request)
            
            assert result.success is False


class TestGhostModeSessionStatistics:
    """Test Ghost mode session statistics tracking."""
    
    @patch("quirkllm.modes.ghost_mode.Observer")
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_get_session_stats_returns_copy(self, mock_console_class, mock_observer_class):
        """Test get_session_stats returns a copy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            mode.session_stats = {
                "changes_detected": 10,
                "files_analyzed": 5,
                "watcher_active": True,
            }
            
            stats = mode.get_session_stats()
            
            # Modify returned stats
            stats["changes_detected"] = 999
            
            # Original should be unchanged
            assert mode.session_stats["changes_detected"] == 10
    
    @patch("quirkllm.modes.ghost_mode.Observer")
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_get_analysis_queue_returns_copy(self, mock_console_class, mock_observer_class):
        """Test get_analysis_queue returns a copy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            mode.activate()
            
            mode._on_file_changed("/test.py", "modified")
            
            queue = mode.get_analysis_queue()
            
            # Modify returned queue
            queue.append({"fake": "item"})
            
            # Original should be unchanged
            assert len(mode.analysis_queue) == 1


class TestGhostModeEdgeCases:
    """Test Ghost mode edge cases and error handling."""
    
    @patch("quirkllm.modes.ghost_mode.Observer")
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_deactivate_without_observer(self, mock_console_class, mock_observer_class):
        """Test deactivation without starting observer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            
            # Deactivate without activate
            mode.deactivate()
            
            # Should handle gracefully
            assert mode.active is False
    
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_handles_unsupported_action(self, mock_console_class):
        """Test handling of unsupported action type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            
            request = ActionRequest(
                action_type="unsupported_action",
                target="",
                details={},
            )
            
            result = mode.handle_action(request)
            
            assert result.success is False
            assert "not supported" in result.message.lower()
    
    @patch("quirkllm.modes.ghost_mode.Observer")
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_multiple_changes_queued_correctly(self, mock_console_class, mock_observer_class):
        """Test multiple file changes are queued in order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            mode.activate()
            
            # Queue multiple changes
            files = ["/test1.py", "/test2.js", "/test3.ts"]
            for filepath in files:
                mode._on_file_changed(filepath, "modified")
            
            assert len(mode.analysis_queue) == 3
            assert mode.analysis_queue[0]["filepath"] == "/test1.py"
            assert mode.analysis_queue[1]["filepath"] == "/test2.js"
            assert mode.analysis_queue[2]["filepath"] == "/test3.ts"


class TestGhostModeIntegration:
    """Integration tests for Ghost mode."""
    
    @patch("quirkllm.modes.ghost_mode.Observer")
    @patch("quirkllm.modes.ghost_mode.Console")
    def test_end_to_end_watching_flow(self, mock_console_class, mock_observer_class):
        """Test complete watching workflow."""
        mock_observer = Mock()
        mock_observer_class.return_value = mock_observer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)
            
            # Activate
            mode.activate()
            assert mode.active is True
            assert mode.watching.is_set()
            
            # Simulate file changes
            mode._on_file_changed(f"{tmpdir}/test.py", "modified")
            assert len(mode.analysis_queue) == 1
            
            # Deactivate
            mode.deactivate()
            assert mode.active is False
            assert mock_observer.stop.called
