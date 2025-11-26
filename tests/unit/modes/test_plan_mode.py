"""
Tests for Plan Mode - Read-only planning and documentation mode.

Plan mode tests verify that:
- Read-only enforcement blocks write operations
- Plan generation creates proper markdown files
- Plans are saved to correct directory
- Session statistics track correctly
- File naming is sanitized properly
- Metadata formatting is correct
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from quirkllm.modes.plan_mode import PlanMode
from quirkllm.modes.base import (
    ModeType,
    ModeConfig,
    ActionRequest,
    ActionResult,
)


class TestPlanModeInitialization:
    """Test Plan mode creation and configuration."""
    
    def test_creates_plan_mode(self):
        """Test basic Plan mode creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            assert mode is not None
            assert mode.mode_type == ModeType.PLAN
            assert mode.config.auto_confirm is True
            assert mode.config.allow_file_edits is False
            assert mode.config.allow_destructive is False
            assert mode.config.diff_display is False
    
    def test_creates_plans_directory(self):
        """Test plans directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plans_dir = Path(tmpdir) / "test_plans"
            mode = PlanMode(output_dir=str(plans_dir))
            
            assert plans_dir.exists()
            assert plans_dir.is_dir()
    
    def test_initializes_session_stats(self):
        """Test session statistics are initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            assert mode.session_stats == {
                "plans_generated": 0,
                "plan_files": [],
            }


class TestPlanModeActivationDeactivation:
    """Test Plan mode activation and deactivation."""
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_activate_sets_flag(self, mock_console_class):
        """Test activation sets active flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            mode.activate()
            
            assert mode.active is True
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_activate_resets_session_stats(self, mock_console_class):
        """Test activation resets session statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            mode.session_stats = {
                "plans_generated": 5,
                "plan_files": ["plan1.md", "plan2.md"],
            }
            
            mode.activate()
            
            assert mode.session_stats == {
                "plans_generated": 0,
                "plan_files": [],
            }
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_activate_displays_welcome_panel(self, mock_console_class):
        """Test activation displays welcome message."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            mode.activate()
            
            # Verify console.print was called
            assert mock_console.print.called
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_deactivate_clears_flag(self, mock_console_class):
        """Test deactivation clears active flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            mode.active = True
            
            mode.deactivate()
            
            assert mode.active is False
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_deactivate_displays_summary(self, mock_console_class):
        """Test deactivation displays session summary."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            mode.session_stats = {
                "plans_generated": 2,
                "plan_files": [
                    {"filename": "plan1.md", "plan_type": "refactoring"},
                    {"filename": "plan2.md", "plan_type": "architecture"},
                ],
            }
            
            mode.deactivate()
            
            # Verify console.print was called (summary table)
            assert mock_console.print.called


class TestPlanModePromptIndicator:
    """Test Plan mode prompt indicator."""
    
    def test_returns_clipboard_emoji(self):
        """Test prompt indicator is clipboard emoji."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            assert mode.get_prompt_indicator() == "📋"


class TestPlanModeSystemPrompt:
    """Test Plan mode system prompt/instruction."""
    
    def test_returns_system_prompt(self):
        """Test system prompt is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            prompt = mode.get_system_prompt()
            
            assert prompt is not None
            assert len(prompt) > 0
            assert "Plan Mode" in prompt
    
    def test_system_prompt_contains_key_concepts(self):
        """Test system prompt contains planning guidance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            prompt = mode.get_system_prompt()
            
            # Check for key planning concepts
            assert "architect" in prompt.lower()
            assert "read-only" in prompt.lower()
            assert "refactoring" in prompt.lower()
            assert "architecture" in prompt.lower()
            assert "risk" in prompt.lower()
            assert "testing" in prompt.lower()
    
    def test_system_prompt_includes_structure_guidance(self):
        """Test system prompt includes plan structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            prompt = mode.get_system_prompt()
            
            # Check for structure sections
            assert "Executive Summary" in prompt
            assert "Current State Analysis" in prompt
            assert "Problem Definition" in prompt
            assert "Implementation Plan" in prompt
            assert "Risk Assessment" in prompt
            assert "Testing Strategy" in prompt


class TestPlanModeReadOnlyEnforcement:
    """Test Plan mode blocks write operations."""
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_blocks_file_edit(self, mock_console_class):
        """Test file edit is blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="file_edit",
                target="src/main.py",
                details={"content": "print('hello')"},
            )
            
            result = mode.handle_action(request)
            
            assert result.success is False
            assert "read-only" in result.message.lower()
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_blocks_delete(self, mock_console_class):
        """Test delete is blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="delete",
                target="old_file.py",
                details={},
            )
            
            result = mode.handle_action(request)
            
            assert result.success is False
            assert "read-only" in result.message.lower()
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_blocks_command(self, mock_console_class):
        """Test command execution is blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="command",
                target="rm -rf /tmp/test",
                details={},
            )
            
            result = mode.handle_action(request)
            
            assert result.success is False
            assert "read-only" in result.message.lower()
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_blocks_write_file(self, mock_console_class):
        """Test write_file is blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="write_file",
                target="new_file.txt",
                details={"content": "test"},
            )
            
            result = mode.handle_action(request)
            
            assert result.success is False
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_displays_read_only_message(self, mock_console_class):
        """Test read-only message is displayed."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="file_edit",
                target="test.py",
                details={},
            )
            
            mode.handle_action(request)
            
            # Verify console.print was called (read-only panel)
            assert mock_console.print.called


class TestPlanModeReadOperations:
    """Test Plan mode allows read operations."""
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_allows_read_file(self, mock_console_class):
        """Test read_file is allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="read_file",
                target="README.md",
                details={},
            )
            
            result = mode.handle_action(request)
            
            assert result.success is True
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_allows_analyze(self, mock_console_class):
        """Test analyze is allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="analyze",
                target="src/",
                details={},
            )
            
            result = mode.handle_action(request)
            
            assert result.success is True
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_allows_list_files(self, mock_console_class):
        """Test list_files is allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="list_files",
                target=".",
                details={},
            )
            
            result = mode.handle_action(request)
            
            assert result.success is True


class TestPlanModePlanGeneration:
    """Test Plan mode plan generation."""
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_generates_plan_file(self, mock_console_class):
        """Test plan file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="generate_plan",
                target="",
                details={
                    "title": "Test Plan",
                    "content": "This is a test plan.",
                    "plan_type": "refactoring",
                },
            )
            
            result = mode.handle_action(request)
            
            assert result.success is True
            assert "filepath" in result.details
            
            # Verify file exists
            filepath = Path(result.details["filepath"])
            assert filepath.exists()
            assert filepath.suffix == ".md"
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_plan_contains_metadata(self, mock_console_class):
        """Test plan file contains metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="generate_plan",
                target="",
                details={
                    "title": "Architecture Plan",
                    "content": "System architecture details.",
                    "plan_type": "architecture",
                },
            )
            
            result = mode.handle_action(request)
            filepath = Path(result.details["filepath"])
            content = filepath.read_text(encoding="utf-8")
            
            assert "Architecture Plan" in content
            assert "architecture" in content.lower()
            assert "Generated:" in content
            assert "Plan Mode" in content
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_sanitizes_filename(self, mock_console_class):
        """Test filename is sanitized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="generate_plan",
                target="",
                details={
                    "title": "Plan with Special Ch@rs! & Spaces",
                    "content": "Content",
                    "plan_type": "general",
                },
            )
            
            result = mode.handle_action(request)
            filename = result.details["filename"]
            
            # Filename should not contain special characters
            assert "@" not in filename
            assert "!" not in filename
            assert " " not in filename or "_" in filename
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_increments_plans_generated(self, mock_console_class):
        """Test plans_generated counter increments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            for i in range(3):
                request = ActionRequest(
                    action_type="generate_plan",
                    target="",
                    details={
                        "title": f"Plan {i}",
                        "content": "Content",
                        "plan_type": "general",
                    },
                )
                mode.handle_action(request)
            
            assert mode.session_stats["plans_generated"] == 3
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_tracks_plan_files(self, mock_console_class):
        """Test plan files are tracked in session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="generate_plan",
                target="",
                details={
                    "title": "Test",
                    "content": "Content",
                    "plan_type": "bugfix",
                },
            )
            
            mode.handle_action(request)
            
            assert len(mode.session_stats["plan_files"]) == 1
            plan_info = mode.session_stats["plan_files"][0]
            assert "filename" in plan_info
            assert plan_info["plan_type"] == "bugfix"


class TestPlanModeSessionStatistics:
    """Test Plan mode session statistics tracking."""
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_get_session_stats_returns_copy(self, mock_console_class):
        """Test get_session_stats returns a copy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            mode.session_stats = {
                "plans_generated": 5,
                "plan_files": ["plan1.md"],
            }
            
            stats = mode.get_session_stats()
            
            # Modify returned stats
            stats["plans_generated"] = 999
            
            # Original should be unchanged
            assert mode.session_stats["plans_generated"] == 5


class TestPlanModeEdgeCases:
    """Test Plan mode edge cases and error handling."""
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_handles_empty_title(self, mock_console_class):
        """Test handling of empty title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="generate_plan",
                target="",
                details={
                    "title": "",
                    "content": "Content",
                    "plan_type": "general",
                },
            )
            
            result = mode.handle_action(request)
            
            # Should still succeed with default title
            assert result.success is True
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_handles_missing_content(self, mock_console_class):
        """Test handling of missing content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="generate_plan",
                target="",
                details={
                    "title": "Test",
                    "plan_type": "general",
                },
            )
            
            result = mode.handle_action(request)
            
            # Should handle gracefully
            assert result.success is True
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_handles_none_details(self, mock_console_class):
        """Test handling of None details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="generate_plan",
                target="",
                details=None,
            )
            
            result = mode.handle_action(request)
            
            # Should handle with defaults
            assert result.success is True
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_handles_unsupported_action(self, mock_console_class):
        """Test handling of unsupported action type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            
            request = ActionRequest(
                action_type="unsupported_action",
                target="",
                details={},
            )
            
            result = mode.handle_action(request)
            
            assert result.success is False
            assert "not supported" in result.message.lower()


class TestPlanModeIntegration:
    """Integration tests for Plan mode."""
    
    @patch("quirkllm.modes.plan_mode.Console")
    def test_end_to_end_plan_generation(self, mock_console_class):
        """Test complete plan generation workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mode = PlanMode(output_dir=tmpdir)
            mode.activate()
            
            # Generate plan
            request = ActionRequest(
                action_type="generate_plan",
                target="",
                details={
                    "title": "Refactoring Plan",
                    "content": "## Steps\n1. Extract method\n2. Rename variables",
                    "plan_type": "refactoring",
                },
            )
            
            result = mode.handle_action(request)
            
            assert result.success is True
            assert mode.session_stats["plans_generated"] == 1
            
            # Verify file exists and has content
            filepath = Path(result.details["filepath"])
            assert filepath.exists()
            content = filepath.read_text()
            assert "Refactoring Plan" in content
            assert "Extract method" in content
            
            mode.deactivate()
