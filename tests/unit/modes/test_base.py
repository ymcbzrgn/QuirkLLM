"""
Tests for base mode classes and dataclasses.

Tests cover:
- ModeType enum values
- ModeConfig dataclass with defaults
- ActionRequest with risk levels
- ActionResult with error/warning tracking
- ModeBase abstract class behavior
"""

import pytest
from quirkllm.modes.base import (
    ModeType,
    ModeConfig,
    ActionRequest,
    ActionResult,
    ModeBase,
)


class TestModeType:
    """Test ModeType enum."""
    
    def test_mode_type_values(self):
        """Test that all mode types have correct string values."""
        assert ModeType.CHAT.value == "chat"
        assert ModeType.YAMI.value == "yami"
        assert ModeType.PLAN.value == "plan"
        assert ModeType.GHOST.value == "ghost"
    
    def test_mode_type_enumeration(self):
        """Test that all expected modes are defined."""
        mode_values = {mode.value for mode in ModeType}
        assert mode_values == {"chat", "yami", "plan", "ghost"}


class TestModeConfig:
    """Test ModeConfig dataclass."""
    
    def test_mode_config_defaults(self):
        """Test that ModeConfig has correct default values."""
        config = ModeConfig()
        
        assert config.auto_confirm is False
        assert config.allow_file_edits is True
        assert config.allow_destructive is False
        assert config.background_watch is False
        assert config.confirm_high_risk is True
        assert config.diff_display is True
        assert config.plan_output_dir == ".quirkllm/plans"
        assert config.watch_patterns == ["**/*.py", "**/*.js", "**/*.ts"]
        assert config.watch_debounce_ms == 500
    
    def test_mode_config_custom_values(self):
        """Test ModeConfig with custom values."""
        config = ModeConfig(
            auto_confirm=True,
            allow_destructive=True,
            plan_output_dir="/tmp/plans",
            watch_patterns=["**/*.md"],
            watch_debounce_ms=1000,
        )
        
        assert config.auto_confirm is True
        assert config.allow_destructive is True
        assert config.plan_output_dir == "/tmp/plans"
        assert config.watch_patterns == ["**/*.md"]
        assert config.watch_debounce_ms == 1000


class TestActionRequest:
    """Test ActionRequest dataclass."""
    
    def test_action_request_basic(self):
        """Test basic ActionRequest creation."""
        action = ActionRequest(
            action_type="edit_file",
            target="test.py",
            details={"content": "new code"},
        )
        
        assert action.action_type == "edit_file"
        assert action.target == "test.py"
        assert action.details == {"content": "new code"}
        assert action.risk_level == "safe"
        assert action.requires_confirmation is False
    
    def test_action_request_high_risk(self):
        """Test high risk action detection."""
        action = ActionRequest(
            action_type="delete_file",
            target="important.py",
            risk_level="high",
        )
        
        assert action.is_high_risk() is True
        assert action.is_critical() is False
    
    def test_action_request_critical_risk(self):
        """Test critical risk action detection."""
        action = ActionRequest(
            action_type="run_command",
            target="rm -rf /",
            risk_level="critical",
        )
        
        assert action.is_high_risk() is True
        assert action.is_critical() is True
    
    def test_action_request_safe_risk(self):
        """Test safe risk level."""
        action = ActionRequest(
            action_type="read_file",
            target="test.py",
            risk_level="safe",
        )
        
        assert action.is_high_risk() is False
        assert action.is_critical() is False


class TestActionResult:
    """Test ActionResult dataclass."""
    
    def test_action_result_success(self):
        """Test successful action result."""
        result = ActionResult(
            success=True,
            message="Operation completed",
            modified_files=["test.py"],
        )
        
        assert result.success is True
        assert result.message == "Operation completed"
        assert result.modified_files == ["test.py"]
        assert result.has_errors() is False
        assert result.has_warnings() is False
    
    def test_action_result_with_errors(self):
        """Test action result with errors."""
        result = ActionResult(
            success=False,
            message="Operation failed",
            errors=["File not found", "Permission denied"],
        )
        
        assert result.success is False
        assert result.has_errors() is True
        assert len(result.errors) == 2
        assert "File not found" in result.errors
    
    def test_action_result_with_warnings(self):
        """Test action result with warnings."""
        result = ActionResult(
            success=True,
            message="Completed with warnings",
            warnings=["Deprecated API used"],
        )
        
        assert result.success is True
        assert result.has_warnings() is True
        assert len(result.warnings) == 1
    
    def test_action_result_defaults(self):
        """Test ActionResult default values."""
        result = ActionResult(success=True, message="Done")
        
        assert result.details == {}
        assert result.modified_files == []
        assert result.errors == []
        assert result.warnings == []


class TestModeBase:
    """Test ModeBase abstract class."""
    
    def test_mode_base_cannot_instantiate(self):
        """Test that ModeBase cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ModeBase(ModeType.CHAT)
    
    def test_mode_base_requires_implementation(self):
        """Test that concrete modes must implement abstract methods."""
        
        # Missing implementations
        class IncompleteMode(ModeBase):
            pass
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteMode(ModeType.CHAT)
    
    def test_mode_base_concrete_implementation(self):
        """Test that concrete mode with all methods works."""
        
        class ConcreteMode(ModeBase):
            def activate(self):
                super().activate()
            
            def deactivate(self):
                super().deactivate()
            
            def handle_action(self, action):
                return ActionResult(success=True, message="Handled")
            
            def get_prompt_indicator(self):
                return "🔧"
        
        mode = ConcreteMode(ModeType.CHAT)
        assert mode.mode_type == ModeType.CHAT
        assert mode.is_active is False
        
        mode.activate()
        assert mode.is_active is True
        
        mode.deactivate()
        assert mode.is_active is False
    
    def test_mode_base_validate_action_file_edits(self):
        """Test action validation for file edits."""
        
        class TestMode(ModeBase):
            def activate(self):
                super().activate()
            
            def deactivate(self):
                super().deactivate()
            
            def handle_action(self, action):
                return ActionResult(success=True, message="OK")
            
            def get_prompt_indicator(self):
                return "🔧"
        
        # Mode that disallows file edits
        config = ModeConfig(allow_file_edits=False)
        mode = TestMode(ModeType.PLAN, config)
        
        action = ActionRequest(action_type="edit_file", target="test.py")
        is_valid, error = mode.validate_action(action)
        
        assert is_valid is False
        assert "File operations not allowed" in error
    
    def test_mode_base_validate_action_destructive(self):
        """Test action validation for destructive operations."""
        
        class TestMode(ModeBase):
            def activate(self):
                super().activate()
            
            def deactivate(self):
                super().deactivate()
            
            def handle_action(self, action):
                return ActionResult(success=True, message="OK")
            
            def get_prompt_indicator(self):
                return "🔧"
        
        # Mode that disallows destructive ops
        config = ModeConfig(allow_destructive=False)
        mode = TestMode(ModeType.CHAT, config)
        
        action = ActionRequest(
            action_type="delete_file",
            target="important.py",
            risk_level="critical",
        )
        is_valid, error = mode.validate_action(action)
        
        assert is_valid is False
        assert "Destructive operations not allowed" in error
    
    def test_mode_base_repr(self):
        """Test string representation of mode."""
        
        class TestMode(ModeBase):
            def activate(self):
                super().activate()
            
            def deactivate(self):
                super().deactivate()
            
            def handle_action(self, action):
                return ActionResult(success=True, message="OK")
            
            def get_prompt_indicator(self):
                return "🔧"
        
        mode = TestMode(ModeType.CHAT)
        repr_str = repr(mode)
        
        assert "TestMode" in repr_str
        assert "type=chat" in repr_str
        assert "active=False" in repr_str
