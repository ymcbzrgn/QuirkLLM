"""
Tests for mode registry and factory.

Tests cover:
- Mode registration and unregistration
- Mode creation via factory
- Current mode tracking
- Mode switching
- History tracking
- Singleton behavior
"""

import pytest
from quirkllm.modes.base import ModeType, ModeConfig, ModeBase, ActionRequest, ActionResult
from quirkllm.modes.registry import ModeRegistry, get_registry


# Test mode implementations
class TestChatMode(ModeBase):
    """Mock Chat mode for testing."""

    def __init__(self, mode_type=None, config=None):
        super().__init__(mode_type or ModeType.CHAT, config)

    def activate(self):
        super().activate()
        self.activated_count = getattr(self, 'activated_count', 0) + 1

    def deactivate(self):
        super().deactivate()
        self.deactivated_count = getattr(self, 'deactivated_count', 0) + 1

    def handle_action(self, action: ActionRequest) -> ActionResult:
        return ActionResult(success=True, message="Chat handled action")

    def get_prompt_indicator(self) -> str:
        return "💬"


class TestYAMIMode(ModeBase):
    """Mock YAMI mode for testing."""

    def __init__(self, mode_type=None, config=None):
        super().__init__(mode_type or ModeType.YAMI, config)

    def activate(self):
        super().activate()

    def deactivate(self):
        super().deactivate()

    def handle_action(self, action: ActionRequest) -> ActionResult:
        return ActionResult(success=True, message="YAMI handled action")

    def get_prompt_indicator(self) -> str:
        return "⚡"


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    reg = ModeRegistry()
    reg.reset()  # Clear any previous state
    return reg


class TestModeRegistry:
    """Test ModeRegistry singleton and basic operations."""
    
    def test_registry_singleton(self):
        """Test that ModeRegistry is a singleton."""
        reg1 = ModeRegistry()
        reg2 = ModeRegistry()
        
        assert reg1 is reg2
    
    def test_get_registry(self):
        """Test get_registry() returns singleton."""
        reg1 = get_registry()
        reg2 = get_registry()
        
        assert reg1 is reg2
        assert isinstance(reg1, ModeRegistry)


class TestModeRegistration:
    """Test mode registration and unregistration."""
    
    def test_register_mode(self, registry):
        """Test registering a mode class."""
        registry.register(ModeType.CHAT, TestChatMode)
        
        assert registry.is_registered(ModeType.CHAT)
    
    def test_register_multiple_modes(self, registry):
        """Test registering multiple mode classes."""
        registry.register(ModeType.CHAT, TestChatMode)
        registry.register(ModeType.YAMI, TestYAMIMode)
        
        assert registry.is_registered(ModeType.CHAT)
        assert registry.is_registered(ModeType.YAMI)
        assert not registry.is_registered(ModeType.PLAN)
    
    def test_register_duplicate_mode_raises_error(self, registry):
        """Test that registering same mode twice raises error."""
        registry.register(ModeType.CHAT, TestChatMode)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register(ModeType.CHAT, TestChatMode)
    
    def test_register_non_mode_class_raises_error(self, registry):
        """Test that registering non-ModeBase class raises error."""
        
        class NotAMode:
            pass
        
        with pytest.raises(TypeError, match="must inherit from ModeBase"):
            registry.register(ModeType.CHAT, NotAMode)
    
    def test_unregister_mode(self, registry):
        """Test unregistering a mode."""
        registry.register(ModeType.CHAT, TestChatMode)
        assert registry.is_registered(ModeType.CHAT)
        
        registry.unregister(ModeType.CHAT)
        assert not registry.is_registered(ModeType.CHAT)
    
    def test_unregister_nonexistent_mode_raises_error(self, registry):
        """Test that unregistering non-registered mode raises error."""
        with pytest.raises(ValueError, match="not registered"):
            registry.unregister(ModeType.CHAT)
    
    def test_unregister_active_mode(self, registry):
        """Test that unregistering active mode deactivates it."""
        registry.register(ModeType.CHAT, TestChatMode)
        mode = registry.create_mode(ModeType.CHAT)
        registry.set_current(mode)
        
        assert mode.is_active
        
        registry.unregister(ModeType.CHAT)
        
        assert not mode.is_active
        assert registry.get_current() is None


class TestModeCreation:
    """Test mode instance creation."""
    
    def test_create_mode(self, registry):
        """Test creating a mode instance."""
        registry.register(ModeType.CHAT, TestChatMode)
        
        mode = registry.create_mode(ModeType.CHAT)
        
        assert isinstance(mode, TestChatMode)
        assert mode.mode_type == ModeType.CHAT
        assert not mode.is_active
    
    def test_create_mode_with_config(self, registry):
        """Test creating mode with custom config."""
        registry.register(ModeType.CHAT, TestChatMode)
        config = ModeConfig(auto_confirm=True)
        
        mode = registry.create_mode(ModeType.CHAT, config)
        
        assert mode.config.auto_confirm is True
    
    def test_create_unregistered_mode_raises_error(self, registry):
        """Test that creating unregistered mode raises error."""
        with pytest.raises(ValueError, match="not registered"):
            registry.create_mode(ModeType.CHAT)
    
    def test_create_multiple_instances(self, registry):
        """Test creating multiple instances of same mode."""
        registry.register(ModeType.CHAT, TestChatMode)
        
        mode1 = registry.create_mode(ModeType.CHAT)
        mode2 = registry.create_mode(ModeType.CHAT)
        
        assert mode1 is not mode2
        assert isinstance(mode1, TestChatMode)
        assert isinstance(mode2, TestChatMode)


class TestCurrentMode:
    """Test current mode tracking."""
    
    def test_set_current_mode(self, registry):
        """Test setting current mode."""
        registry.register(ModeType.CHAT, TestChatMode)
        mode = registry.create_mode(ModeType.CHAT)
        
        registry.set_current(mode)
        
        assert registry.get_current() is mode
        assert mode.is_active
    
    def test_set_current_deactivates_previous(self, registry):
        """Test that setting new mode deactivates previous."""
        registry.register(ModeType.CHAT, TestChatMode)
        registry.register(ModeType.YAMI, TestYAMIMode)
        
        mode1 = registry.create_mode(ModeType.CHAT)
        mode2 = registry.create_mode(ModeType.YAMI)
        
        registry.set_current(mode1)
        assert mode1.is_active
        
        registry.set_current(mode2)
        assert not mode1.is_active
        assert mode2.is_active
    
    def test_get_current_none_initially(self, registry):
        """Test that get_current returns None initially."""
        assert registry.get_current() is None


class TestModeSwitching:
    """Test mode switching functionality."""
    
    def test_switch_mode(self, registry):
        """Test switch_mode convenience method."""
        registry.register(ModeType.CHAT, TestChatMode)
        
        mode = registry.switch_mode(ModeType.CHAT)
        
        assert isinstance(mode, TestChatMode)
        assert mode.is_active
        assert registry.get_current() is mode
    
    def test_switch_mode_with_config(self, registry):
        """Test switching with custom config."""
        registry.register(ModeType.CHAT, TestChatMode)
        config = ModeConfig(diff_display=False)
        
        mode = registry.switch_mode(ModeType.CHAT, config)
        
        assert mode.config.diff_display is False
    
    def test_switch_between_modes(self, registry):
        """Test switching between different modes."""
        registry.register(ModeType.CHAT, TestChatMode)
        registry.register(ModeType.YAMI, TestYAMIMode)
        
        chat_mode = registry.switch_mode(ModeType.CHAT)
        assert isinstance(registry.get_current(), TestChatMode)
        
        yami_mode = registry.switch_mode(ModeType.YAMI)
        assert isinstance(registry.get_current(), TestYAMIMode)
        assert not chat_mode.is_active
        assert yami_mode.is_active


class TestModeHistory:
    """Test mode transition history tracking."""
    
    def test_mode_history_tracks_transitions(self, registry):
        """Test that mode history tracks transitions."""
        registry.register(ModeType.CHAT, TestChatMode)
        registry.register(ModeType.YAMI, TestYAMIMode)
        
        registry.switch_mode(ModeType.CHAT)
        registry.switch_mode(ModeType.YAMI)
        registry.switch_mode(ModeType.CHAT)
        
        history = registry.get_mode_history()
        
        assert len(history) == 3
        assert history == [ModeType.CHAT, ModeType.YAMI, ModeType.CHAT]
    
    def test_clear_history(self, registry):
        """Test clearing mode history."""
        registry.register(ModeType.CHAT, TestChatMode)
        
        registry.switch_mode(ModeType.CHAT)
        assert len(registry.get_mode_history()) == 1
        
        registry.clear_history()
        assert len(registry.get_mode_history()) == 0
    
    def test_get_mode_history_returns_copy(self, registry):
        """Test that get_mode_history returns a copy."""
        registry.register(ModeType.CHAT, TestChatMode)
        registry.switch_mode(ModeType.CHAT)
        
        history1 = registry.get_mode_history()
        history2 = registry.get_mode_history()
        
        assert history1 == history2
        assert history1 is not history2


class TestRegistryUtilities:
    """Test registry utility methods."""
    
    def test_get_registered_modes(self, registry):
        """Test getting list of registered modes."""
        assert registry.get_registered_modes() == []
        
        registry.register(ModeType.CHAT, TestChatMode)
        registry.register(ModeType.YAMI, TestYAMIMode)
        
        registered = registry.get_registered_modes()
        
        assert len(registered) == 2
        assert ModeType.CHAT in registered
        assert ModeType.YAMI in registered
    
    def test_reset_registry(self, registry):
        """Test resetting registry to initial state."""
        registry.register(ModeType.CHAT, TestChatMode)
        mode = registry.switch_mode(ModeType.CHAT)
        
        assert mode.is_active
        assert registry.is_registered(ModeType.CHAT)
        assert len(registry.get_mode_history()) == 1
        
        registry.reset()
        
        assert not mode.is_active
        assert not registry.is_registered(ModeType.CHAT)
        assert len(registry.get_mode_history()) == 0
        assert registry.get_current() is None
