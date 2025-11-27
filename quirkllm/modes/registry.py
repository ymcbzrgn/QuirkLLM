"""
Mode registry and factory for QuirkLLM modes.

Provides centralized mode management:
- Register mode classes
- Create mode instances
- Track current active mode
- Switch between modes

Usage:
    registry = get_registry()
    registry.register("chat", ChatMode)
    
    mode = registry.create_mode(ModeType.CHAT)
    registry.set_current(mode)
    
    current = registry.get_current()
"""

from typing import Dict, Optional, Type
from quirkllm.modes.base import ModeBase, ModeType, ModeConfig


class ModeRegistry:
    """
    Registry for managing QuirkLLM modes.
    
    Singleton pattern ensures single registry instance across application.
    Tracks registered mode classes, creates instances, and maintains
    current active mode state.
    """
    
    _instance: Optional['ModeRegistry'] = None
    
    def __new__(cls) -> 'ModeRegistry':
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize registry on first creation."""
        if self._initialized:
            return
        
        self._modes: Dict[ModeType, Type[ModeBase]] = {}
        self._current_mode: Optional[ModeBase] = None
        self._mode_history: list[ModeType] = []
        self._initialized = True
    
    def register(self, mode_type: ModeType, mode_class: Type[ModeBase]) -> None:
        """
        Register a mode class.
        
        Args:
            mode_type: Type of mode to register
            mode_class: Class that implements ModeBase
            
        Raises:
            ValueError: If mode_type already registered
            TypeError: If mode_class doesn't inherit from ModeBase
        """
        if mode_type in self._modes:
            raise ValueError(f"Mode {mode_type.value} already registered")
        
        if not issubclass(mode_class, ModeBase):
            raise TypeError(f"{mode_class.__name__} must inherit from ModeBase")
        
        self._modes[mode_type] = mode_class
    
    def unregister(self, mode_type: ModeType) -> None:
        """
        Unregister a mode class.
        
        Args:
            mode_type: Type of mode to unregister
            
        Raises:
            ValueError: If mode_type not registered
        """
        if mode_type not in self._modes:
            raise ValueError(f"Mode {mode_type.value} not registered")
        
        # Deactivate if currently active
        if self._current_mode and self._current_mode.mode_type == mode_type:
            self._current_mode.deactivate()
            self._current_mode = None
        
        del self._modes[mode_type]
    
    def is_registered(self, mode_type: ModeType) -> bool:
        """
        Check if a mode type is registered.
        
        Args:
            mode_type: Type to check
            
        Returns:
            True if registered, False otherwise
        """
        return mode_type in self._modes
    
    def create_mode(
        self, 
        mode_type: ModeType, 
        config: Optional[ModeConfig] = None
    ) -> ModeBase:
        """
        Create a new mode instance.
        
        Args:
            mode_type: Type of mode to create
            config: Optional configuration for the mode
            
        Returns:
            New mode instance
            
        Raises:
            ValueError: If mode_type not registered
        """
        if mode_type not in self._modes:
            raise ValueError(f"Mode {mode_type.value} not registered")

        mode_class = self._modes[mode_type]
        # Try creating with (mode_type, config), fall back to no params
        try:
            return mode_class(mode_type, config)
        except TypeError:
            return mode_class()
    
    def set_current(self, mode: ModeBase) -> None:
        """
        Set the current active mode.
        
        Deactivates previous mode if any, activates new mode,
        and records transition in history.
        
        Args:
            mode: Mode instance to activate
        """
        # Deactivate current mode
        if self._current_mode and self._current_mode.is_active:
            self._current_mode.deactivate()
        
        # Activate new mode
        self._current_mode = mode
        mode.activate()
        
        # Record in history
        self._mode_history.append(mode.mode_type)
    
    def get_current(self) -> Optional[ModeBase]:
        """
        Get the current active mode.
        
        Returns:
            Current mode instance, or None if no mode active
        """
        return self._current_mode
    
    def switch_mode(
        self, 
        mode_type: ModeType, 
        config: Optional[ModeConfig] = None
    ) -> ModeBase:
        """
        Switch to a different mode.
        
        Convenience method that creates and activates a mode in one call.
        
        Args:
            mode_type: Type of mode to switch to
            config: Optional configuration for new mode
            
        Returns:
            Newly activated mode instance
        """
        mode = self.create_mode(mode_type, config)
        self.set_current(mode)
        return mode
    
    def get_mode_history(self) -> list[ModeType]:
        """
        Get history of mode transitions.
        
        Returns:
            List of mode types in order of activation
        """
        return self._mode_history.copy()
    
    def get_registered_modes(self) -> list[ModeType]:
        """
        Get list of all registered mode types.
        
        Returns:
            List of registered mode types
        """
        return list(self._modes.keys())
    
    def clear_history(self) -> None:
        """Clear mode transition history."""
        self._mode_history.clear()
    
    def reset(self) -> None:
        """
        Reset registry to initial state.
        
        Deactivates current mode, clears all registrations and history.
        Primarily for testing purposes.
        """
        # Deactivate current mode
        if self._current_mode and self._current_mode.is_active:
            self._current_mode.deactivate()
        
        # Clear state
        self._modes.clear()
        self._current_mode = None
        self._mode_history.clear()


# Global registry instance
_global_registry: Optional[ModeRegistry] = None


def get_registry() -> ModeRegistry:
    """
    Get the global mode registry instance.
    
    Returns:
        Singleton ModeRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ModeRegistry()
    return _global_registry
