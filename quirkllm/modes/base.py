"""
Base classes and dataclasses for QuirkLLM mode system.

The mode system provides four operational modes with different levels
of automation and confirmation:
- Chat: Safe mode with user confirmations (default)
- YAMI: "Yes, And Make It" - auto-accept with safety validation
- Plan: Read-only planning mode that generates plan documents
- Ghost: Background file watcher that analyzes changes

Key Components:
- ModeType: Enum defining the four operational modes
- ModeConfig: Configuration for mode behavior
- ActionRequest: Request to perform an action (edit, command, etc.)
- ActionResult: Result of an action execution
- ModeBase: Abstract base class that all modes must inherit from
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ModeType(Enum):
    """
    Enumeration of QuirkLLM operational modes.
    
    Each mode defines different behavior for action confirmation,
    file editing, and automation levels.
    """
    
    CHAT = "chat"  # Safe mode with user confirmations (default)
    YAMI = "yami"  # Auto-accept with safety validation
    PLAN = "plan"  # Read-only planning mode
    GHOST = "ghost"  # Background file watcher


@dataclass
class ModeConfig:
    """
    Configuration for a mode's behavior.
    
    Attributes:
        auto_confirm: Whether to auto-confirm actions without user prompt
        allow_file_edits: Whether file editing operations are permitted
        allow_destructive: Whether destructive operations are allowed
        background_watch: Whether mode runs as background file watcher
        confirm_high_risk: Whether to confirm high-risk operations even in auto mode
        diff_display: Whether to show diffs before applying changes
        plan_output_dir: Directory for plan mode output files
        watch_patterns: File patterns to watch in ghost mode
        watch_debounce_ms: Debounce delay for file watch events (milliseconds)
    """
    
    auto_confirm: bool = False
    allow_file_edits: bool = True
    allow_destructive: bool = False
    background_watch: bool = False
    confirm_high_risk: bool = True
    diff_display: bool = True
    plan_output_dir: str = ".quirkllm/plans"
    watch_patterns: List[str] = field(default_factory=lambda: ["**/*.py", "**/*.js", "**/*.ts"])
    watch_debounce_ms: int = 500


@dataclass
class ActionRequest:
    """
    Request to perform an action in the current mode.
    
    Actions can be file edits, terminal commands, API calls, etc.
    Each action has a risk level and target specification.
    
    Attributes:
        action_type: Type of action (edit_file, run_command, delete_file, etc.)
        target: Target of the action (file path, command, etc.)
        details: Additional details about the action (edit content, args, etc.)
        risk_level: Risk assessment (safe, low, medium, high, critical)
        requires_confirmation: Whether this action should always be confirmed
        metadata: Additional metadata for the action
    """
    
    action_type: str
    target: str
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "safe"  # safe, low, medium, high, critical
    requires_confirmation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_high_risk(self) -> bool:
        """Check if action is high risk or critical."""
        return self.risk_level in ("high", "critical")
    
    def is_critical(self) -> bool:
        """Check if action is critical risk level."""
        return self.risk_level == "critical"


@dataclass
class ActionResult:
    """
    Result of an action execution.
    
    Captures whether the action succeeded, any messages or errors,
    and additional details about the execution.
    
    Attributes:
        success: Whether the action completed successfully
        message: Human-readable result message
        details: Additional details about the execution
        modified_files: List of files that were modified
        errors: List of errors encountered
        warnings: List of warnings generated
    """
    
    success: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    modified_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def has_errors(self) -> bool:
        """Check if result contains errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if result contains warnings."""
        return len(self.warnings) > 0


class ModeBase(ABC):
    """
    Abstract base class for all QuirkLLM modes.
    
    All modes must implement:
    - activate(): Called when switching to this mode
    - deactivate(): Called when switching away from this mode
    - handle_action(): Process an action request
    - get_prompt_indicator(): Return the prompt indicator for this mode
    
    The base class provides:
    - Mode configuration management
    - State tracking (active/inactive)
    - Common utilities
    """
    
    def __init__(self, mode_type: ModeType, config: Optional[ModeConfig] = None):
        """
        Initialize the mode.
        
        Args:
            mode_type: The type of this mode
            config: Optional configuration, uses defaults if not provided
        """
        self.mode_type = mode_type
        self.config = config or ModeConfig()
        self._active = False
    
    @property
    def is_active(self) -> bool:
        """Check if mode is currently active."""
        return self._active

    @property
    def active(self) -> bool:
        """Alias for is_active - returns True if mode is active."""
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        """Set the active state."""
        self._active = value

    @abstractmethod
    def activate(self) -> None:
        """
        Activate this mode.
        
        Called when switching to this mode. Implementations should:
        - Set up any required state
        - Initialize watchers or background processes
        - Display activation message
        """
        self._active = True
    
    @abstractmethod
    def deactivate(self) -> None:
        """
        Deactivate this mode.
        
        Called when switching away from this mode. Implementations should:
        - Clean up state
        - Stop watchers or background processes
        - Save any pending work
        """
        self._active = False
    
    @abstractmethod
    def handle_action(self, action: ActionRequest) -> ActionResult:
        """
        Handle an action request in this mode.
        
        Process the action according to mode-specific behavior:
        - Chat: Confirm with user before executing
        - YAMI: Auto-execute with safety validation
        - Plan: Generate plan instead of executing
        - Ghost: Analyze impact without executing
        
        Args:
            action: The action request to handle
            
        Returns:
            ActionResult with execution outcome
        """
        pass
    
    @abstractmethod
    def get_prompt_indicator(self) -> str:
        """
        Get the prompt indicator for this mode.
        
        Returns a string to display in the REPL prompt:
        - Chat: "💬"
        - YAMI: "⚡"
        - Plan: "📋"
        - Ghost: "👻"
        
        Returns:
            Emoji or string indicator for the prompt
        """
        pass
    
    def validate_action(self, action: ActionRequest) -> tuple[bool, Optional[str]]:
        """
        Validate an action against mode configuration.
        
        Args:
            action: Action to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if mode allows file edits
        if action.action_type in ("edit_file", "delete_file", "create_file"):
            if not self.config.allow_file_edits:
                return False, f"File operations not allowed in {self.mode_type.value} mode"
        
        # Check if mode allows destructive operations
        if action.is_critical():
            if not self.config.allow_destructive:
                return False, f"Destructive operations not allowed in {self.mode_type.value} mode"
        
        return True, None
    
    def __repr__(self) -> str:
        """String representation of mode."""
        return f"{self.__class__.__name__}(type={self.mode_type.value}, active={self._active})"
