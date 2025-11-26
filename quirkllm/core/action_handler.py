"""
Action Handler - Coordinates mode-aware actions and file operations.

The ActionHandler is the central coordinator between user actions, modes,
and file operations. It routes actions through the appropriate mode,
validates them with SafetyChecker, and executes file operations via FileManager.

Architecture:
    REPL → ActionHandler → Mode → ActionHandler → FileManager
                ↓
         SafetyChecker

Key Responsibilities:
- Route actions to current mode
- Validate actions based on mode and safety rules
- Execute file operations (read, write, edit, delete)
- Execute shell commands safely
- Track action history and statistics
- Aggregate results and errors

Usage:
    handler = ActionHandler(current_mode, file_manager, safety_checker, config)
    result = handler.handle_action(ActionRequest(...))
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess

from quirkllm.modes import (
    ModeBase,
    ActionRequest,
    ActionResult,
    SafetyChecker,
)
from quirkllm.core.config import Config


class ActionHandler:
    """Coordinates mode-aware actions and file operations.
    
    The ActionHandler acts as a bridge between the REPL, modes, and file
    operations. It ensures that all actions are validated according to the
    current mode's rules and safety constraints.
    
    Attributes:
        current_mode: Active mode (Chat, YAMI, Plan, Ghost)
        file_manager: File operation manager (Phase 3)
        safety_checker: Safety validation engine
        config: User configuration
        action_history: List of executed actions
        action_stats: Statistics about actions
    """
    
    def __init__(
        self,
        current_mode: ModeBase,
        file_manager: Any | None = None,
        safety_checker: SafetyChecker | None = None,
        config: Config | None = None,
    ):
        """Initialize ActionHandler.
        
        Args:
            current_mode: Current active mode
            file_manager: File operations manager (can be None for now)
            safety_checker: Safety validation engine
            config: User configuration
        """
        self.current_mode = current_mode
        self.file_manager = file_manager
        self.safety_checker = safety_checker or SafetyChecker()
        self.config = config or Config()
        
        # Action tracking
        self.action_history: List[Dict[str, Any]] = []
        self.action_stats = {
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "blocked_actions": 0,
            "actions_by_type": {},
            "actions_by_mode": {},
        }
    
    def handle_action(self, request: ActionRequest) -> ActionResult:
        """Handle an action request through the current mode.
        
        This is the main entry point for all actions. It:
        1. Validates the action with SafetyChecker
        2. Routes to current mode's handle_action()
        3. Executes file operations if needed
        4. Tracks the action in history
        5. Updates statistics
        
        Args:
            request: Action request to handle
            
        Returns:
            ActionResult with success/failure info
        """
        # Track action
        self.action_stats["total_actions"] += 1
        
        # Track by type
        action_type = request.action_type
        if action_type not in self.action_stats["actions_by_type"]:
            self.action_stats["actions_by_type"][action_type] = 0
        self.action_stats["actions_by_type"][action_type] += 1
        
        # Track by mode
        mode_name = self.current_mode.mode_type.value
        if mode_name not in self.action_stats["actions_by_mode"]:
            self.action_stats["actions_by_mode"][mode_name] = 0
        self.action_stats["actions_by_mode"][mode_name] += 1
        
        # Validate with SafetyChecker first
        validation = self._validate_action(request)
        if validation and not validation.is_safe:
            # Critical actions are blocked
            self.action_stats["blocked_actions"] += 1
            reason = ", ".join(validation.blocked_reasons) if validation.blocked_reasons else "Critical risk detected"
            self._record_action(request, False, reason)
            return ActionResult(
                success=False,
                message=f"Action blocked: {reason}",
                details={"risk_score": validation.risk_score, "severity": validation.severity, "patterns": validation.matched_patterns},
            )
        
        # Route to mode's handler
        try:
            result = self.current_mode.handle_action(request)
            
            # Execute actual file operation if mode approves
            if result.success and self._needs_execution(request):
                result = self._execute_operation(request)
            
            # Update stats
            if result.success:
                self.action_stats["successful_actions"] += 1
            else:
                self.action_stats["failed_actions"] += 1
            
            # Record in history
            self._record_action(request, result.success, result.message)
            
            return result
            
        except Exception as e:
            self.action_stats["failed_actions"] += 1
            self._record_action(request, False, str(e))
            return ActionResult(
                success=False,
                message=f"Action failed: {e}",
                details={"error": str(e)},
            )
    
    def _validate_action(self, request: ActionRequest) -> Any:
        """Validate action with SafetyChecker.
        
        Args:
            request: Action to validate
            
        Returns:
            ValidationResult if validation needed, None otherwise
        """
        # Only validate write/delete/command actions
        if request.action_type in ["file_write", "file_edit", "file_delete", "command", "create_file"]:
            return self.safety_checker.validate_action(request)
        return None
    
    def _needs_execution(self, request: ActionRequest) -> bool:
        """Check if action needs actual execution.
        
        Some actions are handled entirely by the mode (e.g., Plan mode
        just generates plans). Others need actual file operations.
        
        Args:
            request: Action request
            
        Returns:
            True if execution needed, False otherwise
        """
        # Read operations are always executed (even in read-only modes)
        if request.action_type in ["file_read", "read_file"]:
            return True
        
        # Write operations need write permission
        if request.action_type in ["file_write", "file_edit", "file_delete", "file_create", "create_file"]:
            return self.current_mode.config.allow_file_edits
        
        # Commands need execution
        if request.action_type == "command":
            return True
        
        return False
    
    def _execute_operation(self, request: ActionRequest) -> ActionResult:
        """Execute the actual file/command operation.
        
        Args:
            request: Action request
            
        Returns:
            ActionResult from operation
        """
        action_type = request.action_type
        
        # Handle action type aliases
        if action_type in ["file_read", "read_file"]:
            return self._execute_file_read(request)
        elif action_type == "file_write":
            return self._execute_file_write(request)
        elif action_type == "file_edit":
            return self._execute_file_edit(request)
        elif action_type == "file_delete":
            return self._execute_file_delete(request)
        elif action_type in ["file_create", "create_file"]:
            return self._execute_file_create(request)
        elif action_type == "command":
            return self._execute_command(request)
        else:
            return ActionResult(
                success=False,
                message=f"Unknown action type: {action_type}",
            )
    
    def _execute_file_read(self, request: ActionRequest) -> ActionResult:
        """Execute file read operation.
        
        Args:
            request: Read request
            
        Returns:
            ActionResult with file contents
        """
        try:
            filepath = Path(request.target)
            
            if not filepath.exists():
                return ActionResult(
                    success=False,
                    message=f"File not found: {filepath}",
                )
            
            if not filepath.is_file():
                return ActionResult(
                    success=False,
                    message=f"Not a file: {filepath}",
                )
            
            # Read file
            content = filepath.read_text(encoding="utf-8")
            
            return ActionResult(
                success=True,
                message=f"Read {len(content)} bytes from {filepath.name}",
                details={"content": content, "path": str(filepath)},
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to read file: {e}",
                details={"error": str(e)},
            )
    
    def _execute_file_write(self, request: ActionRequest) -> ActionResult:
        """Execute file write operation.
        
        Args:
            request: Write request (expects 'content' in details)
            
        Returns:
            ActionResult with write status
        """
        try:
            filepath = Path(request.target)
            details = request.details or {}
            content = details.get("content", "")
            
            # Create parent directories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            filepath.write_text(content, encoding="utf-8")
            
            return ActionResult(
                success=True,
                message=f"Wrote {len(content)} bytes to {filepath.name}",
                details={"path": str(filepath), "bytes_written": len(content)},
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to write file: {e}",
                details={"error": str(e)},
            )
    
    def _execute_file_edit(self, request: ActionRequest) -> ActionResult:
        """Execute file edit operation.
        
        Args:
            request: Edit request (expects 'old_content' and 'new_content' in details)
            
        Returns:
            ActionResult with edit status
        """
        try:
            filepath = Path(request.target)
            details = request.details or {}
            
            if not filepath.exists():
                return ActionResult(
                    success=False,
                    message=f"File not found: {filepath}",
                )
            
            # Read current content
            current_content = filepath.read_text(encoding="utf-8")
            
            # Get old/new content for replacement
            old_content = details.get("old_content", "")
            new_content = details.get("new_content", "")
            
            if not old_content:
                return ActionResult(
                    success=False,
                    message="No old_content specified for edit",
                )
            
            # Check if old content exists
            if old_content not in current_content:
                return ActionResult(
                    success=False,
                    message="old_content not found in file",
                )
            
            # Replace and write
            updated_content = current_content.replace(old_content, new_content, 1)
            filepath.write_text(updated_content, encoding="utf-8")
            
            return ActionResult(
                success=True,
                message=f"Edited {filepath.name}",
                details={"path": str(filepath), "changes": 1},
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to edit file: {e}",
                details={"error": str(e)},
            )
    
    def _execute_file_delete(self, request: ActionRequest) -> ActionResult:
        """Execute file delete operation.
        
        Args:
            request: Delete request
            
        Returns:
            ActionResult with delete status
        """
        try:
            filepath = Path(request.target)
            
            if not filepath.exists():
                return ActionResult(
                    success=False,
                    message=f"File not found: {filepath}",
                )
            
            # Delete file
            filepath.unlink()
            
            return ActionResult(
                success=True,
                message=f"Deleted {filepath.name}",
                details={"path": str(filepath)},
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to delete file: {e}",
                details={"error": str(e)},
            )
    
    def _execute_file_create(self, request: ActionRequest) -> ActionResult:
        """Execute file creation operation.
        
        Args:
            request: Create request (expects 'content' in details)
            
        Returns:
            ActionResult with creation status
        """
        try:
            filepath = Path(request.target)
            details = request.details or {}
            content = details.get("content", "")
            
            if filepath.exists():
                return ActionResult(
                    success=False,
                    message=f"File already exists: {filepath}",
                )
            
            # Create parent directories
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file
            filepath.write_text(content, encoding="utf-8")
            
            return ActionResult(
                success=True,
                message=f"Created {filepath.name}",
                details={"path": str(filepath), "bytes_written": len(content)},
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to create file: {e}",
                details={"error": str(e)},
            )
    
    def _execute_command(self, request: ActionRequest) -> ActionResult:
        """Execute shell command.
        
        Args:
            request: Command request
            
        Returns:
            ActionResult with command output
        """
        try:
            command = request.target
            details = request.details or {}
            cwd = details.get("cwd")
            timeout = details.get("timeout", 30)
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout,
            )
            
            return ActionResult(
                success=result.returncode == 0,
                message=f"Command {'succeeded' if result.returncode == 0 else 'failed'} (exit {result.returncode})",
                details={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "command": command,
                },
            )
            
        except subprocess.TimeoutExpired:
            return ActionResult(
                success=False,
                message=f"Command timed out after {timeout}s",
                details={"error": "timeout"},
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to execute command: {e}",
                details={"error": str(e)},
            )
    
    def _record_action(self, request: ActionRequest, success: bool, message: str) -> None:
        """Record action in history.
        
        Args:
            request: Action request
            success: Whether action succeeded
            message: Result message
        """
        self.action_history.append({
            "timestamp": datetime.now().isoformat(),
            "action_type": request.action_type,
            "target": request.target,
            "success": success,
            "message": message,
            "mode": self.current_mode.mode_type.value,
        })
        
        # Keep last 100 actions
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
    
    def get_action_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent action history.
        
        Args:
            limit: Maximum number of actions to return
            
        Returns:
            List of recent actions
        """
        return self.action_history[-limit:]
    
    def get_action_stats(self) -> Dict[str, Any]:
        """Get action statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.action_stats.copy()
    
    def update_mode(self, new_mode: ModeBase) -> None:
        """Update the current mode.
        
        Args:
            new_mode: New mode to use
        """
        self.current_mode = new_mode
    
    def clear_history(self) -> None:
        """Clear action history."""
        self.action_history.clear()
    
    def reset_stats(self) -> None:
        """Reset action statistics."""
        self.action_stats = {
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "blocked_actions": 0,
            "actions_by_type": {},
            "actions_by_mode": {},
        }
