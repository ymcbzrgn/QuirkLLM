"""
Chat Mode - Safe interactive mode with user confirmations.

Chat mode is the default QuirkLLM mode, providing a safe interactive
experience where all actions require user confirmation before execution.

Key Features:
- Explicit user confirmation for all actions
- Diff display before file edits
- Critical operation blocking (risk_level='critical')
- Rich UI with colored prompts and syntax highlighting
- Session-level "always allow" option for repeated actions

Confirmation Flow:
1. Display action details (type, target, risk level)
2. Show diff preview for file edits
3. Prompt user: [y]es, [n]o, [a]lways, [v]iew, [q]uit
4. Execute on confirmation, skip on rejection
5. Remember "always" choices for session
"""

import difflib
from typing import Optional, Dict, Set
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

from quirkllm.modes.base import (
    ModeBase,
    ModeType,
    ModeConfig,
    ActionRequest,
    ActionResult,
)


class ChatMode(ModeBase):
    """
    Safe interactive mode with user confirmations.
    
    Chat mode requires explicit user confirmation before executing
    any action. It displays diffs for file edits and blocks
    critical operations entirely.
    
    Attributes:
        console: Rich console for UI rendering
        always_allow: Set of action types user chose to always allow
        session_stats: Statistics for current session
    """
    
    def __init__(self, mode_type: ModeType, config: Optional[ModeConfig] = None):
        """
        Initialize Chat mode.
        
        Args:
            mode_type: Should be ModeType.CHAT
            config: Optional configuration (uses safe defaults)
        """
        # Ensure safe defaults for chat mode
        if config is None:
            config = ModeConfig(
                auto_confirm=False,  # Never auto-confirm in chat
                allow_file_edits=True,
                allow_destructive=False,  # Block destructive ops
                confirm_high_risk=True,
                diff_display=True,
            )
        
        super().__init__(mode_type, config)
        
        self.console = Console()
        self.always_allow: Set[str] = set()
        self.session_stats = {
            "actions_confirmed": 0,
            "actions_rejected": 0,
            "actions_blocked": 0,
        }
    
    def activate(self) -> None:
        """
        Activate Chat mode.
        
        Displays welcome message and clears session state.
        """
        super().activate()
        
        # Reset session state
        self.always_allow.clear()
        self.session_stats = {
            "actions_confirmed": 0,
            "actions_rejected": 0,
            "actions_blocked": 0,
        }
        
        # Display activation message
        self.console.print(
            Panel(
                "[bold cyan]Chat Mode Activated[/bold cyan]\n\n"
                "🛡️  Safe interactive mode\n"
                "✓ All actions require confirmation\n"
                "✓ Diff preview for file edits\n"
                "✗ Critical operations blocked\n\n"
                "[dim]Type /help for available commands[/dim]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
    
    def deactivate(self) -> None:
        """
        Deactivate Chat mode.
        
        Displays session statistics before deactivation.
        """
        super().deactivate()
        
        # Display session stats
        total_actions = sum(self.session_stats.values())
        if total_actions > 0:
            self.console.print(
                Panel(
                    f"[bold]Chat Mode Session Summary[/bold]\n\n"
                    f"✓ Confirmed: {self.session_stats['actions_confirmed']}\n"
                    f"✗ Rejected: {self.session_stats['actions_rejected']}\n"
                    f"🛑 Blocked: {self.session_stats['actions_blocked']}\n"
                    f"Total: {total_actions}",
                    border_style="dim",
                )
            )
    
    def handle_action(self, action: ActionRequest) -> ActionResult:
        """
        Handle an action request with user confirmation.
        
        Flow:
        1. Validate action (check if allowed)
        2. Block critical operations
        3. Display action details
        4. Show diff for file edits
        5. Prompt for confirmation
        6. Execute if confirmed
        
        Args:
            action: The action request to handle
            
        Returns:
            ActionResult with execution outcome
        """
        # Validate action against mode config
        is_valid, error = self.validate_action(action)
        if not is_valid:
            self.session_stats["actions_blocked"] += 1
            return ActionResult(
                success=False,
                message=error,
                errors=[error],
            )
        
        # Block critical operations
        if action.is_critical():
            self.session_stats["actions_blocked"] += 1
            self._display_blocked_message(action)
            return ActionResult(
                success=False,
                message="Critical operation blocked by Chat mode",
                errors=["Critical operations not allowed in Chat mode"],
            )
        
        # Check if user already chose "always allow" for this action type
        if action.action_type in self.always_allow:
            self.session_stats["actions_confirmed"] += 1
            return self._execute_action(action)
        
        # Display action details
        self._display_action_details(action)
        
        # Show diff for file edits
        if action.action_type in ("edit_file", "create_file") and self.config.diff_display:
            self._display_diff(action)
        
        # Get user confirmation
        confirmed, always = self._get_confirmation(action)
        
        if not confirmed:
            self.session_stats["actions_rejected"] += 1
            return ActionResult(
                success=False,
                message="Action rejected by user",
            )
        
        # Remember "always allow" choice
        if always:
            self.always_allow.add(action.action_type)
        
        # Execute action
        self.session_stats["actions_confirmed"] += 1
        return self._execute_action(action)
    
    def get_prompt_indicator(self) -> str:
        """
        Get the prompt indicator for Chat mode.
        
        Returns:
            Chat emoji indicator
        """
        return "💬"
    
    def _display_action_details(self, action: ActionRequest) -> None:
        """
        Display action details in a Rich panel.
        
        Args:
            action: Action to display
        """
        # Create details table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Field", style="bold")
        table.add_column("Value")
        
        table.add_row("Action", action.action_type)
        table.add_row("Target", action.target)
        
        # Risk level with color
        risk_colors = {
            "safe": "green",
            "low": "blue",
            "medium": "yellow",
            "high": "orange1",
            "critical": "red",
        }
        risk_color = risk_colors.get(action.risk_level, "white")
        table.add_row("Risk Level", f"[{risk_color}]{action.risk_level.upper()}[/{risk_color}]")
        
        # Details
        if action.details:
            for key, value in action.details.items():
                table.add_row(key.replace("_", " ").title(), str(value))
        
        self.console.print(
            Panel(
                table,
                title="[bold]Action Request[/bold]",
                border_style="blue",
            )
        )
    
    def _display_diff(self, action: ActionRequest) -> None:
        """
        Display diff for file edit actions.
        
        Args:
            action: Action containing file edit details
        """
        if "old_content" not in action.details or "new_content" not in action.details:
            return
        
        old_content = action.details["old_content"]
        new_content = action.details["new_content"]
        
        # Generate unified diff
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"{action.target} (before)",
            tofile=f"{action.target} (after)",
            lineterm="",
        )
        
        diff_text = "\n".join(diff)
        
        if diff_text:
            # Display with syntax highlighting
            syntax = Syntax(
                diff_text,
                "diff",
                theme="monokai",
                line_numbers=True,
            )
            
            self.console.print(
                Panel(
                    syntax,
                    title="[bold]Diff Preview[/bold]",
                    border_style="yellow",
                )
            )
    
    def _display_blocked_message(self, action: ActionRequest) -> None:
        """
        Display message for blocked critical operations.
        
        Args:
            action: Blocked action
        """
        self.console.print(
            Panel(
                f"[bold red]⛔ Critical Operation Blocked[/bold red]\n\n"
                f"Action: {action.action_type}\n"
                f"Target: {action.target}\n\n"
                f"[yellow]Critical operations are not allowed in Chat mode.[/yellow]\n"
                f"[dim]Switch to YAMI mode if you need to perform this action.[/dim]",
                border_style="red",
                padding=(1, 2),
            )
        )
    
    def _get_confirmation(self, action: ActionRequest) -> tuple[bool, bool]:
        """
        Get user confirmation for an action.
        
        Displays prompt with options:
        - [y]es: Execute this action once
        - [n]o: Skip this action
        - [a]lways: Execute and remember for session
        - [v]iew: View action details again
        - [q]uit: Reject and exit
        
        Args:
            action: Action to confirm
            
        Returns:
            Tuple of (confirmed, always_allow)
        """
        while True:
            response = Prompt.ask(
                "\n[bold]Confirm action?[/bold]",
                choices=["y", "n", "a", "v", "q"],
                default="n",
            ).lower()
            
            if response == "y":
                return True, False
            elif response == "n":
                return False, False
            elif response == "a":
                return True, True
            elif response == "v":
                # Re-display details
                self._display_action_details(action)
                if action.action_type in ("edit_file", "create_file"):
                    self._display_diff(action)
                continue
            elif response == "q":
                return False, False
    
    def _execute_action(self, action: ActionRequest) -> ActionResult:
        """
        Execute an action (placeholder for actual execution).
        
        In the real implementation, this would coordinate with
        FileManager, CommandRunner, etc. For now, it simulates
        successful execution.
        
        Args:
            action: Action to execute
            
        Returns:
            ActionResult with simulated success
        """
        # Placeholder execution
        return ActionResult(
            success=True,
            message=f"Action '{action.action_type}' executed successfully",
            details={"action_type": action.action_type, "target": action.target},
            modified_files=[action.target] if action.action_type in ("edit_file", "create_file") else [],
        )
    
    def get_session_stats(self) -> Dict[str, int]:
        """
        Get current session statistics.
        
        Returns:
            Dictionary with action counts
        """
        return self.session_stats.copy()
    
    def clear_always_allow(self) -> None:
        """Clear the "always allow" set for this session."""
        self.always_allow.clear()
