"""
YAMI Mode (You Asked Me, I did) - Fast auto-accept mode.

YAMI Mode is designed for experienced users who want minimal friction.
It auto-accepts most operations but still enforces safety through the
SafetyChecker to prevent catastrophic mistakes.

Behavior:
- Auto-confirms medium and low-risk actions
- Warns on high-risk operations with single confirmation
- Always blocks critical operations (rm -rf /, fork bombs, etc.)
- Fast execution for experienced users

Safety Features:
- Integrates SafetyChecker for all operations
- Critical patterns always blocked
- High-risk patterns trigger warning + confirmation
- Session statistics tracking

Use Cases:
- Rapid prototyping
- Experienced users who understand risks
- Batch operations with known safety
- Development workflows with frequent edits
"""

from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from quirkllm.modes.base import (
    ModeBase,
    ModeType,
    ModeConfig,
    ActionRequest,
    ActionResult,
)
from quirkllm.modes.safety_checker import SafetyChecker


class YAMIMode(ModeBase):
    """
    YAMI Mode - Fast auto-accept mode with safety validation.
    
    "You Asked Me, I did" - minimal confirmations for power users.
    
    Confirmation Rules:
    - Critical (risk >= 100): Always blocked, no execution
    - High-risk (risk >= 75): Warning + single confirmation required
    - Medium (risk >= 50): Auto-accept with notice
    - Low (risk >= 25): Auto-accept silently
    - Safe (risk < 25): Auto-accept silently
    
    The SafetyChecker provides the risk assessment, and YAMI mode
    applies minimal friction while preventing catastrophic mistakes.
    
    Attributes:
        console: Rich console for UI output
        safety_checker: SafetyChecker instance for validation
        session_stats: Statistics tracking for the session
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize YAMI mode.
        
        Args:
            **kwargs: Additional configuration options
        """
        # YAMI config: auto-confirm enabled, allow destructive operations
        config = ModeConfig(
            auto_confirm=True,
            allow_file_edits=True,
            allow_destructive=True,  # But validated by SafetyChecker
            confirm_high_risk=True,  # Still confirm high-risk
            diff_display=False,  # Skip diff for speed
            background_watch=False,
        )
        super().__init__(ModeType.YAMI, config)
        self.console = Console()
        self.safety_checker = SafetyChecker()
        
        # Session statistics
        self.session_stats: Dict[str, int] = {
            "actions_executed": 0,
            "actions_warned": 0,
            "actions_blocked": 0,
        }
    
    def activate(self) -> None:
        """
        Activate YAMI mode and display welcome message.
        
        Shows warning about auto-accept behavior and safety features.
        Resets session statistics.
        """
        self.active = True
        
        # Reset session stats
        self.session_stats = {
            "actions_executed": 0,
            "actions_warned": 0,
            "actions_blocked": 0,
        }
        
        # Display activation panel
        welcome_panel = Panel(
            "[bold yellow]YAMI MODE ACTIVATED[/]\n\n"
            "[yellow]You Asked Me, I did[/] - Fast auto-accept mode\n\n"
            "🚀 [green]Auto-confirming[/] medium/low-risk actions\n"
            "⚠️  [yellow]Warning prompt[/] for high-risk actions\n"
            "🛑 [red]Blocking[/] critical operations\n\n"
            "[dim]Safety checker active. Use with confidence![/]",
            title="🚀 YAMI Mode",
            border_style="yellow",
        )
        self.console.print(welcome_panel)
    
    def deactivate(self) -> None:
        """
        Deactivate YAMI mode and display session statistics.
        
        Shows summary of actions executed, warned, and blocked.
        """
        self.active = False
        
        # Display session summary
        stats_table = Table(title="YAMI Session Summary", show_header=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Count", style="green", justify="right")
        
        stats_table.add_row("Actions Executed", str(self.session_stats["actions_executed"]))
        stats_table.add_row("Actions Warned", str(self.session_stats["actions_warned"]))
        stats_table.add_row("Actions Blocked", str(self.session_stats["actions_blocked"]))
        
        self.console.print("\n")
        self.console.print(stats_table)
        self.console.print("\n[dim]YAMI mode deactivated[/]")
    
    def handle_action(self, request: ActionRequest) -> ActionResult:
        """
        Handle action with minimal confirmation and safety validation.
        
        Flow:
        1. Validate action with SafetyChecker
        2. Block if critical (risk >= 100)
        3. Warn + confirm if high-risk (risk >= 75)
        4. Auto-accept if medium/low/safe (risk < 75)
        5. Execute action and track statistics
        
        Args:
            request: The action to handle
            
        Returns:
            ActionResult with execution status
        """
        # Validate with safety checker
        validation = self.safety_checker.validate_action(request)
        
        # Update request risk level based on validation
        request.risk_level = validation.severity
        
        # CRITICAL: Always block
        if validation.severity == "critical" or not validation.is_safe:
            self._display_blocked_message(request, validation)
            self.session_stats["actions_blocked"] += 1
            
            return ActionResult(
                success=False,
                message=f"Critical operation blocked: {', '.join(validation.blocked_reasons)}",
                errors=validation.blocked_reasons,
                warnings=validation.warnings,
            )
        
        # HIGH-RISK: Warn and require confirmation
        if validation.severity == "high":
            self._display_high_risk_warning(request, validation)
            
            # Get confirmation
            if not self._get_high_risk_confirmation():
                self.session_stats["actions_warned"] += 1
                return ActionResult(
                    success=False,
                    message="High-risk action cancelled by user",
                    warnings=validation.warnings,
                )
            
            self.session_stats["actions_warned"] += 1
        
        # MEDIUM: Auto-accept with notice
        elif validation.severity == "medium":
            self.console.print(
                f"[dim]⚡ Auto-accepting: {request.action_type} on {request.target}[/]"
            )
            if validation.warnings:
                for warning in validation.warnings:
                    self.console.print(f"[dim yellow]  ⚠ {warning}[/]")
        
        # LOW/SAFE: Auto-accept silently
        else:
            self.console.print(
                f"[dim]✓ Executing: {request.action_type} on {request.target}[/]"
            )
        
        # Execute action
        result = self._execute_action(request)
        
        if result.success:
            self.session_stats["actions_executed"] += 1
        
        return result
    
    def get_prompt_indicator(self) -> str:
        """
        Get the prompt indicator for YAMI mode.
        
        Returns:
            Rocket emoji indicating fast mode
        """
        return "🚀"
    
    def _display_blocked_message(
        self, request: ActionRequest, validation: Any
    ) -> None:
        """
        Display blocked operation message.
        
        Args:
            request: The blocked action request
            validation: ValidationResult from SafetyChecker
        """
        blocked_panel = Panel(
            f"[bold red]OPERATION BLOCKED[/]\n\n"
            f"Action: [yellow]{request.action_type}[/]\n"
            f"Target: [yellow]{request.target}[/]\n"
            f"Risk Score: [red]{validation.risk_score}/100[/]\n\n"
            f"[red]Reasons:[/]\n"
            + "\n".join(f"  • {reason}" for reason in validation.blocked_reasons)
            + (
                f"\n\n[yellow]Matched Patterns:[/]\n"
                + "\n".join(f"  • {pattern}" for pattern in validation.matched_patterns)
                if validation.matched_patterns
                else ""
            ),
            title="🛑 Critical Operation",
            border_style="red",
        )
        self.console.print(blocked_panel)
    
    def _display_high_risk_warning(
        self, request: ActionRequest, validation: Any
    ) -> None:
        """
        Display high-risk operation warning.
        
        Args:
            request: The high-risk action request
            validation: ValidationResult from SafetyChecker
        """
        warning_panel = Panel(
            f"[bold yellow]HIGH-RISK OPERATION[/]\n\n"
            f"Action: [cyan]{request.action_type}[/]\n"
            f"Target: [cyan]{request.target}[/]\n"
            f"Risk Score: [yellow]{validation.risk_score}/100[/]\n\n"
            f"[yellow]Warnings:[/]\n"
            + "\n".join(f"  • {warning}" for warning in validation.warnings)
            + (
                f"\n\n[yellow]Matched Patterns:[/]\n"
                + "\n".join(f"  • {pattern}" for pattern in validation.matched_patterns)
                if validation.matched_patterns
                else ""
            )
            + "\n\n[dim]YAMI mode requires confirmation for high-risk actions[/]",
            title="⚠️  High-Risk Warning",
            border_style="yellow",
        )
        self.console.print(warning_panel)
    
    def _get_high_risk_confirmation(self) -> bool:
        """
        Get user confirmation for high-risk operation.
        
        Returns:
            True if user confirms, False otherwise
        """
        return Confirm.ask(
            "[yellow]⚠️  Proceed with high-risk operation?[/]",
            default=False,
        )
    
    def _execute_action(self, request: ActionRequest) -> ActionResult:
        """
        Execute the validated action.
        
        This is a placeholder implementation. In the full system, this would
        coordinate with FileManager, CommandExecutor, etc.
        
        Args:
            request: The action to execute
            
        Returns:
            ActionResult with execution status
        """
        # Placeholder execution - real implementation will use actual handlers
        return ActionResult(
            success=True,
            message=f"Action '{request.action_type}' executed successfully",
            details={"action_type": request.action_type, "target": request.target},
        )
    
    def get_session_stats(self) -> Dict[str, int]:
        """
        Get current session statistics.
        
        Returns:
            Dictionary with executed/warned/blocked counts
        """
        return self.session_stats.copy()
