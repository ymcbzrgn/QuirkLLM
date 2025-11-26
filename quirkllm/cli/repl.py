"""Interactive REPL (Read-Eval-Print Loop) for QuirkLLM."""

import sys
from collections.abc import Callable
from dataclasses import dataclass

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from quirkllm.core.profile_manager import ProfileConfig
from quirkllm.core.system_detector import SystemInfo
from quirkllm.core.config import Config, load_config, save_config
from quirkllm.modes import (
    ModeBase,
    ModeType,
    get_registry,
)


@dataclass
class Command:
    """Represents a REPL slash command."""

    name: str
    description: str
    handler: Callable[[], None]
    aliases: list[str] | None = None


class REPL:
    """Interactive Read-Eval-Print Loop for QuirkLLM."""

    def __init__(
        self,
        system_info: SystemInfo,
        profile_config: ProfileConfig,
        config: Config | None = None,
        debug: bool = False,
    ):
        """Initialize REPL.

        Args:
            system_info: System detection results
            profile_config: Selected profile configuration
            config: User configuration (loaded from ~/.quirkllm/config.yaml)
            debug: Enable debug mode
        """
        self.system_info = system_info
        self.profile_config = profile_config
        self.config = config or load_config()
        self.debug = debug
        self.running = False
        self.console = Console()  # Instance variable for testability

        # Initialize mode system
        self.mode_registry = get_registry()
        self.current_mode: ModeBase | None = None
        self._current_command_args = ""  # For passing args to command handlers
        self._initialize_mode()

        # Initialize prompt session with history and key bindings
        self.history = InMemoryHistory()
        self.key_bindings = self._create_key_bindings()
        self.session: PromptSession[str] = PromptSession(
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self.key_bindings,
        )

        # Register commands
        self.commands = self._register_commands()

    def _initialize_mode(self) -> None:
        """Initialize the current mode from config."""
        # Get mode from config or default to CHAT
        mode_name = self.config.mode if self.config else "chat"
        mode_type = ModeType(mode_name.lower())
        
        # Create mode instance
        self.current_mode = self.mode_registry.create_mode(mode_type, self.config)
        
        # Activate mode
        if self.current_mode:
            self.current_mode.activate()

    def _create_key_bindings(self) -> KeyBindings:
        """Create custom key bindings for REPL.
        
        Returns:
            KeyBindings with Shift+Tab for mode cycling
        """
        kb = KeyBindings()
        
        @kb.add('s-tab')  # Shift+Tab
        def cycle_mode(event) -> None:
            """Cycle through modes: chat -> yami -> plan -> ghost -> chat."""
            self._cycle_mode()
            # Refresh prompt to show new mode indicator
            event.app.invalidate()
        
        return kb

    def _cycle_mode(self) -> None:
        """Cycle to the next mode in sequence."""
        if not self.current_mode:
            return
        
        # Define cycle order
        mode_cycle = [
            ModeType.CHAT,
            ModeType.YAMI,
            ModeType.PLAN,
            ModeType.GHOST,
        ]
        
        # Find current position and get next
        try:
            current_idx = mode_cycle.index(self.current_mode.mode_type)
            next_idx = (current_idx + 1) % len(mode_cycle)
            next_mode_type = mode_cycle[next_idx]
        except ValueError:
            # Fallback to CHAT if current mode not in cycle
            next_mode_type = ModeType.CHAT
        
        # Switch to next mode
        self.switch_mode(next_mode_type)
        
        # Display mode change notification
        self.console.print(
            f"\n[cyan]🔄 Mode switched to:[/cyan] [bold]{next_mode_type.value.upper()}[/bold] "
            f"{self.current_mode.get_prompt_indicator() if self.current_mode else ''}\n"
        )

    def switch_mode(self, mode_type: ModeType) -> bool:
        """Switch to a different mode.
        
        Args:
            mode_type: Target mode type
            
        Returns:
            True if switch successful, False otherwise
        """
        # Deactivate current mode
        if self.current_mode:
            self.current_mode.deactivate()
        
        # Create and activate new mode
        try:
            self.current_mode = self.mode_registry.create_mode(mode_type, self.config)
            if self.current_mode:
                self.current_mode.activate()
                
                # Persist mode to config
                self.config.mode = mode_type.value
                save_config(self.config)
                
                return True
        except Exception as e:
            self.console.print(f"[red]✗ Failed to switch mode: {e}[/red]")
            if self.debug:
                self.console.print_exception()
        
        return False

    def _get_prompt_text(self) -> str:
        """Get the prompt text with current mode indicator.
        
        Returns:
            Prompt string with mode emoji
        """
        if self.current_mode:
            indicator = self.current_mode.get_prompt_indicator()
            return f"{indicator} quirk> "
        return "quirk> "

    def _register_commands(self) -> dict[str, Command]:
        """Register all available REPL commands.

        Returns:
            Dictionary mapping command names to Command objects
        """
        commands = [
            Command(
                name="help",
                description="Show available commands and usage",
                handler=self._cmd_help,
                aliases=["?", "h"],
            ),
            Command(
                name="status",
                description="Display system and profile information",
                handler=self._cmd_status,
                aliases=["info", "stat"],
            ),
            Command(
                name="mode",
                description="Switch mode or show current mode (chat/yami/plan/ghost)",
                handler=self._cmd_mode,
                aliases=["m"],
            ),
            Command(
                name="quit",
                description="Exit QuirkLLM",
                handler=self._cmd_quit,
                aliases=["exit", "q"],
            ),
        ]

        # Create command mapping (include aliases)
        cmd_map = {}
        for cmd in commands:
            cmd_map[cmd.name] = cmd
            if cmd.aliases:
                for alias in cmd.aliases:
                    cmd_map[alias] = cmd

        return cmd_map

    def _cmd_help(self) -> None:
        """Display help information."""
        table = Table(title="Available Commands", show_header=True)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Aliases", style="dim")
        table.add_column("Description", style="white")

        # Group commands by primary name (skip aliases in display)
        seen = set()
        for cmd in self.commands.values():
            if cmd.name in seen:
                continue
            seen.add(cmd.name)

            aliases_str = ", ".join(cmd.aliases) if cmd.aliases else "-"
            table.add_row(f"/{cmd.name}", aliases_str, cmd.description)

        self.console.print()
        self.console.print(table)
        self.console.print()
        self.console.print(
            "[dim]💡 Tip: Commands start with [bold]/[/bold]. "
            "Everything else is treated as a chat message.[/dim]"
        )
        self.console.print()

    def _cmd_status(self) -> None:
        """Display system and profile status."""
        # System info table
        sys_table = Table(title="System Status", show_header=False, box=None)
        sys_table.add_column("Property", style="cyan")
        sys_table.add_column("Value", style="green")

        sys_table.add_row("Platform", f"{self.system_info.platform}/{self.system_info.processor}")
        sys_table.add_row("Total RAM", f"{self.system_info.total_ram_gb:.2f} GB")
        sys_table.add_row("Available RAM", f"{self.system_info.available_ram_gb:.2f} GB")
        sys_table.add_row(
            "Adjusted RAM",
            f"{self.system_info.adjusted_ram_gb:.2f} GB "
            f"({self.system_info.total_ram_gb - self.system_info.adjusted_ram_gb:.2f} GB reserved)",
        )
        sys_table.add_row("CUDA", "✓ Available" if self.system_info.has_cuda else "✗ Not available")
        sys_table.add_row(
            "Metal", "✓ Available" if self.system_info.has_metal else "✗ Not available"
        )

        # Profile info table
        prof_table = Table(title="Active Profile", show_header=False, box=None)
        prof_table.add_column("Property", style="cyan")
        prof_table.add_column("Value", style="yellow")

        prof_table.add_row("Name", f"🎯 {self.profile_config.name}")
        prof_table.add_row("Context Length", f"{self.profile_config.context_length:,} tokens")
        prof_table.add_row("Quantization", self.profile_config.quantization)
        prof_table.add_row("Batch Size", str(self.profile_config.batch_size))
        prof_table.add_row("RAG Cache", f"{self.profile_config.rag_cache_mb} MB")
        prof_table.add_row("KV Cache", f"{self.profile_config.kv_cache_gb} GB")
        prof_table.add_row("Embedding Model", self.profile_config.embedding_model)
        prof_table.add_row("Concurrent Ops", str(self.profile_config.concurrent_ops))
        prof_table.add_row("Compaction Mode", self.profile_config.compaction_mode)
        prof_table.add_row("Model Loading", self.profile_config.model_loading)
        prof_table.add_row(
            "Expected Speed", f"~{self.profile_config.expected_speed_toks} tokens/sec"
        )

        self.console.print()
        self.console.print(sys_table)
        self.console.print()
        self.console.print(prof_table)
        self.console.print()

    def _cmd_mode(self) -> None:
        """Display current mode or switch to a different mode.
        
        Usage:
            /mode          - Show current mode
            /mode <name>   - Switch to mode (chat, yami, plan, ghost)
        """
        args = getattr(self, '_current_command_args', '').strip()
        
        # No args: show current mode
        if not args:
            self._display_current_mode()
            return
        
        # Parse mode name
        mode_name = args.lower()
        
        # Validate mode name
        valid_modes = ["chat", "yami", "plan", "ghost"]
        if mode_name not in valid_modes:
            self.console.print(
                f"[red]✗ Invalid mode: {mode_name}[/red]\n"
                f"[dim]Valid modes: {', '.join(valid_modes)}[/dim]\n"
            )
            return
        
        # Convert to ModeType
        mode_type = ModeType(mode_name)
        
        # Check if already in this mode
        if self.current_mode and self.current_mode.mode_type == mode_type:
            self.console.print(f"\n[yellow]Already in {mode_name.upper()} mode.[/yellow]\n")
            return
        
        # Switch mode
        self.console.print(f"\n[cyan]Switching to {mode_name.upper()} mode...[/cyan]")
        if self.switch_mode(mode_type):
            self.console.print(
                f"[green]✓ Mode switched successfully![/green] "
                f"{self.current_mode.get_prompt_indicator() if self.current_mode else ''}\n"
            )
            # Show mode details
            self._display_mode_info(mode_type)
        else:
            self.console.print("[red]✗ Failed to switch mode.[/red]\n")

    def _display_current_mode(self) -> None:
        """Display information about the current mode."""
        if not self.current_mode:
            self.console.print("\n[red]No mode active.[/red]\n")
            return
        
        mode_type = self.current_mode.mode_type
        indicator = self.current_mode.get_prompt_indicator()
        
        self.console.print()
        self.console.print(Panel(
            f"[bold cyan]{mode_type.value.upper()}[/bold cyan] {indicator}",
            title="Current Mode",
            expand=False,
        ))
        self.console.print()
        
        # Show mode-specific info
        self._display_mode_info(mode_type)

    def _display_mode_info(self, mode_type: ModeType) -> None:
        """Display information about a specific mode.
        
        Args:
            mode_type: Mode to display info for
        """
        mode_info = {
            ModeType.CHAT: {
                "emoji": "🔄",
                "name": "Chat Mode",
                "description": "Safe mode with confirmations (default)",
                "features": [
                    "Asks before each action",
                    "Shows diff preview",
                    "Blocks critical operations",
                    "Always-allow option",
                ],
            },
            ModeType.YAMI: {
                "emoji": "🚀",
                "name": "YAMI Mode",
                "description": "Auto-accept with safety validation",
                "features": [
                    "Auto-confirms safe actions",
                    "Blocks critical operations",
                    "Warns on high-risk actions",
                    "Fast iterative workflow",
                ],
            },
            ModeType.PLAN: {
                "emoji": "📋",
                "name": "Plan Mode",
                "description": "Read-only planning and architecture",
                "features": [
                    "Generates implementation plans",
                    "Analyzes code architecture",
                    "Read-only (no writes)",
                    "Saves plans to .quirkllm/plans/",
                ],
            },
            ModeType.GHOST: {
                "emoji": "👻",
                "name": "Ghost Mode",
                "description": "Background file watcher",
                "features": [
                    "Watches file changes in background",
                    "Provides real-time analysis",
                    "Non-intrusive notifications",
                    "Impact and breaking change detection",
                ],
            },
        }
        
        info = mode_info.get(mode_type)
        if not info:
            return
        
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Feature", style="dim")
        
        for feature in info["features"]:
            table.add_row(f"• {feature}")
        
        self.console.print(f"[bold]{info['emoji']} {info['name']}[/bold]: {info['description']}")
        self.console.print()
        self.console.print(table)
        self.console.print()
        self.console.print("[dim]💡 Tip: Use Shift+Tab to quickly cycle through modes[/dim]\n")

    def _cmd_quit(self) -> None:
        """Exit the REPL."""
        self.console.print("\n[dim]👋 Goodbye![/dim]\n")
        self.running = False

    def _handle_command(self, input_text: str) -> bool:
        """Handle a slash command.

        Args:
            input_text: User input text

        Returns:
            True if command was handled, False otherwise
        """
        if not input_text.startswith("/"):
            return False

        # Parse command (remove leading slash and split)
        parts = input_text[1:].strip().split(maxsplit=1)
        if not parts:
            return False

        cmd_name = parts[0].lower()
        cmd_args = parts[1] if len(parts) > 1 else ""

        # Look up command
        cmd = self.commands.get(cmd_name)
        if cmd is None:
            self.console.print(
                f"[red]✗ Unknown command: /{cmd_name}[/red]\n"
                "[dim]Type [bold]/help[/bold] to see available commands.[/dim]\n"
            )
            return True

        # Execute command (pass args for commands that need them)
        try:
            # Store args for commands that need them
            self._current_command_args = cmd_args
            cmd.handler()
            self._current_command_args = ""
        except Exception as e:
            self.console.print(f"[red]✗ Command error: {e}[/red]\n")
            if self.debug:
                self.console.print_exception()

        return True

    def _handle_chat(self, user_input: str) -> None:
        """Handle a chat message (non-command input).

        Args:
            user_input: User's chat message
        """
        # Phase 1: Just echo back (no model yet)
        self.console.print("\n[dim]💬 Chat mode not yet implemented (Phase 2).[/dim]")
        self.console.print(f"[dim]You said: {user_input}[/dim]\n")

    def run(self) -> None:
        """Start the REPL loop."""
        self.running = True

        try:
            while self.running:
                try:
                    # Get user input with dynamic prompt
                    user_input = self.session.prompt(
                        self._get_prompt_text(),
                        # Add styling
                        # style=Style.from_dict({"prompt": "cyan bold"}),
                    ).strip()

                    # Skip empty input
                    if not user_input:
                        continue

                    # Handle commands
                    if self._handle_command(user_input):
                        continue

                    # Handle chat
                    self._handle_chat(user_input)

                except KeyboardInterrupt:
                    # Ctrl+C: Clear line and continue
                    self.console.print()
                    continue

                except EOFError:
                    # Ctrl+D: Exit gracefully
                    self._cmd_quit()
                    break

        except Exception as e:
            self.console.print(f"\n[red]✗ REPL error: {e}[/red]\n")
            if self.debug:
                self.console.print_exception()
            sys.exit(1)
