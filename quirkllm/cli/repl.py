"""Interactive REPL (Read-Eval-Print Loop) for QuirkLLM."""

import sys
from collections.abc import Callable
from dataclasses import dataclass

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.table import Table

from quirkllm.core.profile_manager import ProfileConfig
from quirkllm.core.system_detector import SystemInfo

console = Console()


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
        debug: bool = False,
    ):
        """Initialize REPL.

        Args:
            system_info: System detection results
            profile_config: Selected profile configuration
            debug: Enable debug mode
        """
        self.system_info = system_info
        self.profile_config = profile_config
        self.debug = debug
        self.running = False

        # Initialize prompt session with history
        self.history = InMemoryHistory()
        self.session: PromptSession[str] = PromptSession(
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
        )

        # Register commands
        self.commands = self._register_commands()

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

        console.print()
        console.print(table)
        console.print()
        console.print(
            "[dim]💡 Tip: Commands start with [bold]/[/bold]. "
            "Everything else is treated as a chat message.[/dim]"
        )
        console.print()

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

        console.print()
        console.print(sys_table)
        console.print()
        console.print(prof_table)
        console.print()

    def _cmd_quit(self) -> None:
        """Exit the REPL."""
        console.print("\n[dim]👋 Goodbye![/dim]\n")
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
        # cmd_args = parts[1] if len(parts) > 1 else ""  # For future use

        # Look up command
        cmd = self.commands.get(cmd_name)
        if cmd is None:
            console.print(
                f"[red]✗ Unknown command: /{cmd_name}[/red]\n"
                "[dim]Type [bold]/help[/bold] to see available commands.[/dim]\n"
            )
            return True

        # Execute command
        try:
            cmd.handler()
        except Exception as e:
            console.print(f"[red]✗ Command error: {e}[/red]\n")
            if self.debug:
                console.print_exception()

        return True

    def _handle_chat(self, user_input: str) -> None:
        """Handle a chat message (non-command input).

        Args:
            user_input: User's chat message
        """
        # Phase 1: Just echo back (no model yet)
        console.print("\n[dim]💬 Chat mode not yet implemented (Phase 2).[/dim]")
        console.print(f"[dim]You said: {user_input}[/dim]\n")

    def run(self) -> None:
        """Start the REPL loop."""
        self.running = True

        try:
            while self.running:
                try:
                    # Get user input
                    user_input = self.session.prompt(
                        "quirk> ",
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
                    console.print()
                    continue

                except EOFError:
                    # Ctrl+D: Exit gracefully
                    self._cmd_quit()
                    break

        except Exception as e:
            console.print(f"\n[red]✗ REPL error: {e}[/red]\n")
            if self.debug:
                console.print_exception()
            sys.exit(1)
