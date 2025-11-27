"""Interactive REPL (Read-Eval-Print Loop) for QuirkLLM."""

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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
from quirkllm.backends.base import GenerationParams
from quirkllm.cli.response_handler import ResponseHandler
from quirkllm.cli.prompts import build_agentic_prompt
from quirkllm.core.context_manager import ContextManager, FileContextManager
from quirkllm.core.tool_parser import ToolParser, ToolType
from quirkllm.file_ops.file_manager import FileManager


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
        model_path: str | None = None,
    ):
        """Initialize REPL.

        Args:
            system_info: System detection results
            profile_config: Selected profile configuration
            config: User configuration (loaded from ~/.quirkllm/config.yaml)
            debug: Enable debug mode
            model_path: Path to GGUF model file for chat inference
        """
        self.system_info = system_info
        self.profile_config = profile_config
        self.config = config or load_config()
        self.debug = debug
        self.running = False
        self.console = Console()  # Instance variable for testability

        # Backend initialization
        self.backend = None
        self.model_path = model_path
        if self.model_path is None and self.config and hasattr(self.config, 'model_path'):
            self.model_path = self.config.model_path

        if self.model_path:
            self._load_model()

        # Initialize mode system
        self.mode_registry = get_registry()
        self.current_mode: ModeBase | None = None
        self._current_command_args = ""  # For passing args to command handlers
        self._initialize_mode()

        # Initialize response handler for code-to-file functionality
        working_dir = Path.cwd()
        self.file_manager = FileManager(project_root=str(working_dir))
        self.response_handler = ResponseHandler(
            console=self.console,
            file_manager=self.file_manager,
            mode_registry=self.mode_registry,
            working_dir=working_dir,
        )

        # Phase 6.6: Initialize agentic behavior components
        self.file_context = FileContextManager(
            working_dir=working_dir,
            max_context_tokens=4000,
        )
        self.tool_parser = ToolParser()
        self.max_tool_iterations = 3  # Prevent infinite loops

        # Phase 6.7: Initialize conversation history manager
        self.conversation = ContextManager(
            max_context_length=self.profile_config.context_length
        )

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

    def _load_model(self) -> bool:
        """Load LLM model for chat.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if not self.model_path:
            return False

        from pathlib import Path
        model_file = Path(self.model_path).expanduser()

        if not model_file.exists():
            self.console.print(f"[red]✗ Model not found: {model_file}[/red]")
            return False

        try:
            self.console.print(f"[cyan]📦 Loading model: {model_file.name}[/cyan]")

            # Create backend
            from quirkllm.backends.llamacpp import LlamaCppBackend
            self.backend = LlamaCppBackend()

            # Load with profile settings
            self.backend.load_model(
                model_path=str(model_file),
                n_ctx=self.profile_config.context_length,
                n_batch=self.profile_config.batch_size,
                n_gpu_layers=-1 if self.system_info.has_metal else 0,  # Metal = all layers
            )

            self.console.print(f"[green]✓ Model loaded successfully![/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Model loading failed: {e}[/red]")
            if self.debug:
                self.console.print_exception()
            return False

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
            Command(
                name="learn",
                description="Learn from URL or PDF (/learn --url <url> or /learn --pdf <path>)",
                handler=self._cmd_learn,
            ),
            Command(
                name="knowledge",
                description="Manage knowledge sources (/knowledge list|stats|forget <id>)",
                handler=self._cmd_knowledge,
                aliases=["k"],
            ),
            # Phase 6.6: Agentic file context commands
            Command(
                name="read",
                description="Load file into context (/read <path>)",
                handler=self._cmd_read,
                aliases=["r"],
            ),
            Command(
                name="context",
                description="Show loaded files and context status",
                handler=self._cmd_context,
                aliases=["ctx"],
            ),
            Command(
                name="clear",
                description="Clear loaded file context",
                handler=self._cmd_clear_context,
                aliases=["clr"],
            ),
            # Phase 6.7: Conversation history commands
            Command(
                name="history",
                description="Show conversation history stats",
                handler=self._cmd_history,
                aliases=["hist"],
            ),
            Command(
                name="reset",
                description="Reset conversation history (keeps files)",
                handler=self._cmd_reset,
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

    def _cmd_learn(self) -> None:
        """Learn from URL or PDF document.

        Usage:
            /learn --url <url> [--depth <n>]   - Learn from web URL
            /learn --pdf <path>                - Learn from PDF file
        """
        args = getattr(self, '_current_command_args', '').strip()

        if not args:
            self.console.print(
                "\n[yellow]Usage:[/yellow]\n"
                "  /learn --url <url> [--depth <n>]  - Learn from web URL\n"
                "  /learn --pdf <path>               - Learn from PDF file\n"
            )
            return

        parts = args.split()

        if "--url" in parts:
            try:
                url_idx = parts.index("--url") + 1
                if url_idx >= len(parts):
                    self.console.print("[red]✗ URL required after --url[/red]\n")
                    return

                url = parts[url_idx]

                # Check for depth option
                depth = 1
                if "--depth" in parts:
                    depth_idx = parts.index("--depth") + 1
                    if depth_idx < len(parts):
                        try:
                            depth = int(parts[depth_idx])
                        except ValueError:
                            pass

                self._learn_from_url(url, depth)

            except Exception as e:
                self.console.print(f"[red]✗ Error: {e}[/red]\n")

        elif "--pdf" in parts:
            try:
                pdf_idx = parts.index("--pdf") + 1
                if pdf_idx >= len(parts):
                    self.console.print("[red]✗ PDF path required after --pdf[/red]\n")
                    return

                pdf_path = parts[pdf_idx]
                self._learn_from_pdf(pdf_path)

            except Exception as e:
                self.console.print(f"[red]✗ Error: {e}[/red]\n")
        else:
            self.console.print(
                "[yellow]Unknown option. Use --url or --pdf[/yellow]\n"
            )

    def _learn_from_url(self, url: str, depth: int = 1) -> None:
        """Learn from a web URL.

        Args:
            url: Web URL to crawl
            depth: Crawl depth (default: 1)
        """
        from pathlib import Path

        self.console.print(f"\n[cyan]🌐 Learning from URL: {url}[/cyan]")
        self.console.print(f"[dim]Crawl depth: {depth}[/dim]")

        try:
            from quirkllm.knowledge.ingestion_pipeline import IngestionPipeline

            # Create pipeline
            pipeline = IngestionPipeline()

            # Ingest URL
            with self.console.status("[cyan]Crawling and processing...[/cyan]"):
                result = pipeline.ingest_url(url, max_depth=depth)

            if result.get("success"):
                self.console.print(
                    f"[green]✓ Successfully learned from URL![/green]\n"
                    f"  Documents: {result.get('documents', 0)}\n"
                    f"  Chunks: {result.get('chunks', 0)}\n"
                )
            else:
                self.console.print(
                    f"[red]✗ Failed to learn: {result.get('error', 'Unknown error')}[/red]\n"
                )

        except ImportError:
            self.console.print(
                "[yellow]⚠ Knowledge Eater module not available.[/yellow]\n"
                "[dim]The ingestion pipeline is not installed or configured.[/dim]\n"
            )
        except Exception as e:
            self.console.print(f"[red]✗ Error: {e}[/red]\n")

    def _learn_from_pdf(self, pdf_path: str) -> None:
        """Learn from a PDF file.

        Args:
            pdf_path: Path to PDF file
        """
        from pathlib import Path

        path = Path(pdf_path).expanduser().resolve()

        if not path.exists():
            self.console.print(f"[red]✗ File not found: {pdf_path}[/red]\n")
            return

        if not path.suffix.lower() == ".pdf":
            self.console.print(f"[red]✗ Not a PDF file: {pdf_path}[/red]\n")
            return

        self.console.print(f"\n[cyan]📄 Learning from PDF: {path.name}[/cyan]")

        try:
            from quirkllm.knowledge.ingestion_pipeline import IngestionPipeline

            # Create pipeline
            pipeline = IngestionPipeline()

            # Ingest PDF
            with self.console.status("[cyan]Processing PDF...[/cyan]"):
                result = pipeline.ingest_pdf(str(path))

            if result.get("success"):
                self.console.print(
                    f"[green]✓ Successfully learned from PDF![/green]\n"
                    f"  Pages: {result.get('pages', 0)}\n"
                    f"  Chunks: {result.get('chunks', 0)}\n"
                )
            else:
                self.console.print(
                    f"[red]✗ Failed to learn: {result.get('error', 'Unknown error')}[/red]\n"
                )

        except ImportError:
            self.console.print(
                "[yellow]⚠ Knowledge Eater module not available.[/yellow]\n"
                "[dim]The ingestion pipeline is not installed or configured.[/dim]\n"
            )
        except Exception as e:
            self.console.print(f"[red]✗ Error: {e}[/red]\n")

    def _cmd_knowledge(self) -> None:
        """Manage knowledge sources.

        Usage:
            /knowledge           - List knowledge sources
            /knowledge list      - List knowledge sources
            /knowledge stats     - Show knowledge statistics
            /knowledge forget <id> - Remove a knowledge source
        """
        args = getattr(self, '_current_command_args', '').strip()
        parts = args.split() if args else []

        if not parts or parts[0] == "list":
            self._knowledge_list()
        elif parts[0] == "stats":
            self._knowledge_stats()
        elif parts[0] == "forget" and len(parts) > 1:
            self._knowledge_forget(parts[1])
        else:
            self.console.print(
                "\n[yellow]Usage:[/yellow]\n"
                "  /knowledge           - List knowledge sources\n"
                "  /knowledge list      - List knowledge sources\n"
                "  /knowledge stats     - Show knowledge statistics\n"
                "  /knowledge forget <id> - Remove a knowledge source\n"
            )

    def _knowledge_list(self) -> None:
        """List all knowledge sources."""
        self.console.print("\n[cyan]📚 Knowledge Sources[/cyan]\n")

        try:
            from quirkllm.knowledge.ingestion_pipeline import IngestionPipeline

            pipeline = IngestionPipeline()
            sources = pipeline.list_sources()

            if not sources:
                self.console.print("[dim]No knowledge sources found.[/dim]\n")
                return

            table = Table(show_header=True)
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Type", style="yellow")
            table.add_column("Source", style="white")
            table.add_column("Chunks", style="green")
            table.add_column("Added", style="dim")

            for source in sources:
                table.add_row(
                    source.get("id", "?")[:8],
                    source.get("type", "?"),
                    source.get("source", "?")[:50],
                    str(source.get("chunks", 0)),
                    source.get("added", "?"),
                )

            self.console.print(table)
            self.console.print()

        except ImportError:
            self.console.print(
                "[yellow]⚠ Knowledge Eater module not available.[/yellow]\n"
            )
        except Exception as e:
            self.console.print(f"[red]✗ Error: {e}[/red]\n")

    def _knowledge_stats(self) -> None:
        """Show knowledge statistics."""
        self.console.print("\n[cyan]📊 Knowledge Statistics[/cyan]\n")

        try:
            from quirkllm.knowledge.ingestion_pipeline import IngestionPipeline

            pipeline = IngestionPipeline()
            stats = pipeline.get_stats()

            table = Table(show_header=False, box=None)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Sources", str(stats.get("total_sources", 0)))
            table.add_row("Total Chunks", str(stats.get("total_chunks", 0)))
            table.add_row("Web Sources", str(stats.get("web_sources", 0)))
            table.add_row("PDF Sources", str(stats.get("pdf_sources", 0)))
            table.add_row("Storage Size", f"{stats.get('storage_mb', 0):.2f} MB")

            self.console.print(table)
            self.console.print()

        except ImportError:
            self.console.print(
                "[yellow]⚠ Knowledge Eater module not available.[/yellow]\n"
            )
        except Exception as e:
            self.console.print(f"[red]✗ Error: {e}[/red]\n")

    def _knowledge_forget(self, source_id: str) -> None:
        """Remove a knowledge source.

        Args:
            source_id: ID of the source to remove
        """
        self.console.print(f"\n[cyan]🗑️ Removing source: {source_id}[/cyan]")

        try:
            from quirkllm.knowledge.ingestion_pipeline import IngestionPipeline

            pipeline = IngestionPipeline()
            result = pipeline.remove_source(source_id)

            if result.get("success"):
                self.console.print(
                    f"[green]✓ Source removed successfully![/green]\n"
                    f"  Chunks deleted: {result.get('chunks_deleted', 0)}\n"
                )
            else:
                self.console.print(
                    f"[red]✗ Failed to remove: {result.get('error', 'Unknown error')}[/red]\n"
                )

        except ImportError:
            self.console.print(
                "[yellow]⚠ Knowledge Eater module not available.[/yellow]\n"
            )
        except Exception as e:
            self.console.print(f"[red]✗ Error: {e}[/red]\n")

    # ==========================================================================
    # Phase 6.6: Agentic File Context Commands
    # ==========================================================================

    def _cmd_read(self) -> None:
        """Load a file into context.

        Usage:
            /read <path>  - Load file into context for LLM
        """
        args = getattr(self, '_current_command_args', '').strip()

        if not args:
            self.console.print(
                "\n[yellow]Usage:[/yellow] /read <path>\n"
                "[dim]Example: /read main.py[/dim]\n"
            )
            return

        file_path = args
        result = self.file_context.load_file(file_path)

        if result:
            self.console.print(
                f"\n[green]✓ Loaded: {result.path}[/green]\n"
                f"[dim]  {result.line_count} lines, ~{result.token_estimate} tokens[/dim]\n"
            )
        else:
            self.console.print(
                f"\n[red]✗ Failed to load: {file_path}[/red]\n"
                f"[dim]File not found or token limit exceeded.[/dim]\n"
            )

    def _cmd_context(self) -> None:
        """Show loaded files and context status."""
        self.console.print()
        self.console.print(Panel(
            self.file_context.get_loaded_files_summary(),
            title="[bold]File Context[/bold]",
            border_style="cyan",
        ))
        self.console.print()

    def _cmd_clear_context(self) -> None:
        """Clear all loaded file context."""
        self.file_context.clear_files()
        self.console.print("\n[green]✓ File context cleared.[/green]\n")

    # ==========================================================================
    # Phase 6.7: Conversation History Commands
    # ==========================================================================

    def _cmd_history(self) -> None:
        """Show conversation history statistics."""
        stats = self.conversation.get_stats()

        self.console.print()
        self.console.print(Panel(
            f"[bold]Conversation History[/bold]\n\n"
            f"Messages: {stats['total_messages']} "
            f"(user: {stats['user_messages']}, assistant: {stats['assistant_messages']})\n"
            f"Tokens: {stats['total_tokens']:,} / {stats['max_tokens']:,} "
            f"({stats['usage_percentage']:.1f}% used)\n"
            f"Available: {stats['available_tokens']:,} tokens\n"
            f"Warning Level: {stats['warning_level']}",
            title="[bold cyan]📜 History[/bold cyan]",
            border_style="cyan",
        ))
        self.console.print()

    def _cmd_reset(self) -> None:
        """Reset conversation history (keeps file context)."""
        self.conversation.clear(keep_system=False)
        self.console.print("\n[green]✓ Conversation history reset.[/green]")
        self.console.print("[dim]File context preserved. Use /clear to also clear files.[/dim]\n")

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

    def _build_chat_prompt(self, user_input: str, include_current_user: bool = True) -> str:
        """Build ChatML format prompt for Qwen2.5-Instruct models.

        Uses agentic system prompt with file context and conversation history.

        Args:
            user_input: User's message
            include_current_user: Whether to include current user message (False for agentic loop)

        Returns:
            ChatML formatted prompt string with conversation history
        """
        # Build agentic system prompt with file context
        file_context = self.file_context.get_file_context_prompt()
        system_prompt = build_agentic_prompt(
            working_dir=str(self.file_context.working_dir),
            file_context=file_context,
        )

        # Start with system prompt
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

        # Add conversation history (exclude system messages - already in system prompt)
        for msg in self.conversation.messages:
            if msg.role != "system":
                prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"

        # Add current user message if requested
        if include_current_user:
            prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n"

        # Start assistant turn
        prompt += "<|im_start|>assistant\n"

        return prompt

    def _calculate_max_tokens(self) -> int:
        """Calculate max tokens for generation based on profile.

        Uses profile's context_length to determine appropriate max_tokens:
        - Base: context_length / 4 (25% of context for generation)
        - Min: 1024 tokens
        - Max: 4096 tokens (cap to prevent excessive generation time)

        Returns:
            Max tokens for generation
        """
        base = self.profile_config.context_length // 4
        return max(1024, min(base, 4096))

    def _handle_chat(self, user_input: str) -> None:
        """Handle a chat message with streaming and agentic loop.

        Implements the agentic behavior:
        1. Build prompt with file context and conversation history
        2. Generate response with streaming (real-time output)
        3. Check for tool calls ([READ:], [LS:], [SEARCH:])
        4. Execute tools and re-prompt (max 3 iterations)
        5. Process final response with code extraction
        6. Store conversation in history for multi-turn support

        Phase 6.8: Uses streaming output and dynamic max_tokens.

        Args:
            user_input: User's chat message
        """
        # Check if model is loaded
        if self.backend is None or not self.backend.is_loaded():
            self.console.print(
                "\n[yellow]⚠ No model loaded.[/yellow]\n"
                "[dim]Use --model-path to specify a GGUF model, e.g.:[/dim]\n"
                "[dim]  quirkllm --model-path ~/.quirkllm/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf[/dim]\n"
            )
            return

        # Auto-detect files mentioned in user input
        detected_files = self.file_context.auto_detect_files(user_input)
        for file_path in detected_files:
            result = self.file_context.load_file(file_path)
            if result:
                self.console.print(f"[dim]📄 Auto-loaded: {file_path}[/dim]")

        # Phase 6.7: Store user message in conversation history
        self.conversation.add_message("user", user_input)

        # Phase 6.8: Calculate dynamic max_tokens based on profile
        max_tokens = self._calculate_max_tokens()

        # Agentic loop with tool execution
        iteration = 0
        accumulated_response = ""
        final_response = ""

        while iteration < self.max_tool_iterations:
            iteration += 1

            # Build ChatML prompt with conversation history
            prompt = self._build_chat_prompt(user_input, include_current_user=False)

            try:
                # Phase 6.8: Streaming generation
                if iteration == 1:
                    self.console.print("\n[bold cyan]A:[/bold cyan]")
                else:
                    self.console.print(f"\n[dim](iteration {iteration})[/dim]")

                response_text = ""

                # Stream tokens in real-time
                for chunk in self.backend.generate_stream(
                    GenerationParams(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        stop_sequences=["<|im_end|>", "<|im_start|>"],
                    )
                ):
                    print(chunk, end="", flush=True)
                    response_text += chunk

                print()  # Newline after streaming
                response_text = response_text.strip()
                accumulated_response += response_text

                # Check for tool calls
                tool_calls = self.tool_parser.parse(response_text)

                if not tool_calls:
                    # No tool calls - process final response for code blocks
                    final_response = response_text
                    # Handle code extraction and file operations (no re-display)
                    self.response_handler.handle_code_blocks(response_text, user_input)
                    break

                # Execute tool calls
                self.console.print(f"\n[dim]🔧 Executing {len(tool_calls)} tool(s)...[/dim]")

                for tool_call in tool_calls:
                    self._execute_tool_call(tool_call)

                # Continue loop with updated context (tools executed)

            except Exception as e:
                self.console.print(f"\n[red]✗ Generation failed: {e}[/red]\n")
                if self.debug:
                    self.console.print_exception()
                break

        else:
            # Max iterations reached - show what we have
            self.console.print(
                f"\n[yellow]⚠ Max iterations ({self.max_tool_iterations}) reached.[/yellow]"
            )
            if accumulated_response:
                final_response = self.tool_parser.remove_tool_calls(accumulated_response)

        # Phase 6.7: Store assistant response in conversation history
        if final_response:
            self.conversation.add_message("assistant", final_response)

        # Phase 6.7: Check if context needs compaction
        if self.conversation.needs_compaction(threshold=80.0):
            removed = self.conversation.compact(target_percentage=50.0)
            self.console.print(f"[dim]📦 Compacted: removed {removed} old messages[/dim]")

        # Show token info in debug mode
        if self.debug:
            stats = self.conversation.get_stats()
            self.console.print(
                f"[dim]Iterations: {iteration}, max_tokens: {max_tokens}, "
                f"Files: {len(self.file_context.loaded_files)}, "
                f"History: {stats['total_messages']} msgs ({stats['usage_percentage']:.1f}%)[/dim]\n"
            )

    def _execute_tool_call(self, tool_call) -> None:
        """Execute a tool call from model output.

        Args:
            tool_call: ToolCall object with type and argument
        """
        if tool_call.tool_type == ToolType.READ:
            # Read file into context
            file_path = tool_call.argument
            result = self.file_context.load_file(file_path)
            if result:
                self.console.print(
                    f"[green]✓ READ: {result.path} "
                    f"({result.line_count} lines)[/green]"
                )
            else:
                self.console.print(
                    f"[red]✗ READ failed: {file_path}[/red]"
                )

        elif tool_call.tool_type == ToolType.LS:
            # List directory (already in context via get_cwd_listing)
            path = tool_call.argument
            self.console.print(f"[green]✓ LS: {path}[/green]")
            # Directory listing is always included in context

        elif tool_call.tool_type == ToolType.SEARCH:
            # Search is informational - model can see directory listing
            pattern = tool_call.argument
            self.console.print(f"[yellow]⚠ SEARCH: {pattern} (not yet implemented)[/yellow]")

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
