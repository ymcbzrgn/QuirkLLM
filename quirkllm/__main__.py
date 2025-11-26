"""QuirkLLM CLI entry point."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from quirkllm.cli.repl import REPL
from quirkllm.core.profile_manager import ProfileConfig, select_profile
from quirkllm.core.system_detector import SystemInfo, detect_system

console = Console()

# Version info
__version__ = "0.1.0"


def display_welcome_banner(system_info: "SystemInfo", profile_config: ProfileConfig) -> None:
    """Display welcome banner with system information.

    Args:
        system_info: System detection results
        profile_config: Selected profile configuration
    """
    # Create system info table
    table = Table(title="System Information", show_header=False, box=None)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Platform", f"{system_info.platform}/{system_info.processor}")
    table.add_row(
        "RAM",
        f"{system_info.total_ram_gb:.1f} GB total / {system_info.available_ram_gb:.1f} GB available",
    )
    table.add_row(
        "Adjusted RAM",
        f"{system_info.adjusted_ram_gb:.1f} GB "
        f"({system_info.total_ram_gb - system_info.adjusted_ram_gb:.1f} GB reserved)",
    )
    table.add_row(
        "GPU",
        f"CUDA: {'✓' if system_info.has_cuda else '✗'} | "
        f"Metal: {'✓' if system_info.has_metal else '✗'}",
    )

    # Create profile info table
    profile_table = Table(title="Active Profile", show_header=False, box=None)
    profile_table.add_column("Property", style="cyan")
    profile_table.add_column("Value", style="yellow")

    profile_table.add_row("Profile", f"🎯 {profile_config.name}")
    profile_table.add_row("Context Length", f"{profile_config.context_length:,} tokens")
    profile_table.add_row("Quantization", profile_config.quantization)
    profile_table.add_row("Batch Size", str(profile_config.batch_size))
    profile_table.add_row("RAG Cache", f"{profile_config.rag_cache_mb} MB")
    profile_table.add_row("KV Cache", f"{profile_config.kv_cache_gb} GB")
    profile_table.add_row("Expected Speed", f"~{profile_config.expected_speed_toks} tokens/sec")

    # Display banner
    console.print()
    console.print(
        Panel.fit(
            "[bold magenta]QuirkLLM[/bold magenta] - RAM-Aware AI Coding Assistant\n"
            f"[dim]Version {__version__}[/dim]",
            border_style="magenta",
        )
    )
    console.print()
    console.print(table)
    console.print()
    console.print(profile_table)
    console.print()
    console.print("[dim]Type [bold]/help[/bold] for commands or start chatting![/dim]\n")


@click.command()
@click.version_option(version=__version__, prog_name="QuirkLLM")
@click.option(
    "--profile",
    type=click.Choice(["survival", "comfort", "power", "beast"], case_sensitive=False),
    help="Override automatic profile selection",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode with verbose logging",
)
@click.option(
    "--config",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    help="Path to custom config file (default: ~/.quirkllm/config.yaml)",
)
def main(profile: str | None, debug: bool, config: Path | None) -> None:
    """QuirkLLM - RAM-Aware AI Coding Assistant.

    An adaptive local AI assistant that adjusts its behavior based on
    available system resources.
    """
    try:
        # Detect system resources
        if debug:
            console.print("[dim]🔍 Detecting system resources...[/dim]")

        system_info = detect_system()

        # Select profile (platform-aware)
        profile_config = select_profile(
            system_info=system_info,
            override=profile,
        )

        if debug:
            console.print(f"[dim]✓ System detected: {system_info}[/dim]")
            console.print(f"[dim]✓ Profile selected: {profile_config.name}[/dim]\n")

        # Display welcome banner
        display_welcome_banner(system_info, profile_config)

        # Start REPL
        repl = REPL(
            system_info=system_info,
            profile_config=profile_config,
            debug=debug,
        )
        repl.run()

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted by user[/dim]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        if debug:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
