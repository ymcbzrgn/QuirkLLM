"""Response handler for processing LLM responses with code extraction and file writing.

This module processes LLM responses, extracts code blocks, displays them with
syntax highlighting, and offers to write them to files based on mode configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.text import Text

from quirkllm.core.code_parser import CodeBlock, CodeBlockParser
from quirkllm.file_ops.file_manager import FileManager
from quirkllm.modes.base import ModeType

if TYPE_CHECKING:
    from quirkllm.modes.registry import ModeRegistry


class ResponseHandler:
    """Handle LLM responses with code extraction and file writing.

    This class processes LLM responses by:
    1. Separating explanatory text from code blocks
    2. Displaying code with syntax highlighting
    3. Offering to write code to files based on mode configuration

    Mode-specific behavior:
    - CHAT: Asks for confirmation before each file write
    - YAMI: Auto-writes files without confirmation (with safety checks)
    - PLAN/GHOST: Display only, no file writing

    Attributes:
        console: Rich console for output rendering
        file_manager: File manager for safe file operations
        mode_registry: Registry to get current mode
        parser: Code block parser
        working_dir: Current working directory for relative paths
        skip_all: Session flag to skip all file write prompts
    """

    def __init__(
        self,
        console: Console,
        file_manager: FileManager,
        mode_registry: "ModeRegistry",
        working_dir: str | Path | None = None,
    ):
        """Initialize response handler.

        Args:
            console: Rich console for output
            file_manager: File manager for file operations
            mode_registry: Mode registry to get current mode
            working_dir: Working directory for relative paths (default: cwd)
        """
        self.console = console
        self.file_manager = file_manager
        self.mode_registry = mode_registry
        self.parser = CodeBlockParser()
        self.working_dir = Path(working_dir or os.getcwd()).resolve()
        self.skip_all = False

    def process_response(self, response: str, user_input: str = "") -> None:
        """Process LLM response, extract code, and offer file writes.

        This is the main entry point for response handling. It:
        1. Extracts non-code text segments and displays them
        2. Parses code blocks from the response
        3. For each code block:
           - Displays with syntax highlighting
           - Offers to write to file (based on mode)

        Args:
            response: The full LLM response text
            user_input: The original user input (for context)
        """
        # Reset skip_all flag for new response
        self.skip_all = False

        # Get non-code text segments
        text_segments = self.parser.get_non_code_text(response)

        # Parse code blocks
        code_blocks = self.parser.parse(response)

        # If no code blocks, just display the response as-is
        if not code_blocks:
            self.console.print(response)
            return

        # Process response by displaying text and code blocks in order
        self._display_mixed_content(response, text_segments, code_blocks)

        # Offer file writes for code blocks (based on mode)
        self._process_code_blocks(code_blocks)

    def _display_mixed_content(
        self,
        response: str,
        text_segments: list[str],
        code_blocks: list[CodeBlock],
    ) -> None:
        """Display response with interleaved text and code blocks.

        Args:
            response: Original response text
            text_segments: Non-code text segments
            code_blocks: Parsed code blocks
        """
        # Simple approach: display text segments first, then code blocks
        # This preserves the logical flow while keeping code blocks visually distinct

        if text_segments:
            for segment in text_segments:
                self.console.print(segment)
                self.console.print()

        for i, block in enumerate(code_blocks):
            self._display_code_block(block, i + 1, len(code_blocks))
            self.console.print()

    def _display_code_block(
        self, block: CodeBlock, index: int, total: int
    ) -> None:
        """Display a single code block with syntax highlighting.

        Args:
            block: The code block to display
            index: Block index (1-based)
            total: Total number of blocks
        """
        # Suggest filename
        suggested_filename = self.parser.suggest_filename(block)

        # Create title with filename suggestion
        if total > 1:
            title = f"[bold]Code Block {index}/{total}[/bold]"
        else:
            title = "[bold]Code[/bold]"

        if suggested_filename:
            title += f" [dim]({suggested_filename})[/dim]"

        # Create syntax highlighted content
        language = block.language or "text"
        syntax = Syntax(
            block.code,
            language,
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        )

        self.console.print(
            Panel(
                syntax,
                title=title,
                border_style="cyan",
                padding=(0, 1),
            )
        )

    def _process_code_blocks(self, code_blocks: list[CodeBlock]) -> None:
        """Process code blocks for file writing based on mode.

        Args:
            code_blocks: List of code blocks to process
        """
        current_mode = self.mode_registry.get_current()

        if not current_mode:
            return

        mode_type = current_mode.mode_type

        # PLAN and GHOST modes: display only, no file writing
        if mode_type in (ModeType.PLAN, ModeType.GHOST):
            self.console.print(
                "[dim]File writing disabled in this mode[/dim]"
            )
            return

        # Process each code block
        for i, block in enumerate(code_blocks):
            if self.skip_all:
                break

            # Skip empty code blocks
            if not block.code.strip():
                continue

            # Offer to write file
            self._offer_file_write(block, mode_type, i + 1, len(code_blocks))

    def _offer_file_write(
        self,
        block: CodeBlock,
        mode_type: ModeType,
        index: int,
        total: int,
    ) -> bool:
        """Offer to write a code block to a file.

        Args:
            block: The code block to write
            mode_type: Current mode type
            index: Block index (1-based)
            total: Total number of blocks

        Returns:
            True if file was written, False otherwise
        """
        # Suggest filename
        suggested = self.parser.suggest_filename(block)

        # YAMI mode: auto-write without confirmation (with safety checks)
        if mode_type == ModeType.YAMI:
            return self._auto_write_file(block, suggested)

        # CHAT mode: ask for confirmation
        return self._confirm_and_write_file(block, suggested, index, total)

    def _auto_write_file(self, block: CodeBlock, suggested: str) -> bool:
        """Auto-write file without confirmation (YAMI mode).

        Args:
            block: Code block to write
            suggested: Suggested filename

        Returns:
            True if written, False otherwise
        """
        filepath = self._resolve_filepath(suggested)

        # Safety check
        if self._is_dangerous_path(filepath):
            self.console.print(
                f"[yellow]⚠️  Skipping dangerous path: {filepath}[/yellow]"
            )
            return False

        # Check if file exists (show warning but still write)
        if filepath.exists():
            # Show diff
            self._display_diff(filepath, block.code)
            self.console.print(
                f"[yellow]⚠️  Overwriting existing file: {filepath}[/yellow]"
            )

        return self._write_file(filepath, block.code)

    def _confirm_and_write_file(
        self,
        block: CodeBlock,
        suggested: str,
        index: int,
        total: int,
    ) -> bool:
        """Ask for confirmation and write file (CHAT mode).

        Args:
            block: Code block to write
            suggested: Suggested filename
            index: Block index
            total: Total blocks

        Returns:
            True if written, False otherwise
        """
        filepath = self._resolve_filepath(suggested)

        # Safety check
        if self._is_dangerous_path(filepath):
            self.console.print(
                f"[red]⛔ Cannot write to dangerous path: {filepath}[/red]"
            )
            return False

        # Check if file exists and show diff
        file_exists = filepath.exists()
        if file_exists:
            self._display_diff(filepath, block.code)

        # Prompt for confirmation
        self.console.print()
        prompt_text = (
            f"[bold]💾 Write to file?[/bold] "
            f"[dim]({index}/{total})[/dim]"
        )
        self.console.print(prompt_text)
        self.console.print(f"   Suggested: [cyan]{filepath}[/cyan]")

        # Get user choice
        while True:
            choice = Prompt.ask(
                "   [y]es / [n]o / [e]dit path / [s]kip all",
                choices=["y", "n", "e", "s"],
                default="n",
            ).lower()

            if choice == "y":
                return self._write_file(filepath, block.code)

            elif choice == "n":
                self.console.print("[dim]Skipped[/dim]")
                return False

            elif choice == "s":
                self.skip_all = True
                self.console.print("[dim]Skipping all remaining files[/dim]")
                return False

            elif choice == "e":
                # Edit path
                new_path = Prompt.ask("   New path", default=str(filepath))
                filepath = self._resolve_filepath(new_path)

                if self._is_dangerous_path(filepath):
                    self.console.print(
                        f"[red]⛔ Cannot write to dangerous path: {filepath}[/red]"
                    )
                    continue

                # Show diff if file exists at new path
                if filepath.exists():
                    self._display_diff(filepath, block.code)

                # Re-prompt for confirmation
                continue

    def _resolve_filepath(self, suggested: str) -> Path:
        """Resolve filepath, making it relative to working directory.

        Args:
            suggested: Suggested filename (may be absolute or relative)

        Returns:
            Resolved absolute path
        """
        path = Path(suggested)

        # If absolute, use as-is
        if path.is_absolute():
            return path.resolve()

        # Otherwise, relative to working directory
        return (self.working_dir / path).resolve()

    def _is_dangerous_path(self, filepath: Path) -> bool:
        """Check if path is dangerous and should be blocked.

        Args:
            filepath: Path to check

        Returns:
            True if dangerous, False otherwise
        """
        # First, check if path is within working directory - always safe
        try:
            filepath.resolve().relative_to(self.working_dir.resolve())
            # Path is in working dir - check for dangerous file names only
            return self._has_dangerous_name(filepath)
        except ValueError:
            pass  # Not in working dir, continue checks

        # Check if in home directory
        try:
            home = Path.home()
            filepath.resolve().relative_to(home)
            # Path is under home - check for dangerous patterns
            return self._has_dangerous_name(filepath)
        except ValueError:
            pass  # Not in home dir

        # Check if in temp directory (allow temp dirs)
        try:
            import tempfile
            temp_root = Path(tempfile.gettempdir()).resolve()
            filepath.resolve().relative_to(temp_root)
            # In temp directory - allow
            return self._has_dangerous_name(filepath)
        except ValueError:
            pass  # Not in temp dir

        # Outside all safe directories - dangerous
        return True

    def _has_dangerous_name(self, filepath: Path) -> bool:
        """Check if any path component is a dangerous name.

        Args:
            filepath: Path to check

        Returns:
            True if contains dangerous names, False otherwise
        """
        # Get all parts of the path
        parts = filepath.parts

        # Dangerous hidden files and directories
        dangerous_names = {
            ".ssh",
            ".gnupg",
            ".aws",
            ".bashrc",
            ".bash_profile",
            ".zshrc",
            ".profile",
            ".env",
        }

        # Check each path component
        for part in parts:
            if part.lower() in dangerous_names:
                return True

        # Check for .git/config specifically
        filepath_str = str(filepath)
        if ".git/config" in filepath_str or ".git\\config" in filepath_str:
            return True

        return False

    def _display_diff(self, filepath: Path, new_content: str) -> None:
        """Display diff between existing file and new content.

        Args:
            filepath: Path to existing file
            new_content: New content to write
        """
        try:
            diff = self.file_manager.generate_diff(str(filepath), new_content)

            if diff:
                syntax = Syntax(
                    diff,
                    "diff",
                    theme="monokai",
                    line_numbers=True,
                )

                self.console.print(
                    Panel(
                        syntax,
                        title="[bold]Diff Preview[/bold]",
                        border_style="yellow",
                        padding=(0, 1),
                    )
                )
        except Exception as e:
            self.console.print(f"[dim]Could not generate diff: {e}[/dim]")

    def _write_file(self, filepath: Path, content: str) -> bool:
        """Write content to file using FileManager.

        Args:
            filepath: Path to write to
            content: Content to write

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write file with backup
            backup = self.file_manager.write_file(
                str(filepath),
                content,
                create_backup=True,
                reason="REPL code write",
            )

            # Success message
            self.console.print(f"[green]✓ Written to: {filepath}[/green]")

            if backup:
                self.console.print(
                    f"[dim]  Backup created: {backup.id}[/dim]"
                )
                self.console.print(
                    f"[dim]  Rollback: /rollback {filepath} {backup.id}[/dim]"
                )

            return True

        except PermissionError:
            self.console.print(
                f"[red]✗ Permission denied: {filepath}[/red]"
            )
            return False

        except Exception as e:
            self.console.print(f"[red]✗ Error writing file: {e}[/red]")
            return False

    def reset(self) -> None:
        """Reset handler state for new session."""
        self.skip_all = False

    def handle_code_blocks(self, response: str, user_input: str = "") -> None:
        """Process code blocks for file writing without re-displaying response.

        This is used with streaming mode where the response was already displayed
        during generation. It only handles code extraction and file writing.

        Args:
            response: The full LLM response text (already displayed)
            user_input: The original user input (for context)
        """
        # Reset skip_all flag for new response
        self.skip_all = False

        # Parse code blocks
        code_blocks = self.parser.parse(response)

        # If no code blocks, nothing to do
        if not code_blocks:
            return

        # Process code blocks for file writing (based on mode)
        self._process_code_blocks(code_blocks)
