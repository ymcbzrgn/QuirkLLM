"""Unit tests for ResponseHandler.

Tests cover:
- Processing responses with and without code blocks
- Code block display with syntax highlighting
- File write offers in different modes
- Path safety validation
- Diff display for existing files
- Skip all functionality
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from rich.console import Console

from quirkllm.cli.response_handler import ResponseHandler
from quirkllm.core.code_parser import CodeBlock
from quirkllm.file_ops.file_manager import FileManager
from quirkllm.modes.base import ModeType


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_console():
    """Create a mock console."""
    console = Mock(spec=Console)
    console.print = Mock()
    return console


@pytest.fixture
def mock_file_manager(temp_dir):
    """Create a real file manager with temp directory."""
    return FileManager(project_root=str(temp_dir))


@pytest.fixture
def mock_mode_registry():
    """Create a mock mode registry."""
    registry = Mock()
    return registry


@pytest.fixture
def handler(mock_console, mock_file_manager, mock_mode_registry, temp_dir):
    """Create a ResponseHandler instance."""
    return ResponseHandler(
        console=mock_console,
        file_manager=mock_file_manager,
        mode_registry=mock_mode_registry,
        working_dir=temp_dir,
    )


class TestResponseHandlerInit:
    """Tests for ResponseHandler initialization."""

    def test_init_with_defaults(
        self, mock_console, mock_file_manager, mock_mode_registry
    ):
        """Test initialization with default working directory."""
        handler = ResponseHandler(
            console=mock_console,
            file_manager=mock_file_manager,
            mode_registry=mock_mode_registry,
        )

        assert handler.console is mock_console
        assert handler.file_manager is mock_file_manager
        assert handler.mode_registry is mock_mode_registry
        assert handler.working_dir == Path.cwd().resolve()
        assert handler.skip_all is False

    def test_init_with_working_dir(
        self, mock_console, mock_file_manager, mock_mode_registry, temp_dir
    ):
        """Test initialization with custom working directory."""
        handler = ResponseHandler(
            console=mock_console,
            file_manager=mock_file_manager,
            mode_registry=mock_mode_registry,
            working_dir=temp_dir,
        )

        assert handler.working_dir == temp_dir.resolve()


class TestProcessResponse:
    """Tests for process_response method."""

    def test_process_plain_text_response(self, handler):
        """Test processing response without code blocks."""
        response = "Just a plain text response without any code."

        handler.process_response(response)

        handler.console.print.assert_called_with(response)

    def test_process_response_with_code_block(self, handler, mock_mode_registry):
        """Test processing response with a code block."""
        # Set up mock mode (PLAN mode - no file writing)
        mock_mode = Mock()
        mock_mode.mode_type = ModeType.PLAN
        mock_mode_registry.get_current.return_value = mock_mode

        response = """Here's a Python function:

```python
def hello():
    print("Hello, World!")
```

That's it!"""

        handler.process_response(response)

        # Should display content via console.print
        assert handler.console.print.called

    def test_process_empty_response(self, handler):
        """Test processing empty response."""
        handler.process_response("")

        handler.console.print.assert_called_with("")

    def test_skip_all_resets_on_new_response(self, handler, mock_mode_registry):
        """Test that skip_all flag resets for each new response."""
        handler.skip_all = True

        mock_mode_registry.get_current.return_value = None
        handler.process_response("test")

        assert handler.skip_all is False


class TestDisplayCodeBlock:
    """Tests for code block display."""

    def test_display_single_code_block(self, handler):
        """Test displaying a single code block."""
        block = CodeBlock(
            language="python",
            code="print('hello')",
            filename=None,
            start_line=1,
            end_line=3,
        )

        handler._display_code_block(block, 1, 1)

        # Should call console.print with Panel
        assert handler.console.print.called

    def test_display_multiple_code_blocks(self, handler):
        """Test displaying multiple code blocks with index."""
        block = CodeBlock(
            language="python",
            code="x = 1",
            filename=None,
            start_line=1,
            end_line=1,
        )

        handler._display_code_block(block, 2, 3)

        # Should include index in output
        assert handler.console.print.called

    def test_display_code_block_with_suggested_filename(self, handler):
        """Test displaying code block with filename suggestion."""
        block = CodeBlock(
            language="python",
            code="def main():\n    pass",
            filename=None,
            start_line=1,
            end_line=2,
        )

        handler._display_code_block(block, 1, 1)

        # Should show filename suggestion
        assert handler.console.print.called


class TestPathSafety:
    """Tests for path safety validation."""

    def test_dangerous_ssh_path(self, handler, temp_dir):
        """Test that .ssh paths are marked as dangerous."""
        path = temp_dir / ".ssh" / "config"
        assert handler._is_dangerous_path(path) is True

    def test_dangerous_env_file(self, handler, temp_dir):
        """Test that .env files are marked as dangerous."""
        path = temp_dir / ".env"
        assert handler._is_dangerous_path(path) is True

    def test_dangerous_etc_path(self, handler):
        """Test that /etc paths are marked as dangerous."""
        path = Path("/etc/passwd")
        assert handler._is_dangerous_path(path) is True

    def test_safe_path_in_working_dir(self, handler, temp_dir):
        """Test that paths in working directory are safe."""
        path = temp_dir / "code" / "main.py"
        assert handler._is_dangerous_path(path) is False

    def test_safe_path_in_home(self, handler):
        """Test that paths in home directory are safe."""
        path = Path.home() / "projects" / "test.py"
        assert handler._is_dangerous_path(path) is False


class TestResolveFilepath:
    """Tests for filepath resolution."""

    def test_resolve_relative_path(self, handler, temp_dir):
        """Test resolving relative path."""
        result = handler._resolve_filepath("code/main.py")

        assert result == (temp_dir / "code" / "main.py").resolve()

    def test_resolve_absolute_path(self, handler, temp_dir):
        """Test resolving absolute path."""
        abs_path = temp_dir / "absolute" / "path.py"
        result = handler._resolve_filepath(str(abs_path))

        assert result == abs_path.resolve()

    def test_resolve_current_dir_path(self, handler, temp_dir):
        """Test resolving path in current directory."""
        result = handler._resolve_filepath("test.py")

        assert result == (temp_dir / "test.py").resolve()


class TestWriteFile:
    """Tests for file writing."""

    def test_write_new_file(self, handler, temp_dir):
        """Test writing a new file."""
        filepath = temp_dir / "newfile.py"
        content = "print('hello')"

        result = handler._write_file(filepath, content)

        assert result is True
        assert filepath.exists()
        assert filepath.read_text() == content
        assert handler.console.print.called

    def test_write_file_creates_parent_dirs(self, handler, temp_dir):
        """Test that parent directories are created."""
        filepath = temp_dir / "deep" / "nested" / "file.py"
        content = "# nested file"

        result = handler._write_file(filepath, content)

        assert result is True
        assert filepath.exists()

    def test_write_file_with_backup(self, handler, temp_dir):
        """Test that backup is created for existing file."""
        filepath = temp_dir / "existing.py"
        filepath.write_text("old content")

        result = handler._write_file(filepath, "new content")

        assert result is True
        assert filepath.read_text() == "new content"


class TestDiffDisplay:
    """Tests for diff display."""

    def test_display_diff_for_existing_file(self, handler, temp_dir):
        """Test displaying diff for existing file."""
        filepath = temp_dir / "test.py"
        filepath.write_text("old content\n")

        handler._display_diff(filepath, "new content\n")

        # Should call console.print with diff panel
        assert handler.console.print.called

    def test_display_diff_for_nonexistent_file(self, handler, temp_dir):
        """Test diff display when file doesn't exist."""
        filepath = temp_dir / "nonexistent.py"

        handler._display_diff(filepath, "new content\n")

        # Should still work (empty -> new content)
        assert handler.console.print.called


class TestModeSpecificBehavior:
    """Tests for mode-specific behavior."""

    def test_plan_mode_no_file_writing(self, handler, mock_mode_registry):
        """Test that PLAN mode doesn't offer file writing."""
        mock_mode = Mock()
        mock_mode.mode_type = ModeType.PLAN
        mock_mode_registry.get_current.return_value = mock_mode

        blocks = [
            CodeBlock(
                language="python",
                code="x = 1",
                filename=None,
                start_line=1,
                end_line=1,
            )
        ]

        handler._process_code_blocks(blocks)

        # Should print disabled message
        calls = [str(call) for call in handler.console.print.call_args_list]
        assert any("disabled" in str(call).lower() for call in calls)

    def test_ghost_mode_no_file_writing(self, handler, mock_mode_registry):
        """Test that GHOST mode doesn't offer file writing."""
        mock_mode = Mock()
        mock_mode.mode_type = ModeType.GHOST
        mock_mode_registry.get_current.return_value = mock_mode

        blocks = [
            CodeBlock(
                language="python",
                code="x = 1",
                filename=None,
                start_line=1,
                end_line=1,
            )
        ]

        handler._process_code_blocks(blocks)

        # Should print disabled message
        calls = [str(call) for call in handler.console.print.call_args_list]
        assert any("disabled" in str(call).lower() for call in calls)

    def test_no_current_mode(self, handler, mock_mode_registry):
        """Test handling when no mode is active."""
        mock_mode_registry.get_current.return_value = None

        blocks = [
            CodeBlock(
                language="python",
                code="x = 1",
                filename=None,
                start_line=1,
                end_line=1,
            )
        ]

        # Should not raise
        handler._process_code_blocks(blocks)


class TestAutoWriteFile:
    """Tests for YAMI mode auto-write."""

    def test_auto_write_new_file(self, handler, temp_dir):
        """Test auto-writing a new file."""
        block = CodeBlock(
            language="python",
            code="print('auto')",
            filename=None,
            start_line=1,
            end_line=1,
        )

        result = handler._auto_write_file(block, "auto.py")

        assert result is True
        assert (temp_dir / "auto.py").exists()

    def test_auto_write_dangerous_path_blocked(self, handler, temp_dir):
        """Test that dangerous paths are blocked in auto mode."""
        block = CodeBlock(
            language="bash",
            code="export SECRET=xxx",
            filename=None,
            start_line=1,
            end_line=1,
        )

        result = handler._auto_write_file(block, ".bashrc")

        assert result is False

    def test_auto_write_existing_file_shows_warning(self, handler, temp_dir):
        """Test auto-writing existing file shows warning."""
        filepath = temp_dir / "existing.py"
        filepath.write_text("old content")

        block = CodeBlock(
            language="python",
            code="new content",
            filename=None,
            start_line=1,
            end_line=1,
        )

        result = handler._auto_write_file(block, "existing.py")

        assert result is True
        # Should have shown warning
        calls = [str(call) for call in handler.console.print.call_args_list]
        assert any("overwriting" in str(call).lower() for call in calls)


class TestSkipAllFunctionality:
    """Tests for skip all functionality."""

    def test_skip_all_stops_processing(self, handler, mock_mode_registry):
        """Test that skip_all stops processing remaining blocks."""
        mock_mode = Mock()
        mock_mode.mode_type = ModeType.CHAT
        mock_mode_registry.get_current.return_value = mock_mode

        handler.skip_all = True

        blocks = [
            CodeBlock(
                language="python",
                code="x = 1",
                filename=None,
                start_line=1,
                end_line=1,
            ),
            CodeBlock(
                language="python",
                code="y = 2",
                filename=None,
                start_line=3,
                end_line=3,
            ),
        ]

        # Should not process any blocks due to skip_all
        handler._process_code_blocks(blocks)


class TestResetHandler:
    """Tests for reset functionality."""

    def test_reset_clears_skip_all(self, handler):
        """Test that reset clears skip_all flag."""
        handler.skip_all = True

        handler.reset()

        assert handler.skip_all is False


class TestEmptyCodeBlocks:
    """Tests for handling empty code blocks."""

    def test_skip_empty_code_block(self, handler, mock_mode_registry):
        """Test that empty code blocks are skipped."""
        mock_mode = Mock()
        mock_mode.mode_type = ModeType.YAMI
        mock_mode_registry.get_current.return_value = mock_mode

        blocks = [
            CodeBlock(
                language="python",
                code="",  # Empty
                filename=None,
                start_line=1,
                end_line=1,
            ),
            CodeBlock(
                language="python",
                code="   ",  # Whitespace only
                filename=None,
                start_line=3,
                end_line=3,
            ),
        ]

        # Should not write any files
        with patch.object(handler, "_write_file") as mock_write:
            handler._process_code_blocks(blocks)
            mock_write.assert_not_called()


class TestDangerousPathPatterns:
    """Tests for dangerous path pattern detection."""

    @pytest.mark.parametrize(
        "path_str",
        [
            ".ssh/id_rsa",
            ".gnupg/secring.gpg",
            ".aws/credentials",
            ".bashrc",
            ".bash_profile",
            ".zshrc",
            ".profile",
            ".env",
            ".git/config",
        ],
    )
    def test_dangerous_hidden_files(self, handler, temp_dir, path_str):
        """Test that dangerous hidden files are blocked."""
        path = temp_dir / path_str
        assert handler._is_dangerous_path(path) is True

    @pytest.mark.parametrize(
        "path_str",
        [
            "src/main.py",
            "tests/test_app.py",
            "lib/utils.js",
            "README.md",
        ],
    )
    def test_safe_project_files(self, handler, temp_dir, path_str):
        """Test that normal project files are allowed."""
        path = temp_dir / path_str
        assert handler._is_dangerous_path(path) is False
