"""Unit tests for FileContextManager.

Tests cover:
- File loading and token management
- Directory listing generation
- Context prompt building
- Auto-detection of file references
- Language detection from extensions
"""

import tempfile
from pathlib import Path

import pytest

from quirkllm.core.context_manager import (
    FileContext,
    FileContextManager,
    DirectoryEntry,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # Create test files
        (base / "main.py").write_text("def main():\n    print('hello')\n")
        (base / "utils.py").write_text("def helper(): pass\n")
        (base / "test.js").write_text("function test() {}\n")
        (base / "README.md").write_text("# Project\n")

        # Create subdirectory
        (base / "src").mkdir()
        (base / "src" / "app.py").write_text("# app code\n")

        yield base


@pytest.fixture
def manager(temp_dir):
    """Create a FileContextManager instance."""
    return FileContextManager(working_dir=temp_dir, max_context_tokens=1000)


class TestFileContextManagerInit:
    """Tests for FileContextManager initialization."""

    def test_init_default(self):
        """Test initialization with defaults."""
        mgr = FileContextManager()
        assert mgr.working_dir == Path.cwd().resolve()
        assert mgr.max_context_tokens == 4000
        assert len(mgr.loaded_files) == 0

    def test_init_with_custom_dir(self, temp_dir):
        """Test initialization with custom directory."""
        mgr = FileContextManager(working_dir=temp_dir, max_context_tokens=2000)
        assert mgr.working_dir == temp_dir.resolve()
        assert mgr.max_context_tokens == 2000


class TestFileLoading:
    """Tests for file loading functionality."""

    def test_load_existing_file(self, manager, temp_dir):
        """Test loading an existing file."""
        result = manager.load_file("main.py")

        assert result is not None
        assert result.path == "main.py"
        assert "def main()" in result.content
        assert result.language == "python"
        assert result.line_count == 2

    def test_load_file_with_relative_path(self, manager, temp_dir):
        """Test loading file in subdirectory."""
        result = manager.load_file("src/app.py")

        assert result is not None
        assert result.path == "src/app.py"

    def test_load_nonexistent_file(self, manager):
        """Test loading a file that doesn't exist."""
        result = manager.load_file("nonexistent.py")
        assert result is None

    def test_load_file_tracks_tokens(self, manager, temp_dir):
        """Test that loading files tracks token usage."""
        initial = manager.total_tokens
        manager.load_file("main.py")

        assert manager.total_tokens > initial

    def test_load_file_exceeds_limit(self, temp_dir):
        """Test that loading fails when token limit exceeded."""
        # Create a large file
        large_content = "x = 1\n" * 1000
        (temp_dir / "large.py").write_text(large_content)

        mgr = FileContextManager(working_dir=temp_dir, max_context_tokens=100)
        result = mgr.load_file("large.py")

        assert result is None

    def test_load_same_file_twice_returns_cached(self, manager, temp_dir):
        """Test that loading same file twice returns cached version."""
        result1 = manager.load_file("main.py")
        result2 = manager.load_file("main.py")

        assert result1 is result2

    def test_unload_file(self, manager, temp_dir):
        """Test unloading a file."""
        manager.load_file("main.py")
        assert "main.py" in manager.loaded_files

        result = manager.unload_file("main.py")

        assert result is True
        assert "main.py" not in manager.loaded_files

    def test_unload_nonexistent_file(self, manager):
        """Test unloading a file that's not loaded."""
        result = manager.unload_file("nonexistent.py")
        assert result is False

    def test_clear_files(self, manager, temp_dir):
        """Test clearing all loaded files."""
        manager.load_file("main.py")
        manager.load_file("utils.py")
        assert len(manager.loaded_files) == 2

        manager.clear_files()

        assert len(manager.loaded_files) == 0
        assert manager.total_tokens == 0


class TestLanguageDetection:
    """Tests for language detection."""

    def test_detect_python(self, manager):
        """Test Python file detection."""
        assert manager._detect_language(Path("test.py")) == "python"

    def test_detect_javascript(self, manager):
        """Test JavaScript file detection."""
        assert manager._detect_language(Path("test.js")) == "javascript"

    def test_detect_typescript(self, manager):
        """Test TypeScript file detection."""
        assert manager._detect_language(Path("test.ts")) == "typescript"

    def test_detect_unknown(self, manager):
        """Test unknown extension returns empty string."""
        assert manager._detect_language(Path("test.xyz")) == ""


class TestDirectoryListing:
    """Tests for directory listing functionality."""

    def test_get_cwd_listing(self, manager, temp_dir):
        """Test getting directory listing."""
        entries = manager.get_cwd_listing()

        assert len(entries) > 0
        names = [e.name for e in entries]
        assert "main.py" in names
        assert "src/" in names

    def test_listing_excludes_ignored(self, manager, temp_dir):
        """Test that ignored files are excluded."""
        # Create ignored files
        (temp_dir / "__pycache__").mkdir()
        (temp_dir / ".git").mkdir()

        entries = manager.get_cwd_listing()
        names = [e.name for e in entries]

        assert "__pycache__/" not in names
        assert ".git/" not in names

    def test_get_directory_listing_text(self, manager, temp_dir):
        """Test formatted directory listing."""
        text = manager.get_directory_listing_text()

        assert "main.py" in text
        assert "python" in text.lower()
        assert "📁" in text  # Directory emoji
        assert "📄" in text  # File emoji


class TestContextPrompt:
    """Tests for context prompt building."""

    def test_get_file_context_prompt_empty(self, manager):
        """Test context prompt with no loaded files."""
        prompt = manager.get_file_context_prompt()

        assert "<directory_listing>" in prompt
        assert "</directory_listing>" in prompt

    def test_get_file_context_prompt_with_file(self, manager, temp_dir):
        """Test context prompt with loaded file."""
        manager.load_file("main.py")
        prompt = manager.get_file_context_prompt()

        assert "<file" in prompt
        assert "main.py" in prompt
        assert "def main()" in prompt
        assert "</file>" in prompt

    def test_get_loaded_files_summary(self, manager, temp_dir):
        """Test loaded files summary."""
        manager.load_file("main.py")
        summary = manager.get_loaded_files_summary()

        assert "main.py" in summary
        assert "lines" in summary
        assert "tokens" in summary


class TestAutoDetection:
    """Tests for auto-detection of file references."""

    def test_auto_detect_simple_filename(self, manager, temp_dir):
        """Test detecting simple filename."""
        result = manager.auto_detect_files("please look at main.py")

        assert "main.py" in result

    def test_auto_detect_path(self, manager, temp_dir):
        """Test detecting file path."""
        result = manager.auto_detect_files("check src/app.py")

        assert "src/app.py" in result

    def test_auto_detect_nonexistent_file(self, manager, temp_dir):
        """Test that nonexistent files are not detected."""
        result = manager.auto_detect_files("look at missing.py")

        assert "missing.py" not in result

    def test_auto_detect_multiple_files(self, manager, temp_dir):
        """Test detecting multiple files."""
        result = manager.auto_detect_files("compare main.py and utils.py")

        assert "main.py" in result
        assert "utils.py" in result


class TestTokenTracking:
    """Tests for token tracking properties."""

    def test_total_tokens(self, manager, temp_dir):
        """Test total tokens property."""
        assert manager.total_tokens == 0
        manager.load_file("main.py")
        assert manager.total_tokens > 0

    def test_remaining_tokens(self, manager, temp_dir):
        """Test remaining tokens property."""
        initial = manager.remaining_tokens
        manager.load_file("main.py")

        assert manager.remaining_tokens < initial
        assert manager.remaining_tokens == manager.max_context_tokens - manager.total_tokens


class TestIgnorePatterns:
    """Tests for ignore pattern matching."""

    def test_should_ignore_pycache(self, manager):
        """Test __pycache__ is ignored."""
        assert manager._should_ignore("__pycache__") is True

    def test_should_ignore_git(self, manager):
        """Test .git is ignored."""
        assert manager._should_ignore(".git") is True

    def test_should_ignore_pyc(self, manager):
        """Test *.pyc files are ignored."""
        assert manager._should_ignore("test.pyc") is True

    def test_should_not_ignore_py(self, manager):
        """Test .py files are not ignored."""
        assert manager._should_ignore("test.py") is False
