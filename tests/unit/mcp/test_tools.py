"""
Tests for MCP Tool Registry.

Test Categories:
1. Tool Registry Tests (3)
2. Core Tool Tests (7)
3. Expanded Tool Tests (6)

Total: 16 tests
"""

import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import pytest

from quirkllm.mcp.tools import ToolRegistry, ToolDefinition


def run_async(coro):
    """Helper to run async code in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def registry():
    """Create a fresh tool registry."""
    return ToolRegistry()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file."""
    file_path = temp_dir / "test.txt"
    file_path.write_text("Hello, World!\nLine 2\nLine 3")
    return file_path


# =============================================================================
# 1. Tool Registry Tests (3)
# =============================================================================


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_tool(self, registry):
        """Test registering a custom tool."""
        def custom_handler(args):
            return "custom result"

        tool = ToolDefinition(
            name="custom_tool",
            description="A custom tool",
            inputSchema={"type": "object"},
        )

        registry.register(tool, custom_handler)

        assert "custom_tool" in registry.tools
        assert "custom_tool" in registry.handlers

    def test_list_tools(self, registry):
        """Test listing all tools."""
        tools = registry.list_tools()

        assert isinstance(tools, list)
        assert len(tools) == 8  # 5 core + 3 expanded

        tool_names = [t["name"] for t in tools]
        assert "analyze_project" in tool_names
        assert "search_code" in tool_names
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "execute_command" in tool_names
        assert "search_documents" in tool_names

    def test_call_tool_unknown(self, registry):
        """Test calling unknown tool returns error."""
        result = run_async(registry.call_tool("nonexistent", {}))

        assert result["isError"] is True
        assert "Unknown tool" in result["content"][0]["text"]

    def test_unregister_tool(self, registry):
        """Test unregistering a tool."""
        # Register a custom tool
        tool = ToolDefinition(
            name="temp_tool",
            description="Temporary",
            inputSchema={},
        )
        registry.register(tool, lambda x: x)

        assert "temp_tool" in registry.tools

        # Unregister
        result = registry.unregister("temp_tool")

        assert result is True
        assert "temp_tool" not in registry.tools


# =============================================================================
# 2. Core Tool Tests (7)
# =============================================================================


class TestCoreTools:
    """Tests for core tools (5 tools)."""

    def test_analyze_project(self, registry, temp_dir):
        """Test analyze_project tool."""
        # Create some test files
        (temp_dir / "test.py").write_text("print('hello')")
        (temp_dir / "subdir").mkdir()

        result = run_async(registry.call_tool("analyze_project", {"path": str(temp_dir)}))

        assert "isError" not in result or not result.get("isError")
        assert "content" in result

    def test_analyze_project_not_found(self, registry):
        """Test analyze_project with non-existent path."""
        result = run_async(registry._analyze_project({"path": "/nonexistent/path"}))

        assert "error" in result

    def test_get_context(self, registry):
        """Test get_context tool."""
        result = run_async(registry.call_tool("get_context", {}))

        assert "content" in result
        # Should contain some context info

    def test_read_file_success(self, registry, temp_file):
        """Test read_file tool with valid file."""
        result = run_async(registry.call_tool("read_file", {"path": str(temp_file)}))

        assert "content" in result
        content = result["content"][0]["text"]
        assert "Hello, World!" in content

    def test_read_file_not_found(self, registry):
        """Test read_file with non-existent file."""
        result = run_async(registry._read_file({"path": "/nonexistent/file.txt"}))

        assert "Error" in result
        assert "not found" in result.lower()

    def test_list_files(self, registry, temp_dir):
        """Test list_files tool."""
        # Create test files
        (temp_dir / "file1.txt").write_text("a")
        (temp_dir / "file2.py").write_text("b")

        result = run_async(registry._list_files({"path": str(temp_dir)}))

        assert isinstance(result, list)
        assert "file1.txt" in result
        assert "file2.py" in result

    def test_list_files_pattern(self, registry, temp_dir):
        """Test list_files with pattern filter."""
        # Create test files
        (temp_dir / "file1.txt").write_text("a")
        (temp_dir / "file2.py").write_text("b")

        result = run_async(registry._list_files({
            "path": str(temp_dir),
            "pattern": "*.py",
        }))

        assert "file2.py" in result
        assert "file1.txt" not in result


# =============================================================================
# 3. Expanded Tool Tests (6)
# =============================================================================


class TestExpandedTools:
    """Tests for expanded tools (3 tools)."""

    def test_write_file_success(self, registry, temp_dir):
        """Test write_file tool."""
        file_path = temp_dir / "new_file.txt"

        result = run_async(registry._write_file({
            "path": str(file_path),
            "content": "Test content",
        }))

        assert result["success"] is True
        assert file_path.exists()
        assert file_path.read_text() == "Test content"

    def test_write_file_safety_blocked(self, registry):
        """Test write_file blocks dangerous paths."""
        result = run_async(registry._write_file({
            "path": "/etc/passwd",
            "content": "malicious",
        }))

        assert result["success"] is False
        # Should be blocked by safety check

    def test_execute_command_success(self, registry):
        """Test execute_command with safe command."""
        result = run_async(registry._execute_command({
            "command": "echo hello",
        }))

        assert result["success"] is True
        assert "hello" in result["stdout"]

    def test_execute_command_blocked(self, registry):
        """Test execute_command blocks dangerous commands."""
        result = run_async(registry._execute_command({
            "command": "rm -rf /",
        }))

        assert result["success"] is False
        # Should be blocked by safety check

    def test_execute_command_timeout(self, registry):
        """Test execute_command with timeout."""
        result = run_async(registry._execute_command({
            "command": "sleep 10",
            "timeout": 1,
        }))

        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    def test_search_documents(self, registry):
        """Test search_documents tool."""
        # This will fail gracefully if RAG not available
        result = run_async(registry._search_documents({"query": "test query"}))

        assert isinstance(result, list)
        # Either returns results or error


# =============================================================================
# Additional Tests
# =============================================================================


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_tool_definition_to_dict(self):
        """Test ToolDefinition.to_dict()."""
        tool = ToolDefinition(
            name="test",
            description="A test tool",
            inputSchema={"type": "object", "properties": {}},
        )

        result = tool.to_dict()

        assert result["name"] == "test"
        assert result["description"] == "A test tool"
        assert result["inputSchema"] == {"type": "object", "properties": {}}


class TestToolCallFormatting:
    """Tests for tool call result formatting."""

    def test_call_tool_string_result(self, registry):
        """Test call_tool formats string results."""

        async def string_handler(args):
            return "simple string"

        tool = ToolDefinition(name="string_tool", description="", inputSchema={})
        registry.register(tool, string_handler)

        result = run_async(registry.call_tool("string_tool", {}))

        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "simple string"

    def test_call_tool_dict_result(self, registry):
        """Test call_tool formats dict results as JSON."""

        async def dict_handler(args):
            return {"key": "value", "number": 42}

        tool = ToolDefinition(name="dict_tool", description="", inputSchema={})
        registry.register(tool, dict_handler)

        result = run_async(registry.call_tool("dict_tool", {}))

        import json
        content = result["content"][0]["text"]
        parsed = json.loads(content)
        assert parsed["key"] == "value"
        assert parsed["number"] == 42

    def test_call_tool_exception(self, registry):
        """Test call_tool handles exceptions."""

        async def error_handler(args):
            raise RuntimeError("Something went wrong")

        tool = ToolDefinition(name="error_tool", description="", inputSchema={})
        registry.register(tool, error_handler)

        result = run_async(registry.call_tool("error_tool", {}))

        assert result["isError"] is True
        assert "Something went wrong" in result["content"][0]["text"]
