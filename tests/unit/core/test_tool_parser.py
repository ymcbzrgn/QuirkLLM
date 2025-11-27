"""Unit tests for ToolParser.

Tests cover:
- Pattern detection for [READ:], [LS:], [SEARCH:]
- Tool call extraction from model output
- Removal of tool calls from text
- Edge cases and various formatting
"""

import pytest

from quirkllm.core.tool_parser import ToolParser, ToolType, ToolCall


@pytest.fixture
def parser():
    """Create a ToolParser instance."""
    return ToolParser()


class TestToolParserInit:
    """Tests for ToolParser initialization."""

    def test_init(self, parser):
        """Test basic initialization."""
        assert parser.READ_PATTERN is not None
        assert parser.LS_PATTERN is not None
        assert parser.SEARCH_PATTERN is not None


class TestReadPattern:
    """Tests for [READ: path] pattern detection."""

    def test_read_simple_filename(self, parser):
        """Test reading simple filename."""
        text = "[READ: main.py]"
        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0].tool_type == ToolType.READ
        assert calls[0].argument == "main.py"

    def test_read_with_path(self, parser):
        """Test reading file with path."""
        text = "[READ: src/utils/helper.py]"
        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0].argument == "src/utils/helper.py"

    def test_read_in_sentence(self, parser):
        """Test READ pattern within a sentence."""
        text = "Let me read that file [READ: config.json] for you."
        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0].argument == "config.json"

    def test_read_case_insensitive(self, parser):
        """Test READ is case insensitive."""
        text = "[read: main.py]"
        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0].tool_type == ToolType.READ

    def test_read_with_quotes(self, parser):
        """Test READ with quoted path."""
        text = '[READ: "path with space.py"]'
        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0].argument == "path with space.py"

    def test_read_multiple(self, parser):
        """Test multiple READ calls."""
        text = "[READ: file1.py] and [READ: file2.py]"
        calls = parser.parse(text)

        assert len(calls) == 2
        assert calls[0].argument == "file1.py"
        assert calls[1].argument == "file2.py"


class TestLsPattern:
    """Tests for [LS: path] pattern detection."""

    def test_ls_with_path(self, parser):
        """Test LS with explicit path."""
        text = "[LS: src/]"
        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0].tool_type == ToolType.LS
        assert calls[0].argument == "src/"

    def test_ls_without_path(self, parser):
        """Test LS without path defaults to current dir."""
        text = "[LS]"
        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0].tool_type == ToolType.LS
        assert calls[0].argument == "."

    def test_ls_current_dir(self, parser):
        """Test LS with dot."""
        text = "[LS: .]"
        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0].argument == "."

    def test_ls_case_insensitive(self, parser):
        """Test LS is case insensitive."""
        text = "[ls: tests/]"
        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0].tool_type == ToolType.LS


class TestSearchPattern:
    """Tests for [SEARCH: pattern] detection."""

    def test_search_simple(self, parser):
        """Test simple search pattern."""
        text = "[SEARCH: def main]"
        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0].tool_type == ToolType.SEARCH
        assert calls[0].argument == "def main"

    def test_search_regex(self, parser):
        """Test search with regex-like pattern."""
        text = "[SEARCH: class.*Test]"
        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0].argument == "class.*Test"

    def test_search_quoted(self, parser):
        """Test search with quoted string."""
        text = '[SEARCH: "error handling"]'
        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0].argument == "error handling"

    def test_search_case_insensitive(self, parser):
        """Test SEARCH is case insensitive."""
        text = "[search: TODO]"
        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0].tool_type == ToolType.SEARCH


class TestMixedToolCalls:
    """Tests for mixed tool calls in text."""

    def test_multiple_different_tools(self, parser):
        """Test different tool types in same text."""
        text = """First [LS: src/] then [READ: main.py]
        and finally [SEARCH: TODO]"""
        calls = parser.parse(text)

        assert len(calls) == 3
        types = [c.tool_type for c in calls]
        assert ToolType.LS in types
        assert ToolType.READ in types
        assert ToolType.SEARCH in types

    def test_tools_on_separate_lines(self, parser):
        """Test tools on separate lines."""
        text = """[READ: file1.py]
[READ: file2.py]
[LS]"""
        calls = parser.parse(text)

        assert len(calls) == 3


class TestHasToolCalls:
    """Tests for has_tool_calls method."""

    def test_has_tool_calls_true(self, parser):
        """Test text with tool calls."""
        assert parser.has_tool_calls("[READ: test.py]") is True
        assert parser.has_tool_calls("[LS]") is True
        assert parser.has_tool_calls("[SEARCH: foo]") is True

    def test_has_tool_calls_false(self, parser):
        """Test text without tool calls."""
        assert parser.has_tool_calls("No tool calls here") is False
        assert parser.has_tool_calls("") is False

    def test_has_tool_calls_partial(self, parser):
        """Test text with partial matches that shouldn't match."""
        assert parser.has_tool_calls("[READ test.py]") is False  # Missing colon
        assert parser.has_tool_calls("READ: test.py") is False  # Missing brackets


class TestRemoveToolCalls:
    """Tests for remove_tool_calls method."""

    def test_remove_single_call(self, parser):
        """Test removing single tool call."""
        text = "Here is the file [READ: main.py] content."
        result = parser.remove_tool_calls(text)

        assert "[READ:" not in result
        assert "main.py" not in result
        assert "Here is the file" in result
        assert "content" in result

    def test_remove_multiple_calls(self, parser):
        """Test removing multiple tool calls."""
        text = "[LS] Let me read [READ: test.py] that."
        result = parser.remove_tool_calls(text)

        assert "[LS]" not in result
        assert "[READ:" not in result
        assert "Let me read" in result

    def test_remove_preserves_content(self, parser):
        """Test that non-tool content is preserved."""
        text = "Here is some code:\n```python\nprint('hello')\n```"
        result = parser.remove_tool_calls(text)

        assert result == text

    def test_remove_cleans_whitespace(self, parser):
        """Test that extra whitespace is cleaned."""
        text = "Start\n\n[READ: file.py]\n\n\nEnd"
        result = parser.remove_tool_calls(text)

        # Should not have triple newlines
        assert "\n\n\n" not in result


class TestGetFirstToolCall:
    """Tests for get_first_tool_call method."""

    def test_get_first_with_calls(self, parser):
        """Test getting first tool call."""
        text = "[READ: first.py] and [READ: second.py]"
        result = parser.get_first_tool_call(text)

        assert result is not None
        assert result.argument == "first.py"

    def test_get_first_without_calls(self, parser):
        """Test getting first when no calls present."""
        text = "No tool calls here"
        result = parser.get_first_tool_call(text)

        assert result is None


class TestToolCallDataclass:
    """Tests for ToolCall dataclass."""

    def test_tool_call_attributes(self, parser):
        """Test ToolCall has all expected attributes."""
        text = "[READ: test.py]"
        calls = parser.parse(text)

        call = calls[0]
        assert hasattr(call, 'tool_type')
        assert hasattr(call, 'argument')
        assert hasattr(call, 'raw_match')
        assert call.raw_match == "[READ: test.py]"


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_string(self, parser):
        """Test with empty string."""
        calls = parser.parse("")
        assert len(calls) == 0

    def test_no_tool_calls(self, parser):
        """Test text without any tool calls."""
        text = "Just regular text without any special patterns."
        calls = parser.parse(text)
        assert len(calls) == 0

    def test_malformed_calls(self, parser):
        """Test malformed tool calls are ignored."""
        text = "[READ test.py] [READ:] [LS: "
        calls = parser.parse(text)
        # Should not match malformed patterns
        assert all(c.argument for c in calls)

    def test_nested_brackets(self, parser):
        """Test text with nested brackets."""
        text = "Array [0] and [READ: file.py]"
        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0].argument == "file.py"

    def test_special_characters_in_path(self, parser):
        """Test paths with special characters."""
        text = "[READ: file-name_v2.py]"
        calls = parser.parse(text)

        assert len(calls) == 1
        assert calls[0].argument == "file-name_v2.py"
