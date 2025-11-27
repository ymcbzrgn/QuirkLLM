"""Tool Parser for QuirkLLM Agentic Behavior.

Phase 6.6 - Parses model output for tool calls like [READ: path], [LS: path], [SEARCH: pattern].
These simulated tools allow the model to request file operations during the agentic loop.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ToolType(Enum):
    """Types of tools the model can request."""
    READ = "read"       # Read file contents
    LS = "ls"           # List directory
    SEARCH = "search"   # Search in files


@dataclass
class ToolCall:
    """Represents a tool call extracted from model output.

    Attributes:
        tool_type: Type of tool (READ, LS, SEARCH)
        argument: The argument passed to the tool (path or pattern)
        raw_match: Original matched text from model output
    """
    tool_type: ToolType
    argument: str
    raw_match: str


class ToolParser:
    """Parse model output for tool calls.

    Detects patterns like:
    - [READ: filename.py] or [READ: path/to/file.py]
    - [LS: path] or [LS: .] or [LS]
    - [SEARCH: pattern] or [SEARCH: "pattern with spaces"]

    The parser is case-insensitive and handles various formatting.
    """

    # Pattern for [READ: path]
    READ_PATTERN = re.compile(
        r'\[READ:\s*([^\]]+)\]',
        re.IGNORECASE
    )

    # Pattern for [LS: path] or [LS]
    LS_PATTERN = re.compile(
        r'\[LS(?::\s*([^\]]*))?\]',
        re.IGNORECASE
    )

    # Pattern for [SEARCH: pattern]
    SEARCH_PATTERN = re.compile(
        r'\[SEARCH:\s*([^\]]+)\]',
        re.IGNORECASE
    )

    def parse(self, text: str) -> list[ToolCall]:
        """Parse text for tool calls.

        Args:
            text: Model output text to parse

        Returns:
            List of ToolCall objects found in the text
        """
        tool_calls: list[ToolCall] = []

        # Find READ calls
        for match in self.READ_PATTERN.finditer(text):
            argument = match.group(1).strip().strip('"\'')
            tool_calls.append(ToolCall(
                tool_type=ToolType.READ,
                argument=argument,
                raw_match=match.group(0),
            ))

        # Find LS calls
        for match in self.LS_PATTERN.finditer(text):
            argument = (match.group(1) or ".").strip().strip('"\'')
            tool_calls.append(ToolCall(
                tool_type=ToolType.LS,
                argument=argument,
                raw_match=match.group(0),
            ))

        # Find SEARCH calls
        for match in self.SEARCH_PATTERN.finditer(text):
            argument = match.group(1).strip().strip('"\'')
            tool_calls.append(ToolCall(
                tool_type=ToolType.SEARCH,
                argument=argument,
                raw_match=match.group(0),
            ))

        return tool_calls

    def has_tool_calls(self, text: str) -> bool:
        """Check if text contains any tool calls.

        Args:
            text: Model output text to check

        Returns:
            True if any tool calls found
        """
        return bool(
            self.READ_PATTERN.search(text) or
            self.LS_PATTERN.search(text) or
            self.SEARCH_PATTERN.search(text)
        )

    def remove_tool_calls(self, text: str) -> str:
        """Remove tool call patterns from text.

        Useful for cleaning model output before display.

        Args:
            text: Text with tool calls

        Returns:
            Text with tool calls removed
        """
        result = text
        result = self.READ_PATTERN.sub('', result)
        result = self.LS_PATTERN.sub('', result)
        result = self.SEARCH_PATTERN.sub('', result)
        # Clean up extra whitespace
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
        return result.strip()

    def get_first_tool_call(self, text: str) -> Optional[ToolCall]:
        """Get the first tool call in text.

        Args:
            text: Model output text

        Returns:
            First ToolCall found, or None
        """
        calls = self.parse(text)
        return calls[0] if calls else None
