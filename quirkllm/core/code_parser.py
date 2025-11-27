"""Code block parser for extracting code from LLM responses.

This module provides functionality to parse markdown code blocks from LLM responses
and suggest appropriate filenames based on language and content analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class CodeBlock:
    """Represents a parsed code block from an LLM response.

    Attributes:
        language: Programming language (e.g., "python", "javascript")
        code: The actual code content
        filename: Suggested filename from comment or None
        start_line: Starting line number in the response
        end_line: Ending line number in the response
    """

    language: str
    code: str
    filename: str | None
    start_line: int
    end_line: int


class CodeBlockParser:
    """Parse markdown code blocks from LLM responses.

    Supports formats:
    - ```language
      code
      ```
    - ```language:filename.ext
      code
      ```
    - ```language
      # filename: something.py
      code
      ```
    """

    # Pattern for code blocks with optional language and filename
    # Matches: ```lang:filename\ncode\n``` or ```lang\ncode\n```
    CODE_BLOCK_PATTERN = re.compile(
        r"```(\w+)?(?::([^\n]+))?\n(.*?)```",
        re.DOTALL,
    )

    # Pattern to extract filename from first comment line
    FILENAME_COMMENT_PATTERNS = [
        re.compile(r"^#\s*filename:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^//\s*filename:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^/\*\s*filename:\s*(.+)\s*\*/$", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^<!--\s*filename:\s*(.+)\s*-->$", re.MULTILINE | re.IGNORECASE),
    ]

    # Language to file extension mapping
    LANGUAGE_EXTENSIONS: dict[str, str] = {
        "python": ".py",
        "py": ".py",
        "javascript": ".js",
        "js": ".js",
        "typescript": ".ts",
        "ts": ".ts",
        "jsx": ".jsx",
        "tsx": ".tsx",
        "rust": ".rs",
        "go": ".go",
        "java": ".java",
        "c": ".c",
        "cpp": ".cpp",
        "c++": ".cpp",
        "csharp": ".cs",
        "cs": ".cs",
        "ruby": ".rb",
        "rb": ".rb",
        "php": ".php",
        "swift": ".swift",
        "kotlin": ".kt",
        "scala": ".scala",
        "html": ".html",
        "css": ".css",
        "scss": ".scss",
        "sass": ".sass",
        "less": ".less",
        "json": ".json",
        "yaml": ".yaml",
        "yml": ".yml",
        "toml": ".toml",
        "xml": ".xml",
        "sql": ".sql",
        "shell": ".sh",
        "bash": ".sh",
        "sh": ".sh",
        "zsh": ".zsh",
        "powershell": ".ps1",
        "dockerfile": "Dockerfile",
        "makefile": "Makefile",
        "markdown": ".md",
        "md": ".md",
    }

    def parse(self, text: str) -> list[CodeBlock]:
        """Extract all code blocks from LLM response text.

        Args:
            text: The full LLM response text containing markdown

        Returns:
            List of CodeBlock objects in order of appearance
        """
        blocks: list[CodeBlock] = []

        # Track line numbers
        lines = text.split("\n")
        current_line = 0
        remaining_text = text

        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            # Calculate line numbers
            text_before = text[: match.start()]
            start_line = text_before.count("\n") + 1

            matched_text = match.group(0)
            end_line = start_line + matched_text.count("\n")

            # Extract components
            language = match.group(1) or ""
            inline_filename = match.group(2)
            code = match.group(3)

            # Clean up code (remove trailing whitespace but preserve indentation)
            code = code.rstrip()

            # Try to find filename from inline or comment
            filename = inline_filename
            if not filename:
                filename = self._extract_filename_from_comment(code)

            blocks.append(
                CodeBlock(
                    language=language.lower() if language else "",
                    code=code,
                    filename=filename.strip() if filename else None,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

        return blocks

    def _extract_filename_from_comment(self, code: str) -> str | None:
        """Extract filename from first comment line if present.

        Args:
            code: The code content

        Returns:
            Filename if found, None otherwise
        """
        for pattern in self.FILENAME_COMMENT_PATTERNS:
            match = pattern.search(code)
            if match:
                return match.group(1).strip()
        return None

    def suggest_filename(self, block: CodeBlock) -> str:
        """Suggest a filename based on language and content analysis.

        Args:
            block: The CodeBlock to analyze

        Returns:
            Suggested filename (may include path components)
        """
        # If filename already specified, return it
        if block.filename:
            return block.filename

        # Get extension for language
        extension = self.LANGUAGE_EXTENSIONS.get(block.language, ".txt")

        # Special case: Dockerfile, Makefile don't need extension
        if extension in ("Dockerfile", "Makefile"):
            return extension

        # Try to extract meaningful name from content
        suggested_name = self._analyze_content_for_name(block)

        if suggested_name:
            # Ensure proper extension
            if not suggested_name.endswith(extension):
                suggested_name = f"{suggested_name}{extension}"
            return suggested_name

        # Default fallback
        return f"code{extension}"

    def _analyze_content_for_name(self, block: CodeBlock) -> str | None:
        """Analyze code content to suggest a meaningful filename.

        Args:
            block: The CodeBlock to analyze

        Returns:
            Suggested name without extension, or None
        """
        code = block.code
        language = block.language

        # Python: Look for class or main function
        if language in ("python", "py"):
            # Check for class definition
            class_match = re.search(r"^class\s+(\w+)", code, re.MULTILINE)
            if class_match:
                # Convert CamelCase to snake_case
                name = class_match.group(1)
                name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
                return name

            # Check for main block
            if "if __name__" in code or "def main(" in code:
                return "main"

            # Check for first function definition
            func_match = re.search(r"^def\s+(\w+)", code, re.MULTILINE)
            if func_match:
                return func_match.group(1)

        # JavaScript/TypeScript: Look for React components or exports
        if language in ("javascript", "js", "typescript", "ts", "jsx", "tsx"):
            # React component (function Component or const Component)
            component_match = re.search(
                r"(?:function|const)\s+([A-Z]\w+)\s*(?:\(|=)", code
            )
            if component_match:
                return component_match.group(1)

            # Default export
            export_match = re.search(r"export\s+default\s+(?:function\s+)?(\w+)", code)
            if export_match:
                return export_match.group(1)

        # Rust: Look for mod or main
        if language in ("rust", "rs"):
            if "fn main(" in code:
                return "main"
            mod_match = re.search(r"^mod\s+(\w+)", code, re.MULTILINE)
            if mod_match:
                return mod_match.group(1)

        # Go: Look for package or main
        if language == "go":
            if "func main(" in code:
                return "main"
            pkg_match = re.search(r"^package\s+(\w+)", code, re.MULTILINE)
            if pkg_match and pkg_match.group(1) != "main":
                return pkg_match.group(1)

        # Java: Look for class
        if language == "java":
            class_match = re.search(r"(?:public\s+)?class\s+(\w+)", code)
            if class_match:
                return class_match.group(1)

        return None

    def get_non_code_text(self, text: str) -> list[str]:
        """Extract non-code text segments from response.

        Args:
            text: The full LLM response text

        Returns:
            List of text segments that are not inside code blocks
        """
        segments: list[str] = []
        last_end = 0

        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            # Get text before this code block
            before = text[last_end : match.start()].strip()
            if before:
                segments.append(before)
            last_end = match.end()

        # Get remaining text after last code block
        after = text[last_end:].strip()
        if after:
            segments.append(after)

        return segments
