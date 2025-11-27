"""
Context Manager for QuirkLLM
Phase 2.4 - Context window tracking, token counting, compaction logic

Tracks conversation context, monitors token usage, provides warnings
when approaching context limits, and handles context compaction.
"""
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ContextWarningLevel(Enum):
    """Context window warning levels."""
    NONE = "none"           # < 70% full
    LOW = "low"             # 70-80% full
    MEDIUM = "medium"       # 80-90% full
    HIGH = "high"           # 90-95% full
    CRITICAL = "critical"   # > 95% full


@dataclass
class Message:
    """Represents a conversation message.
    
    Attributes:
        role: Message role (system, user, assistant)
        content: Message text content
        tokens: Approximate token count
    """
    role: str
    content: str
    tokens: int = 0
    
    def __post_init__(self):
        """Calculate tokens if not provided."""
        if self.tokens == 0:
            self.tokens = estimate_tokens(self.content)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.
    
    Uses approximate conversion: 1 token ≈ 0.75 words (common for English text).
    For code, uses character-based estimation: 1 token ≈ 4 characters.
    
    Args:
        text: Input text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Detect if text looks like code (contains code indicators)
    code_indicators = ['def ', 'class ', 'import ', 'function', '```', '{', '}', 'const ', 'let ', 'var ']
    is_code = any(indicator in text for indicator in code_indicators)
    
    if is_code:
        # Code: ~4 chars per token
        return max(1, len(text) // 4)
    else:
        # Natural language: ~0.75 words per token (or 1.33 tokens per word)
        words = len(text.split())
        return max(1, int(words * 1.33))


class ContextManager:
    """
    Manages conversation context and token tracking.
    
    Features:
    - Token counting for messages
    - Context window monitoring
    - Warning system for approaching limits
    - Context compaction (truncate oldest messages)
    - Profile-based limits
    """
    
    def __init__(self, max_context_length: int):
        """
        Initialize context manager.
        
        Args:
            max_context_length: Maximum context window size in tokens
        """
        self.max_context_length = max_context_length
        self.messages: list[Message] = []
        self._current_tokens = 0
        self._system_prompt_tokens = 0
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to context.
        
        Args:
            role: Message role (system, user, assistant)
            content: Message content
        """
        message = Message(role=role, content=content)
        self.messages.append(message)
        self._current_tokens += message.tokens
        
        # Track system prompt separately (never compacted)
        if role == "system":
            self._system_prompt_tokens += message.tokens
    
    def get_token_count(self) -> int:
        """
        Get current total token count.
        
        Returns:
            Total tokens in context
        """
        return self._current_tokens
    
    def get_available_tokens(self) -> int:
        """
        Get available tokens before hitting limit.
        
        Returns:
            Remaining tokens in context window
        """
        return max(0, self.max_context_length - self._current_tokens)
    
    def get_usage_percentage(self) -> float:
        """
        Get context window usage as percentage.
        
        Returns:
            Usage percentage (0.0 to 100.0)
        """
        if self.max_context_length == 0:
            return 0.0
        return (self._current_tokens / self.max_context_length) * 100.0
    
    def get_warning_level(self) -> ContextWarningLevel:
        """
        Get current warning level based on context usage.
        
        Returns:
            ContextWarningLevel enum value
        """
        usage = self.get_usage_percentage()
        
        if usage < 70.0:
            return ContextWarningLevel.NONE
        elif usage < 80.0:
            return ContextWarningLevel.LOW
        elif usage < 90.0:
            return ContextWarningLevel.MEDIUM
        elif usage < 95.0:
            return ContextWarningLevel.HIGH
        else:
            return ContextWarningLevel.CRITICAL
    
    def needs_compaction(self, threshold: float = 80.0) -> bool:
        """
        Check if context needs compaction.
        
        Args:
            threshold: Usage percentage threshold (default 80%)
            
        Returns:
            True if compaction recommended
        """
        return self.get_usage_percentage() >= threshold
    
    def compact(self, target_percentage: float = 50.0) -> int:
        """
        Compact context by removing oldest non-system messages.
        
        Keeps system prompt and removes oldest messages until
        reaching target percentage.
        
        Args:
            target_percentage: Target usage percentage after compaction
            
        Returns:
            Number of messages removed
        """
        target_tokens = int(self.max_context_length * (target_percentage / 100.0))
        removed_count = 0
        
        # Don't compact if already under target
        if self._current_tokens <= target_tokens:
            return 0
        
        # Remove oldest non-system messages
        while self._current_tokens > target_tokens and len(self.messages) > 0:
            # Find first non-system message
            for i, msg in enumerate(self.messages):
                if msg.role != "system":
                    removed_msg = self.messages.pop(i)
                    self._current_tokens -= removed_msg.tokens
                    removed_count += 1
                    break
            else:
                # No non-system messages left
                break
        
        return removed_count
    
    def get_context_for_prompt(self, max_tokens: Optional[int] = None) -> list[Message]:
        """
        Get messages formatted for prompt, respecting token limits.
        
        Args:
            max_tokens: Maximum tokens to include (uses all if None)
            
        Returns:
            List of messages that fit in token limit
        """
        if max_tokens is None:
            return self.messages.copy()
        
        # Always include system messages
        result = [msg for msg in self.messages if msg.role == "system"]
        current_tokens = sum(msg.tokens for msg in result)
        
        # Add other messages from most recent
        other_messages = [msg for msg in self.messages if msg.role != "system"]
        for msg in reversed(other_messages):
            if current_tokens + msg.tokens <= max_tokens:
                result.insert(len([m for m in result if m.role == "system"]), msg)
                current_tokens += msg.tokens
            else:
                break
        
        return result
    
    def clear(self, keep_system: bool = True) -> None:
        """
        Clear all messages from context.
        
        Args:
            keep_system: If True, keeps system messages
        """
        if keep_system:
            system_messages = [msg for msg in self.messages if msg.role == "system"]
            self.messages = system_messages
            self._current_tokens = self._system_prompt_tokens
        else:
            self.messages = []
            self._current_tokens = 0
            self._system_prompt_tokens = 0
    
    def get_stats(self) -> dict[str, int | float]:
        """
        Get context statistics.

        Returns:
            Dictionary with context stats
        """
        return {
            "total_messages": len(self.messages),
            "system_messages": sum(1 for msg in self.messages if msg.role == "system"),
            "user_messages": sum(1 for msg in self.messages if msg.role == "user"),
            "assistant_messages": sum(1 for msg in self.messages if msg.role == "assistant"),
            "total_tokens": self._current_tokens,
            "system_tokens": self._system_prompt_tokens,
            "max_tokens": self.max_context_length,
            "available_tokens": self.get_available_tokens(),
            "usage_percentage": self.get_usage_percentage(),
            "warning_level": self.get_warning_level().value,
        }


# ==============================================================================
# Phase 6.6: File Context Manager for Agentic Behavior
# ==============================================================================

import os
import re
from pathlib import Path


@dataclass
class FileContext:
    """Represents a file loaded into context for LLM prompts.

    Attributes:
        path: Relative path to the file
        content: File content
        language: Programming language
        line_count: Number of lines
        token_estimate: Estimated token count
    """

    path: str
    content: str
    language: str
    line_count: int
    token_estimate: int = 0


@dataclass
class DirectoryEntry:
    """Represents an entry in a directory listing.

    Attributes:
        name: File or directory name
        is_dir: True if directory
        size: File size in bytes
        line_count: Number of lines (for text files)
        language: Programming language
    """

    name: str
    is_dir: bool
    size: int = 0
    line_count: int = 0
    language: str = ""


class FileContextManager:
    """Manage file and directory context for LLM prompts.

    This class provides functionality for agentic behavior:
    - Load files into context
    - Generate directory listings
    - Build context prompts for the LLM
    - Auto-detect file references in user input
    """

    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "jsx",
        ".tsx": "tsx",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".txt": "text",
        ".sh": "bash",
        ".sql": "sql",
    }

    IGNORE_PATTERNS = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".idea",
        ".vscode",
        ".DS_Store",
        "*.pyc",
        ".env",
    }

    def __init__(
        self,
        working_dir: Optional[Path] = None,
        max_context_tokens: int = 4000,
    ):
        """Initialize file context manager.

        Args:
            working_dir: Working directory (default: cwd)
            max_context_tokens: Maximum tokens for file context
        """
        self.working_dir = Path(working_dir or os.getcwd()).resolve()
        self.max_context_tokens = max_context_tokens
        self.loaded_files: dict[str, FileContext] = {}
        self._total_tokens = 0

    def _detect_language(self, path: Path) -> str:
        """Detect language from file extension."""
        return self.LANGUAGE_MAP.get(path.suffix.lower(), "")

    def _should_ignore(self, name: str) -> bool:
        """Check if file/directory should be ignored."""
        for pattern in self.IGNORE_PATTERNS:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif name == pattern:
                return True
        return False

    def load_file(self, path: str) -> Optional[FileContext]:
        """Load a file into context.

        Args:
            path: Path to file (relative or absolute)

        Returns:
            FileContext if loaded, None otherwise
        """
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.working_dir / file_path
        file_path = file_path.resolve()

        if not file_path.exists() or not file_path.is_file():
            return None

        try:
            rel_path = str(file_path.relative_to(self.working_dir))
        except ValueError:
            rel_path = str(file_path)

        if rel_path in self.loaded_files:
            return self.loaded_files[rel_path]

        try:
            content = file_path.read_text(encoding="utf-8")
            token_estimate = len(content) // 4

            if self._total_tokens + token_estimate > self.max_context_tokens:
                return None

            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

            context = FileContext(
                path=rel_path,
                content=content,
                language=self._detect_language(file_path),
                line_count=line_count,
                token_estimate=token_estimate,
            )

            self.loaded_files[rel_path] = context
            self._total_tokens += token_estimate
            return context

        except (IOError, UnicodeDecodeError):
            return None

    def unload_file(self, path: str) -> bool:
        """Remove file from context."""
        file_path = Path(path)
        if file_path.is_absolute():
            try:
                rel_path = str(file_path.relative_to(self.working_dir))
            except ValueError:
                rel_path = path
        else:
            rel_path = path

        if rel_path in self.loaded_files:
            context = self.loaded_files.pop(rel_path)
            self._total_tokens -= context.token_estimate
            return True
        return False

    def clear_files(self) -> None:
        """Clear all loaded files."""
        self.loaded_files.clear()
        self._total_tokens = 0

    def get_cwd_listing(self, max_depth: int = 1) -> list[DirectoryEntry]:
        """Get directory listing for working directory."""
        entries: list[DirectoryEntry] = []

        def scan_dir(dir_path: Path, depth: int = 0) -> None:
            if depth > max_depth:
                return
            try:
                for item in sorted(dir_path.iterdir()):
                    if self._should_ignore(item.name):
                        continue

                    if item.is_dir():
                        entries.append(DirectoryEntry(
                            name=str(item.relative_to(self.working_dir)) + "/",
                            is_dir=True,
                        ))
                        if depth < max_depth:
                            scan_dir(item, depth + 1)
                    else:
                        try:
                            size = item.stat().st_size
                            language = self._detect_language(item)
                            line_count = 0
                            if language and size < 50000:
                                try:
                                    content = item.read_text(encoding="utf-8")
                                    line_count = content.count("\n") + 1
                                except:
                                    pass

                            entries.append(DirectoryEntry(
                                name=str(item.relative_to(self.working_dir)),
                                is_dir=False,
                                size=size,
                                line_count=line_count,
                                language=language,
                            ))
                        except OSError:
                            pass
            except PermissionError:
                pass

        scan_dir(self.working_dir)
        return entries

    def get_directory_listing_text(self) -> str:
        """Get formatted directory listing."""
        entries = self.get_cwd_listing()
        if not entries:
            return "(empty directory)"

        lines = []
        for entry in entries:
            if entry.is_dir:
                lines.append(f"📁 {entry.name}")
            else:
                info = []
                if entry.line_count:
                    info.append(f"{entry.line_count} lines")
                if entry.language:
                    info.append(entry.language)
                info_str = f" ({', '.join(info)})" if info else ""
                lines.append(f"📄 {entry.name}{info_str}")

        return "\n".join(lines)

    def get_file_context_prompt(self) -> str:
        """Build context section with directory listing and loaded files."""
        parts = []

        # Directory listing
        listing = self.get_directory_listing_text()
        parts.append(f"<directory_listing>\n{listing}\n</directory_listing>")

        # Loaded files
        for rel_path, ctx in self.loaded_files.items():
            tag = f'<file path="{rel_path}" lines="{ctx.line_count}"'
            if ctx.language:
                tag += f' language="{ctx.language}"'
            tag += ">"
            parts.append(f"{tag}\n{ctx.content}\n</file>")

        return "\n\n".join(parts)

    def get_loaded_files_summary(self) -> str:
        """Get summary of loaded files."""
        if not self.loaded_files:
            return "No files loaded."

        lines = ["Loaded files:"]
        for path, ctx in self.loaded_files.items():
            lines.append(f"  • {path} ({ctx.line_count} lines, ~{ctx.token_estimate} tokens)")
        lines.append(f"\nTotal: ~{self._total_tokens}/{self.max_context_tokens} tokens")
        return "\n".join(lines)

    def auto_detect_files(self, user_input: str) -> list[str]:
        """Detect file references in user input."""
        detected = []
        pattern = r'(?:^|[\s\'"(])([./]?[\w\-./]+\.\w+)(?:[\s\'"),]|$)'
        matches = re.findall(pattern, user_input)

        for match in matches:
            file_path = self.working_dir / match
            if file_path.exists() and file_path.is_file():
                detected.append(match)

        return detected

    @property
    def total_tokens(self) -> int:
        """Total tokens in file context."""
        return self._total_tokens

    @property
    def remaining_tokens(self) -> int:
        """Remaining token capacity."""
        return self.max_context_tokens - self._total_tokens
