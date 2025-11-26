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
