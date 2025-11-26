"""
Unit tests for Context Manager (Phase 2.4)
Token counting, context tracking, compaction, warnings.
"""
import pytest
from quirkllm.core.context_manager import (
    ContextManager,
    Message,
    ContextWarningLevel,
    estimate_tokens,
)


class TestEstimateTokens:
    """Test suite for token estimation"""
    
    def test_empty_string(self):
        """Empty string should return 0 tokens"""
        assert estimate_tokens("") == 0
    
    def test_simple_text(self):
        """Natural language: ~1.33 tokens per word"""
        text = "Hello world"  # 2 words
        tokens = estimate_tokens(text)
        assert tokens == 2  # 2 * 1.33 = 2.66 -> 2
    
    def test_longer_text(self):
        """Longer natural language text"""
        text = "This is a longer sentence with multiple words"  # 8 words
        tokens = estimate_tokens(text)
        assert tokens == 10  # 8 * 1.33 = 10.64 -> 10
    
    def test_code_detection(self):
        """Code should use character-based estimation"""
        code = "def hello():\n    return 'Hello'"  # 31 chars
        tokens = estimate_tokens(code)
        assert tokens == 7  # 31 / 4 = 7.75 -> 7
    
    def test_code_with_braces(self):
        """JavaScript/C-style code detection"""
        code = "function test() { return 42; }"  # 30 chars
        tokens = estimate_tokens(code)
        assert tokens == 7  # 30 / 4 = 7.5 -> 7
    
    def test_markdown_code_block(self):
        """Code blocks with ``` should be detected as code"""
        code = "```python\nprint('hello')\n```"  # 29 chars
        tokens = estimate_tokens(code)
        assert tokens == 7  # 29 / 4 = 7.25 -> 7


class TestContextManagerEdgeCases:
    """Test edge cases for 100% coverage"""
    
    def test_zero_max_context_length(self):
        """Test usage percentage when max_context_length is 0 (edge case)"""
        manager = ContextManager(max_context_length=0)
        assert manager.get_usage_percentage() == 0.0
        assert manager.get_token_count() == 0


class TestMessage:
    """Test suite for Message dataclass"""
    
    def test_message_creation(self):
        """Message should auto-calculate tokens"""
        msg = Message(role="user", content="Hello world")
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert msg.tokens > 0
    
    def test_message_with_explicit_tokens(self):
        """Explicit token count should be respected"""
        msg = Message(role="assistant", content="Test", tokens=100)
        assert msg.tokens == 100
    
    def test_message_empty_content(self):
        """Empty content should still work"""
        msg = Message(role="system", content="")
        assert msg.tokens == 0


class TestContextManager:
    """Test suite for ContextManager"""
    
    def test_init(self):
        """Context manager initialization"""
        ctx = ContextManager(max_context_length=32768)
        assert ctx.max_context_length == 32768
        assert ctx.get_token_count() == 0
        assert len(ctx.messages) == 0
    
    def test_add_message(self):
        """Adding messages should update token count"""
        ctx = ContextManager(max_context_length=1000)
        ctx.add_message("user", "Hello")
        
        assert len(ctx.messages) == 1
        assert ctx.get_token_count() > 0
        assert ctx.messages[0].role == "user"
        assert ctx.messages[0].content == "Hello"
    
    def test_add_multiple_messages(self):
        """Multiple messages should accumulate tokens"""
        ctx = ContextManager(max_context_length=1000)
        ctx.add_message("system", "You are a helpful assistant")
        ctx.add_message("user", "Hello")
        ctx.add_message("assistant", "Hi there")
        
        assert len(ctx.messages) == 3
        assert ctx.get_token_count() > 0
    
    def test_system_prompt_tracking(self):
        """System prompts should be tracked separately"""
        ctx = ContextManager(max_context_length=1000)
        ctx.add_message("system", "You are helpful")
        system_tokens = ctx._system_prompt_tokens
        
        ctx.add_message("user", "Hello")
        # System tokens shouldn't change
        assert ctx._system_prompt_tokens == system_tokens
    
    def test_get_available_tokens(self):
        """Available tokens calculation"""
        ctx = ContextManager(max_context_length=100)
        ctx.add_message("user", "Hello world")  # ~2 tokens
        
        available = ctx.get_available_tokens()
        assert available > 90
        assert available < 100
    
    def test_get_usage_percentage(self):
        """Usage percentage calculation"""
        ctx = ContextManager(max_context_length=100)
        ctx.add_message("user", "Hello world")  # ~2 tokens
        
        usage = ctx.get_usage_percentage()
        assert 0.0 < usage < 10.0
    
    def test_warning_level_none(self):
        """Warning level NONE when < 70% full"""
        ctx = ContextManager(max_context_length=100)
        ctx.add_message("user", "Short")  # < 70 tokens
        
        assert ctx.get_warning_level() == ContextWarningLevel.NONE
    
    def test_warning_level_progression(self):
        """Warning levels should progress with usage"""
        ctx = ContextManager(max_context_length=100)
        
        # Manually set tokens to test thresholds
        ctx._current_tokens = 75  # 75% -> LOW
        assert ctx.get_warning_level() == ContextWarningLevel.LOW
        
        ctx._current_tokens = 85  # 85% -> MEDIUM
        assert ctx.get_warning_level() == ContextWarningLevel.MEDIUM
        
        ctx._current_tokens = 92  # 92% -> HIGH
        assert ctx.get_warning_level() == ContextWarningLevel.HIGH
        
        ctx._current_tokens = 97  # 97% -> CRITICAL
        assert ctx.get_warning_level() == ContextWarningLevel.CRITICAL
    
    def test_needs_compaction_false(self):
        """Should not need compaction when below threshold"""
        ctx = ContextManager(max_context_length=100)
        ctx.add_message("user", "Short message")
        
        assert ctx.needs_compaction(threshold=80.0) is False
    
    def test_needs_compaction_true(self):
        """Should need compaction when above threshold"""
        ctx = ContextManager(max_context_length=100)
        ctx._current_tokens = 85  # 85% usage
        ctx.add_message("user", "Message")
        
        assert ctx.needs_compaction(threshold=80.0) is True
    
    def test_compact_removes_oldest(self):
        """Compaction should remove oldest non-system messages"""
        ctx = ContextManager(max_context_length=100)
        ctx.add_message("system", "System prompt")
        ctx.add_message("user", "First user message")
        ctx.add_message("assistant", "First assistant message")
        ctx.add_message("user", "Second user message")
        
        initial_count = len(ctx.messages)
        ctx._current_tokens = 90  # Simulate 90% usage
        
        removed = ctx.compact(target_percentage=50.0)
        
        assert removed > 0
        assert len(ctx.messages) < initial_count
        # System message should still be there
        assert any(msg.role == "system" for msg in ctx.messages)
    
    def test_compact_keeps_system(self):
        """Compaction should never remove system messages"""
        ctx = ContextManager(max_context_length=100)
        ctx.add_message("system", "System prompt")
        ctx.add_message("user", "User message")
        ctx._current_tokens = 90
        
        ctx.compact(target_percentage=10.0)
        
        # System message must remain
        assert len(ctx.messages) >= 1
        assert ctx.messages[0].role == "system"
    
    def test_compact_no_op_under_target(self):
        """Compaction should do nothing if already under target"""
        ctx = ContextManager(max_context_length=100)
        ctx.add_message("user", "Short")
        ctx._current_tokens = 20  # 20% usage
        
        removed = ctx.compact(target_percentage=50.0)
        assert removed == 0
    
    def test_get_context_for_prompt_all(self):
        """Get all messages when no limit specified"""
        ctx = ContextManager(max_context_length=1000)
        ctx.add_message("system", "System")
        ctx.add_message("user", "User")
        ctx.add_message("assistant", "Assistant")
        
        messages = ctx.get_context_for_prompt()
        assert len(messages) == 3
    
    def test_get_context_for_prompt_limited(self):
        """Get messages respecting token limit"""
        ctx = ContextManager(max_context_length=1000)
        ctx.add_message("system", "System prompt")  # ~2 tokens
        ctx.add_message("user", "First user")       # ~2 tokens
        ctx.add_message("assistant", "Response")    # ~1 token
        ctx.add_message("user", "Second user")      # ~2 tokens
        
        # Request only 5 tokens worth
        messages = ctx.get_context_for_prompt(max_tokens=5)
        
        # Should prioritize system + recent messages
        assert len(messages) >= 1  # At least system
        assert messages[0].role == "system"
    
    def test_clear_keep_system(self):
        """Clear should keep system messages by default"""
        ctx = ContextManager(max_context_length=1000)
        ctx.add_message("system", "System")
        ctx.add_message("user", "User")
        ctx.add_message("assistant", "Assistant")
        
        ctx.clear(keep_system=True)
        
        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == "system"
    
    def test_clear_all(self):
        """Clear all should remove everything"""
        ctx = ContextManager(max_context_length=1000)
        ctx.add_message("system", "System")
        ctx.add_message("user", "User")
        
        ctx.clear(keep_system=False)
        
        assert len(ctx.messages) == 0
        assert ctx.get_token_count() == 0
    
    def test_get_stats(self):
        """Stats should include all metrics"""
        ctx = ContextManager(max_context_length=1000)
        ctx.add_message("system", "System")
        ctx.add_message("user", "User 1")
        ctx.add_message("assistant", "Assistant 1")
        ctx.add_message("user", "User 2")
        
        stats = ctx.get_stats()
        
        assert stats["total_messages"] == 4
        assert stats["system_messages"] == 1
        assert stats["user_messages"] == 2
        assert stats["assistant_messages"] == 1
        assert stats["total_tokens"] > 0
        assert stats["max_tokens"] == 1000
        assert "usage_percentage" in stats
        assert "warning_level" in stats
