"""
Unit tests for Conversation Manager (Phase 2.5)
Multi-turn conversations, context integration, profile limits.
"""
import pytest
import json
from pathlib import Path
from datetime import datetime
from quirkllm.core.conversation import ConversationManager, Turn
from quirkllm.core.profile_manager import ProfileConfig
from quirkllm.core.context_manager import ContextWarningLevel


@pytest.fixture
def test_profile():
    """Create a test profile configuration."""
    return ProfileConfig(
        name="TestProfile",
        context_length=1000,
        quantization="Q4_K_M",
        batch_size=4,
        rag_cache_mb=500,
        kv_cache_gb=4,
        embedding_model="base",
        concurrent_ops=2,
        compaction_mode="smart",
        model_loading="hybrid",
        expected_speed_toks=5,
    )


class TestTurn:
    """Test suite for Turn dataclass"""
    
    def test_turn_creation(self):
        """Turn should be created with messages"""
        turn = Turn(
            user_message="Hello",
            assistant_message="Hi there",
        )
        assert turn.user_message == "Hello"
        assert turn.assistant_message == "Hi there"
        assert isinstance(turn.timestamp, datetime)
        assert turn.tokens_used == 0
    
    def test_turn_with_tokens(self):
        """Turn can include token count"""
        turn = Turn(
            user_message="Test",
            assistant_message="Response",
            tokens_used=50,
        )
        assert turn.tokens_used == 50


class TestConversationManager:
    """Test suite for ConversationManager"""
    
    def test_init(self, test_profile):
        """Conversation manager initialization"""
        conv = ConversationManager(profile=test_profile)
        
        assert conv.profile == test_profile
        assert conv.context.max_context_length == 1000
        assert conv.get_turn_count() == 0
        assert len(conv.turns) == 0
    
    def test_init_with_system_prompt(self, test_profile):
        """System prompt should be added to context"""
        conv = ConversationManager(
            profile=test_profile,
            system_prompt="You are a helpful assistant",
        )
        
        assert len(conv.context.messages) == 1
        assert conv.context.messages[0].role == "system"
        assert conv.context.messages[0].content == "You are a helpful assistant"
    
    def test_add_turn(self, test_profile):
        """Adding a turn should update context and turns list"""
        conv = ConversationManager(profile=test_profile)
        
        conv.add_turn("Hello", "Hi there")
        
        assert conv.get_turn_count() == 1
        assert len(conv.context.messages) == 2
        assert conv.context.messages[0].role == "user"
        assert conv.context.messages[1].role == "assistant"
    
    def test_multiple_turns(self, test_profile):
        """Multiple turns should accumulate"""
        conv = ConversationManager(profile=test_profile)
        
        conv.add_turn("First message", "First response")
        conv.add_turn("Second message", "Second response")
        conv.add_turn("Third message", "Third response")
        
        assert conv.get_turn_count() == 3
        assert len(conv.context.messages) == 6  # 3 user + 3 assistant
    
    def test_get_last_turn(self, test_profile):
        """Get last turn should return most recent"""
        conv = ConversationManager(profile=test_profile)
        
        assert conv.get_last_turn() is None
        
        conv.add_turn("First", "Response 1")
        conv.add_turn("Second", "Response 2")
        
        last = conv.get_last_turn()
        assert last is not None
        assert last.user_message == "Second"
        assert last.assistant_message == "Response 2"
    
    def test_get_history_all(self, test_profile):
        """Get all history"""
        conv = ConversationManager(profile=test_profile)
        
        conv.add_turn("Turn 1", "Response 1")
        conv.add_turn("Turn 2", "Response 2")
        conv.add_turn("Turn 3", "Response 3")
        
        history = conv.get_history()
        assert len(history) == 3
        assert history[0].user_message == "Turn 1"
        assert history[2].user_message == "Turn 3"
    
    def test_get_history_last_n(self, test_profile):
        """Get last N turns"""
        conv = ConversationManager(profile=test_profile)
        
        conv.add_turn("Turn 1", "Response 1")
        conv.add_turn("Turn 2", "Response 2")
        conv.add_turn("Turn 3", "Response 3")
        conv.add_turn("Turn 4", "Response 4")
        
        history = conv.get_history(last_n=2)
        assert len(history) == 2
        assert history[0].user_message == "Turn 3"
        assert history[1].user_message == "Turn 4"
    
    def test_format_prompt(self, test_profile):
        """Prompt formatting should include context"""
        conv = ConversationManager(
            profile=test_profile,
            system_prompt="You are helpful",
        )
        
        conv.add_turn("Hello", "Hi there")
        prompt = conv.format_prompt("How are you?")
        
        assert "System: You are helpful" in prompt
        assert "User: Hello" in prompt
        assert "Assistant: Hi there" in prompt
        assert "User: How are you?" in prompt
        assert prompt.endswith("Assistant:")
    
    def test_get_context_info(self, test_profile):
        """Context info should include all stats"""
        conv = ConversationManager(profile=test_profile)
        conv.add_turn("Hello", "Hi")
        
        info = conv.get_context_info()
        
        assert "turn_count" in info
        assert "total_tokens" in info
        assert "usage_percentage" in info
        assert "warning_level" in info
        assert "profile_name" in info
        assert info["profile_name"] == "TestProfile"
        assert info["max_context"] == 1000
    
    def test_should_warn_user_false(self, test_profile):
        """Should not warn when context usage is low"""
        conv = ConversationManager(profile=test_profile)
        conv.add_turn("Short", "Message")
        
        assert conv.should_warn_user() is False
        assert conv.get_warning_message() is None
    
    def test_should_warn_user_high(self, test_profile):
        """Should warn when context usage is high"""
        conv = ConversationManager(profile=test_profile)
        
        # Simulate high usage (manually set)
        conv.context._current_tokens = 920  # 92% usage
        
        assert conv.should_warn_user() is True
        warning = conv.get_warning_message()
        assert warning is not None
        assert "nearly full" in warning.lower()
    
    def test_should_warn_user_critical(self, test_profile):
        """Should warn critically when context almost full"""
        conv = ConversationManager(profile=test_profile)
        
        # Simulate critical usage
        conv.context._current_tokens = 970  # 97% usage
        
        assert conv.should_warn_user() is True
        warning = conv.get_warning_message()
        assert warning is not None
        assert "critically" in warning.lower()
    
    def test_auto_compact_disabled(self, test_profile):
        """Auto-compact can be disabled"""
        conv = ConversationManager(
            profile=test_profile,
            auto_compact=False,
        )
        
        # Fill context significantly
        conv.context._current_tokens = 850  # 85% usage
        conv.add_turn("Test", "Response")
        
        # Should not compact even though over threshold
        assert conv.context._current_tokens > 800
    
    def test_auto_compact_enabled(self, test_profile):
        """Auto-compact should trigger when threshold reached"""
        conv = ConversationManager(
            profile=test_profile,
            auto_compact=True,
            compact_threshold=80.0,
        )
        
        # Add system prompt
        conv.context.add_message("system", "System")
        
        # Add many turns to trigger compaction
        for i in range(10):
            conv.add_turn(f"User message {i}" * 10, f"Assistant response {i}" * 10)
        
        # Context should have been compacted
        # (exact count depends on token estimation, but should be < all messages)
        assert len(conv.context.messages) < 22  # system + 10*2 turns = 21
    
    def test_clear_history_keep_system(self, test_profile):
        """Clear should keep system prompt by default"""
        conv = ConversationManager(
            profile=test_profile,
            system_prompt="System",
        )
        conv.add_turn("User", "Assistant")
        
        conv.clear_history(keep_system=True)
        
        assert conv.get_turn_count() == 0
        assert len(conv.context.messages) == 1
        assert conv.context.messages[0].role == "system"
    
    def test_clear_history_remove_all(self, test_profile):
        """Clear can remove everything including system"""
        conv = ConversationManager(
            profile=test_profile,
            system_prompt="System",
        )
        conv.add_turn("User", "Assistant")
        
        conv.clear_history(keep_system=False)
        
        assert conv.get_turn_count() == 0
        assert len(conv.context.messages) == 0
    
    def test_save_session(self, test_profile, tmp_path):
        """Session can be saved to JSON"""
        conv = ConversationManager(profile=test_profile)
        conv.add_turn("Hello", "Hi")
        conv.add_turn("How are you?", "I'm good")
        
        filepath = conv.save_session(save_dir=str(tmp_path))
        
        assert Path(filepath).exists()
        
        # Verify JSON content
        with open(filepath) as f:
            data = json.load(f)
        
        assert "session_id" in data
        assert "turns" in data
        assert len(data["turns"]) == 2
        assert data["turns"][0]["user"] == "Hello"
        assert data["turns"][1]["assistant"] == "I'm good"
    
    def test_load_session(self, test_profile, tmp_path):
        """Session can be loaded from JSON"""
        # Create and save session
        conv1 = ConversationManager(profile=test_profile)
        conv1.add_turn("First", "Response 1")
        conv1.add_turn("Second", "Response 2")
        filepath = conv1.save_session(save_dir=str(tmp_path))
        
        # Load session
        conv2 = ConversationManager.load_session(filepath, test_profile)
        
        assert conv2.get_turn_count() == 2
        assert conv2.turns[0].user_message == "First"
        assert conv2.turns[1].assistant_message == "Response 2"
        assert conv2.session_id == conv1.session_id
    
    def test_get_summary(self, test_profile):
        """Summary should provide readable overview"""
        conv = ConversationManager(profile=test_profile)
        conv.add_turn("Hello", "Hi")
        
        summary = conv.get_summary()
        
        assert "Session:" in summary
        assert "Profile: TestProfile" in summary
        assert "Turns: 1" in summary
        assert "Tokens:" in summary
        assert "Warning:" in summary
    
    def test_profile_integration(self, test_profile):
        """Conversation should respect profile limits"""
        conv = ConversationManager(profile=test_profile)
        
        # Context length should match profile
        assert conv.context.max_context_length == test_profile.context_length
        
        # Profile info should be accessible
        info = conv.get_context_info()
        assert info["profile_name"] == test_profile.name


class TestConversationEdgeCases:
    """Test edge cases for 100% coverage"""
    
    def test_warning_message_none_level(self, test_profile):
        """Test warning message returns None for NONE level"""
        conv = ConversationManager(profile=test_profile)
        conv.add_turn("Short", "Response")
        
        # Verify warning level is NONE
        assert conv.context.get_warning_level() == ContextWarningLevel.NONE
        
        # Should return None for NONE warning level (line 202)
        message = conv.get_warning_message()
        assert message is None
    
    def test_warning_message_low_and_medium_levels(self, test_profile):
        """Test warning messages for LOW and MEDIUM levels return None"""
        conv = ConversationManager(profile=test_profile)
        
        # Add turns but not enough to trigger HIGH/CRITICAL
        for i in range(5):
            conv.add_turn(f"Message {i}" * 3, f"Response {i}" * 3)
        
        # Should return None for LOW/MEDIUM levels (line 202)
        message = conv.get_warning_message()
        assert message is None
    
    def test_warning_message_high_level(self, test_profile):
        """Test warning message for HIGH level"""
        # Create profile with very small context
        small_profile = ProfileConfig(
            name="SmallProfile",
            context_length=100,  # Very small
            quantization="Q4_K_M",
            batch_size=1,
            rag_cache_mb=100,
            kv_cache_gb=1,
            embedding_model="base",
            concurrent_ops=1,
            compaction_mode="aggressive",
            model_loading="lazy",
            expected_speed_toks=1,
        )
        
        # Disable auto-compact to let context fill up
        conv = ConversationManager(profile=small_profile, auto_compact=False)
        
        # Fill context to HIGH level (90%+)
        for i in range(20):
            conv.add_turn(f"Message {i}" * 5, f"Response {i}" * 5)
        
        message = conv.get_warning_message()
        assert message is not None
        assert "⚠️" in message or "Context" in message
    
    def test_save_session_default_directory(self, test_profile):
        """Test saving session with default directory"""
        conv = ConversationManager(profile=test_profile)
        conv.add_turn("Test", "Response")
        
        # Save with default directory (None)
        filepath = conv.save_session(save_dir=None)
        
        assert Path(filepath).exists()
        assert ".quirkllm/sessions" in filepath
        
        # Cleanup
        Path(filepath).unlink()
    
    def test_auto_compact_triggers_removal(self, test_profile):
        """Test that auto-compact internal method removes messages"""
        # Create profile with small context to force compaction
        small_profile = ProfileConfig(
            name="SmallProfile",
            context_length=200,  # Small enough to trigger compaction
            quantization="Q4_K_M",
            batch_size=1,
            rag_cache_mb=100,
            kv_cache_gb=1,
            embedding_model="base",
            concurrent_ops=1,
            compaction_mode="aggressive",
            model_loading="lazy",
            expected_speed_toks=1,
        )
        
        conv = ConversationManager(
            profile=small_profile,
            auto_compact=True,
            compact_threshold=70.0,
        )
        
        # Add system prompt
        conv.context.add_message("system", "You are a helpful assistant")
        
        # Add many long turns to fill context beyond threshold
        for i in range(12):
            conv.add_turn(
                f"User message number {i}" * 10,  # Long messages
                f"Assistant response number {i}" * 10
            )
        
        # Auto-compact should have triggered during add_turn calls
        # and removed some messages (line 99-106 coverage)
        # The _auto_compact method's if removed > 0 branch should execute
        # We can't easily assert exact counts due to auto-compact, but
        # we can verify the method works by checking context isn't full
        assert conv.context.get_usage_percentage() < 95.0
