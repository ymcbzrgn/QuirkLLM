"""
Conversation Manager for QuirkLLM
Phase 2.5 - Multi-turn conversation management

Manages conversation history, integrates with ContextManager,
handles profile-based limits, and formats prompts for backends.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json
from pathlib import Path

from quirkllm.core.context_manager import ContextManager, Message, ContextWarningLevel
from quirkllm.core.profile_manager import ProfileConfig


@dataclass
class Turn:
    """Represents a conversation turn.
    
    Attributes:
        user_message: User's input message
        assistant_message: Assistant's response
        timestamp: When the turn occurred
        tokens_used: Total tokens used in this turn
    """
    user_message: str
    assistant_message: str
    timestamp: datetime = field(default_factory=datetime.now)
    tokens_used: int = 0


class ConversationManager:
    """
    Manages multi-turn conversations with context tracking.
    
    Features:
    - Turn-based conversation tracking
    - Context manager integration
    - Profile-based limits enforcement
    - Automatic context compaction
    - Session persistence (save/load)
    """
    
    def __init__(
        self,
        profile: ProfileConfig,
        system_prompt: Optional[str] = None,
        auto_compact: bool = True,
        compact_threshold: float = 80.0,
    ):
        """
        Initialize conversation manager.
        
        Args:
            profile: Active profile configuration
            system_prompt: System prompt for the conversation
            auto_compact: Automatically compact when threshold reached
            compact_threshold: Usage percentage to trigger compaction
        """
        self.profile = profile
        self.auto_compact = auto_compact
        self.compact_threshold = compact_threshold
        
        # Initialize context manager with profile's context length
        self.context = ContextManager(max_context_length=profile.context_length)
        
        # Add system prompt if provided
        if system_prompt:
            self.context.add_message("system", system_prompt)
        
        # Track turns
        self.turns: list[Turn] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """
        Add a conversation turn.
        
        Args:
            user_message: User's input
            assistant_message: Assistant's response
        """
        # Add messages to context
        self.context.add_message("user", user_message)
        self.context.add_message("assistant", assistant_message)
        
        # Create turn record
        turn = Turn(
            user_message=user_message,
            assistant_message=assistant_message,
            tokens_used=self.context.get_token_count(),
        )
        self.turns.append(turn)
        
        # Auto-compact if needed
        if self.auto_compact and self.context.needs_compaction(self.compact_threshold):
            self._auto_compact()
    
    def _auto_compact(self) -> None:
        """Automatically compact context when threshold reached."""
        removed = self.context.compact(target_percentage=50.0)
        if removed > 0:
            # Could log this or notify user
            pass
    
    def get_turn_count(self) -> int:
        """Get total number of turns in conversation."""
        return len(self.turns)
    
    def get_last_turn(self) -> Optional[Turn]:
        """Get the most recent turn."""
        return self.turns[-1] if self.turns else None
    
    def get_history(self, last_n: Optional[int] = None) -> list[Turn]:
        """
        Get conversation history.
        
        Args:
            last_n: Get only last N turns (all if None)
            
        Returns:
            List of turns
        """
        if last_n is None:
            return self.turns.copy()
        return self.turns[-last_n:]
    
    def format_prompt(self, new_user_message: str) -> str:
        """
        Format a prompt for the backend including conversation context.
        
        Args:
            new_user_message: New user message to add
            
        Returns:
            Formatted prompt string with context
        """
        # Get messages from context manager
        messages = self.context.get_context_for_prompt()
        
        # Format as simple text (can be extended for chat templates)
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        
        # Add new user message
        parts.append(f"User: {new_user_message}")
        parts.append("Assistant:")
        
        return "\n\n".join(parts)
    
    def get_context_info(self) -> dict:
        """
        Get current context information.
        
        Returns:
            Dictionary with context stats and warnings
        """
        stats = self.context.get_stats()
        warning_level = self.context.get_warning_level()
        
        return {
            **stats,
            "turn_count": len(self.turns),
            "warning_level": warning_level.value,
            "needs_compaction": self.context.needs_compaction(self.compact_threshold),
            "profile_name": self.profile.name,
            "max_context": self.profile.context_length,
        }
    
    def should_warn_user(self) -> bool:
        """Check if user should be warned about context usage."""
        warning = self.context.get_warning_level()
        return warning in [ContextWarningLevel.HIGH, ContextWarningLevel.CRITICAL]
    
    def get_warning_message(self) -> Optional[str]:
        """Get warning message for user if needed."""
        if not self.should_warn_user():
            return None
        
        warning = self.context.get_warning_level()
        usage = self.context.get_usage_percentage()
        
        if warning == ContextWarningLevel.CRITICAL:
            return (
                f"⚠️  Context window critically full ({usage:.1f}%)! "
                "Oldest messages will be removed automatically."
            )
        elif warning == ContextWarningLevel.HIGH:
            return (
                f"⚠️  Context window nearly full ({usage:.1f}%). "
                "Consider starting a new conversation soon."
            )
        
        return None
    
    def clear_history(self, keep_system: bool = True) -> None:
        """
        Clear conversation history.
        
        Args:
            keep_system: Keep system prompt
        """
        self.turns = []
        self.context.clear(keep_system=keep_system)
    
    def save_session(self, save_dir: Optional[str] = None) -> str:
        """
        Save conversation session to JSON file.
        
        Args:
            save_dir: Directory to save session (default: ~/.quirkllm/sessions/)
            
        Returns:
            Path to saved session file
        """
        if save_dir is None:
            save_dir = str(Path.home() / ".quirkllm" / "sessions")
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Prepare session data
        session_data = {
            "session_id": self.session_id,
            "profile_name": self.profile.name,
            "created_at": self.turns[0].timestamp.isoformat() if self.turns else datetime.now().isoformat(),
            "turns": [
                {
                    "user": turn.user_message,
                    "assistant": turn.assistant_message,
                    "timestamp": turn.timestamp.isoformat(),
                    "tokens": turn.tokens_used,
                }
                for turn in self.turns
            ],
            "context_stats": self.context.get_stats(),
        }
        
        # Save to file
        filepath = Path(save_dir) / f"session_{self.session_id}.json"
        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)
        
        return str(filepath)
    
    @classmethod
    def load_session(
        cls,
        filepath: str,
        profile: ProfileConfig,
    ) -> "ConversationManager":
        """
        Load conversation session from JSON file.
        
        Args:
            filepath: Path to session file
            profile: Profile configuration to use
            
        Returns:
            ConversationManager instance with loaded history
        """
        with open(filepath, "r") as f:
            session_data = json.load(f)
        
        # Create new conversation manager
        manager = cls(profile=profile)
        manager.session_id = session_data["session_id"]
        
        # Restore turns
        for turn_data in session_data["turns"]:
            # Add messages to context
            manager.context.add_message("user", turn_data["user"])
            manager.context.add_message("assistant", turn_data["assistant"])
            
            # Create turn record
            turn = Turn(
                user_message=turn_data["user"],
                assistant_message=turn_data["assistant"],
                timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                tokens_used=turn_data["tokens"],
            )
            manager.turns.append(turn)
        
        return manager
    
    def get_summary(self) -> str:
        """
        Get a summary of the conversation.
        
        Returns:
            Human-readable summary string
        """
        info = self.get_context_info()
        
        summary_parts = [
            f"Session: {self.session_id}",
            f"Profile: {self.profile.name}",
            f"Turns: {info['turn_count']}",
            f"Tokens: {info['total_tokens']:,}/{info['max_context']:,} ({info['usage_percentage']:.1f}%)",
            f"Warning: {info['warning_level']}",
        ]
        
        return " | ".join(summary_parts)
