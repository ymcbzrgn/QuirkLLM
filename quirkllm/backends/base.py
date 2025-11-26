"""
Backend abstraction layer for QuirkLLM.

Provides a unified interface for different inference backends (llama-cpp, MLX, etc.)
and a mock backend for testing Phase 1 functionality without requiring models.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any


class BackendType(Enum):
    """Available backend types."""

    LLAMACPP = "llama-cpp"
    MLX = "mlx"
    MOCK = "mock"


@dataclass
class GenerationParams:
    """Parameters for text generation.

    Attributes:
        prompt: Input prompt text
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        stop_sequences: List of sequences that stop generation
        stream: Whether to stream the response token-by-token
    """

    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    stop_sequences: list[str] | None = None
    stream: bool = False


@dataclass
class GenerationResult:
    """Result from text generation.

    Attributes:
        text: Generated text
        tokens_generated: Number of tokens generated
        tokens_prompt: Number of tokens in the prompt
        finish_reason: Reason generation stopped (max_tokens, stop_sequence, etc.)
        model_name: Name of the model used
    """

    text: str
    tokens_generated: int
    tokens_prompt: int
    finish_reason: str
    model_name: str


class Backend(ABC):
    """Abstract base class for inference backends.

    All backends must implement this interface to work with QuirkLLM.
    """

    @abstractmethod
    def load_model(self, model_path: str, **kwargs: Any) -> None:
        """Load a model from the specified path.

        Args:
            model_path: Path to the model file
            **kwargs: Backend-specific loading parameters
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded.

        Returns:
            True if model is loaded and ready for inference
        """
        ...

    @abstractmethod
    def generate(self, params: GenerationParams) -> GenerationResult:
        """Generate text from a prompt.

        Args:
            params: Generation parameters

        Returns:
            GenerationResult containing generated text and metadata
        """
        ...

    @abstractmethod
    def generate_stream(self, params: GenerationParams) -> Iterator[str]:
        """Generate text from a prompt with streaming.

        Args:
            params: Generation parameters

        Yields:
            Generated text chunks as they become available
        """
        ...

    @abstractmethod
    def get_model_info(self) -> dict[str, str | int]:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model metadata (name, size, context_length, etc.)
        """
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the currently loaded model to free memory."""
        ...


class MockBackend(Backend):
    """Mock backend for testing without actual models.

    This backend provides hardcoded responses for Phase 1 testing,
    allowing CLI and infrastructure development without model dependencies.
    """

    def __init__(self) -> None:
        """Initialize mock backend."""
        self._loaded = False
        self._model_name = "MockModel-1.3B"

    def load_model(self, model_path: str, **kwargs: Any) -> None:
        """Simulate model loading.

        Args:
            model_path: Path to model (ignored in mock)
            **kwargs: Additional parameters (ignored in mock)
        """
        self._loaded = True
        self._model_name = f"MockModel ({model_path})"

    def is_loaded(self) -> bool:
        """Check if mock model is loaded.

        Returns:
            True if load_model has been called
        """
        return self._loaded

    def generate(self, params: GenerationParams) -> GenerationResult:
        """Generate mock response.

        Args:
            params: Generation parameters

        Returns:
            Mock generation result with hardcoded response
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Simulate token counting
        prompt_tokens = len(params.prompt.split())

        # Generate mock response based on prompt
        if "hello" in params.prompt.lower():
            response = "Hello! I'm QuirkLLM, your local coding assistant. How can I help you today?"
        elif "code" in params.prompt.lower() or "function" in params.prompt.lower():
            response = (
                "Sure! Here's a Python function:\n\n"
                "```python\n"
                "def example_function(param: str) -> str:\n"
                '    """Example function."""\n'
                '    return f"Processed: {param}"\n'
                "```"
            )
        elif "explain" in params.prompt.lower():
            response = (
                "I'd be happy to explain that! This is a mock response from the MockBackend. "
                "In Phase 2, this will be replaced with actual model inference."
            )
        else:
            response = (
                "This is a mock response from QuirkLLM's test backend. "
                "In Phase 2, this will be replaced with real AI-generated content from "
                f"the loaded model. Your prompt was: {params.prompt[:50]}..."
            )

        response_tokens = len(response.split())

        return GenerationResult(
            text=response,
            tokens_generated=response_tokens,
            tokens_prompt=prompt_tokens,
            finish_reason="max_tokens" if response_tokens >= params.max_tokens else "stop",
            model_name=self._model_name,
        )

    def generate_stream(self, params: GenerationParams) -> Iterator[str]:
        """Generate mock streaming response.

        Args:
            params: Generation parameters

        Yields:
            Mock text chunks simulating streaming
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Generate the full response
        result = self.generate(params)

        # Split into words and yield one at a time to simulate streaming
        words = result.text.split()
        for i, word in enumerate(words):
            # Add space before all words except the first
            yield f" {word}" if i > 0 else word

    def get_model_info(self) -> dict[str, str | int]:
        """Get mock model information.

        Returns:
            Dictionary with mock model metadata
        """
        return {
            "name": self._model_name,
            "type": "mock",
            "size_mb": 0,
            "context_length": 32768,
            "quantization": "none",
            "architecture": "mock",
        }

    def unload_model(self) -> None:
        """Simulate model unloading."""
        self._loaded = False
        self._model_name = "MockModel-1.3B"


def create_backend(backend_type: BackendType | str) -> Backend:
    """Factory function to create backend instances.

    Args:
        backend_type: Type of backend to create (BackendType enum or string)

    Returns:
        Backend instance of the requested type

    Raises:
        ValueError: If backend_type is not recognized
    """
    # Convert string to enum if necessary
    if isinstance(backend_type, str):
        backend_type = BackendType(backend_type)

    if backend_type == BackendType.MOCK:
        return MockBackend()
    elif backend_type == BackendType.LLAMACPP:
        from quirkllm.backends.llamacpp import LlamaCppBackend
        return LlamaCppBackend()
    elif backend_type == BackendType.MLX:
        from quirkllm.backends.mlx_backend import MLXBackend
        return MLXBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
