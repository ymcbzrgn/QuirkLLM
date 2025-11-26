"""
Tests for backend abstraction layer (quirkllm/backends/base.py).
"""

import pytest

from quirkllm.backends.base import (
    Backend,
    BackendType,
    GenerationParams,
    GenerationResult,
    MockBackend,
    create_backend,
)


class TestBackendType:
    """Test BackendType enum."""

    def test_backend_types(self):
        """Test that all backend types are defined."""
        assert BackendType.LLAMACPP.value == "llama-cpp"
        assert BackendType.MLX.value == "mlx"
        assert BackendType.MOCK.value == "mock"


class TestGenerationParams:
    """Test GenerationParams dataclass."""

    def test_default_params(self):
        """Test default generation parameters."""
        params = GenerationParams(prompt="Hello")

        assert params.prompt == "Hello"
        assert params.max_tokens == 512
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 40
        assert params.stop_sequences is None
        assert params.stream is False

    def test_custom_params(self):
        """Test custom generation parameters."""
        params = GenerationParams(
            prompt="Test prompt",
            max_tokens=1024,
            temperature=0.5,
            top_p=0.95,
            top_k=50,
            stop_sequences=["END", "STOP"],
            stream=True,
        )

        assert params.prompt == "Test prompt"
        assert params.max_tokens == 1024
        assert params.temperature == 0.5
        assert params.top_p == 0.95
        assert params.top_k == 50
        assert params.stop_sequences == ["END", "STOP"]
        assert params.stream is True


class TestGenerationResult:
    """Test GenerationResult dataclass."""

    def test_generation_result(self):
        """Test generation result creation."""
        result = GenerationResult(
            text="Generated text",
            tokens_generated=10,
            tokens_prompt=5,
            finish_reason="stop",
            model_name="TestModel",
        )

        assert result.text == "Generated text"
        assert result.tokens_generated == 10
        assert result.tokens_prompt == 5
        assert result.finish_reason == "stop"
        assert result.model_name == "TestModel"


class TestMockBackend:
    """Test MockBackend implementation."""

    def test_initialization(self):
        """Test that mock backend initializes correctly."""
        backend = MockBackend()

        assert not backend.is_loaded()
        assert isinstance(backend, Backend)

    def test_load_model(self):
        """Test loading a mock model."""
        backend = MockBackend()
        backend.load_model("/fake/path/model.gguf")

        assert backend.is_loaded()

    def test_generate_before_load_raises_error(self):
        """Test that generating before loading raises error."""
        backend = MockBackend()
        params = GenerationParams(prompt="Hello")

        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.generate(params)

    def test_generate_after_load(self):
        """Test basic text generation."""
        backend = MockBackend()
        backend.load_model("/fake/path/model.gguf")

        params = GenerationParams(prompt="Hello world")
        result = backend.generate(params)

        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0
        assert result.tokens_generated > 0
        assert result.tokens_prompt > 0
        assert result.finish_reason in ["stop", "max_tokens"]
        assert "MockModel" in result.model_name

    def test_generate_greeting_response(self):
        """Test that hello prompts get greeting responses."""
        backend = MockBackend()
        backend.load_model("/fake/path/model.gguf")

        params = GenerationParams(prompt="Hello!")
        result = backend.generate(params)

        assert "Hello" in result.text or "hello" in result.text
        assert "QuirkLLM" in result.text

    def test_generate_code_response(self):
        """Test that code prompts get code responses."""
        backend = MockBackend()
        backend.load_model("/fake/path/model.gguf")

        params = GenerationParams(prompt="Write a Python function")
        result = backend.generate(params)

        assert "```python" in result.text or "def " in result.text

    def test_generate_explain_response(self):
        """Test that explain prompts get explanation responses."""
        backend = MockBackend()
        backend.load_model("/fake/path/model.gguf")

        params = GenerationParams(prompt="Explain this concept")
        result = backend.generate(params)

        assert "explain" in result.text.lower() or "mock" in result.text.lower()

    def test_generate_default_response(self):
        """Test default response for unknown prompts."""
        backend = MockBackend()
        backend.load_model("/fake/path/model.gguf")

        params = GenerationParams(prompt="Random unknown prompt")
        result = backend.generate(params)

        assert "mock" in result.text.lower()
        assert len(result.text) > 0

    def test_generate_with_custom_params(self):
        """Test generation with custom parameters."""
        backend = MockBackend()
        backend.load_model("/fake/path/model.gguf")

        params = GenerationParams(
            prompt="Hello",
            max_tokens=100,
            temperature=0.5,
            top_p=0.95,
            stop_sequences=["END"],
        )
        result = backend.generate(params)

        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0

    def test_generate_stream_before_load_raises_error(self):
        """Test that streaming before loading raises error."""
        backend = MockBackend()
        params = GenerationParams(prompt="Hello")

        with pytest.raises(RuntimeError, match="Model not loaded"):
            list(backend.generate_stream(params))

    def test_generate_stream(self):
        """Test streaming text generation."""
        backend = MockBackend()
        backend.load_model("/fake/path/model.gguf")

        params = GenerationParams(prompt="Hello world")
        chunks = list(backend.generate_stream(params))

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0

    def test_generate_stream_matches_generate(self):
        """Test that streaming produces same text as non-streaming."""
        backend = MockBackend()
        backend.load_model("/fake/path/model.gguf")

        params = GenerationParams(prompt="Hello world")

        # Generate normally
        result = backend.generate(params)

        # Generate with streaming
        chunks = list(backend.generate_stream(params))
        streamed_text = "".join(chunks)

        assert result.text == streamed_text

    def test_get_model_info_after_load(self):
        """Test getting model information."""
        backend = MockBackend()
        backend.load_model("/fake/path/model.gguf")

        info = backend.get_model_info()

        assert "name" in info
        assert "type" in info
        assert "size_mb" in info
        assert "context_length" in info
        assert "quantization" in info
        assert "architecture" in info
        assert info["type"] == "mock"
        assert isinstance(info["context_length"], int)

    def test_unload_model(self):
        """Test unloading a model."""
        backend = MockBackend()
        backend.load_model("/fake/path/model.gguf")

        assert backend.is_loaded()

        backend.unload_model()

        assert not backend.is_loaded()

    def test_reload_model(self):
        """Test loading, unloading, and reloading."""
        backend = MockBackend()

        # Load
        backend.load_model("/path1/model.gguf")
        assert backend.is_loaded()

        # Unload
        backend.unload_model()
        assert not backend.is_loaded()

        # Reload
        backend.load_model("/path2/model.gguf")
        assert backend.is_loaded()


class TestCreateBackend:
    """Test backend factory function."""

    def test_create_mock_backend_from_enum(self):
        """Test creating mock backend from enum."""
        backend = create_backend(BackendType.MOCK)

        assert isinstance(backend, MockBackend)
        assert isinstance(backend, Backend)

    def test_create_mock_backend_from_string(self):
        """Test creating mock backend from string."""
        backend = create_backend("mock")

        assert isinstance(backend, MockBackend)

    def test_create_llamacpp_backend_implemented(self):
        """Test that llama-cpp backend can be created."""
        from quirkllm.backends.llamacpp import LlamaCppBackend
        backend = create_backend(BackendType.LLAMACPP)
        assert isinstance(backend, LlamaCppBackend)

    def test_create_mlx_backend_requires_platform(self):
        """Test that MLX backend raises RuntimeError on unsupported platforms."""
        # MLX requires macOS ARM64, will raise RuntimeError if not available
        # or if MLX not installed
        try:
            backend = create_backend(BackendType.MLX)
            # If successful, we're on macOS ARM with MLX installed
            from quirkllm.backends.mlx_backend import MLXBackend
            assert isinstance(backend, MLXBackend)
        except RuntimeError as e:
            # Expected on non-Mac or when MLX not installed
            assert "MLX" in str(e) or "macOS" in str(e)

    def test_create_invalid_backend_raises_error(self):
        """Test that invalid backend type raises ValueError."""
        with pytest.raises(ValueError, match="is not a valid BackendType"):
            # This will fail at the BackendType(invalid_string) step
            create_backend("invalid_backend")
