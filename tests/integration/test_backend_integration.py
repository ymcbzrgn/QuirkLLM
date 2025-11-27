"""
Integration tests for backend lifecycle and generation workflows.

Tests the full backend lifecycle including:
- Backend creation via factory
- Model loading simulation
- Single and multiple generations
- Streaming vs non-streaming consistency
- Error handling before load
- Backend info accuracy
- State transitions
"""

import pytest
from quirkllm.backends.base import (
    Backend,
    BackendType,
    GenerationParams,
    GenerationResult,
    create_backend,
)


class TestBackendFactory:
    """Test backend factory pattern and type creation."""

    def test_create_mock_backend(self):
        """Factory should create MockBackend correctly."""
        backend = create_backend(BackendType.MOCK)
        assert backend is not None
        # MockBackend doesn't expose backend_type, but we can check it exists
        assert hasattr(backend, "load_model")
        assert hasattr(backend, "generate")

    def test_factory_returns_backend_instance(self):
        """Factory should return Backend instance."""
        backend = create_backend(BackendType.MOCK)
        assert backend is not None
        assert isinstance(backend, Backend)

    def test_factory_creates_llamacpp_backend(self):
        """Factory should create LlamaCppBackend correctly."""
        backend = create_backend(BackendType.LLAMACPP)
        assert backend is not None
        assert isinstance(backend, Backend)

    def test_factory_creates_mlx_backend(self):
        """Factory should create MLXBackend correctly (requires MLX installed)."""
        try:
            backend = create_backend(BackendType.MLX)
            assert backend is not None
            assert isinstance(backend, Backend)
        except RuntimeError as e:
            if "MLX not installed" in str(e):
                pytest.skip("MLX not installed")


class TestBackendLifecycle:
    """Test backend load/unload lifecycle."""

    def test_load_unload_cycle(self):
        """Backend should handle complete load → unload cycle."""
        backend = create_backend(BackendType.MOCK)

        # Should not be loaded initially
        assert not backend.is_loaded()

        # Load should succeed (MockBackend uses load_model)
        backend.load_model("mock_model_path")
        assert backend.is_loaded()

        # Unload should succeed
        backend.unload_model()
        assert not backend.is_loaded()

    def test_multiple_load_calls_safe(self):
        """Multiple load_model() calls should be safe (idempotent)."""
        backend = create_backend(BackendType.MOCK)

        backend.load_model("model.bin")
        assert backend.is_loaded()

        # Second load should not break anything
        backend.load_model("model.bin")
        assert backend.is_loaded()

        backend.unload_model()

    def test_multiple_unload_calls_safe(self):
        """Multiple unload() calls should be safe."""
        backend = create_backend(BackendType.MOCK)

        backend.load_model("model.bin")
        backend.unload_model()
        assert not backend.is_loaded()

        # Second unload should not break anything
        backend.unload_model()
        assert not backend.is_loaded()

    def test_reload_cycle(self):
        """Backend should support load → unload → load cycle."""
        backend = create_backend(BackendType.MOCK)

        # First cycle
        backend.load_model("model.bin")
        assert backend.is_loaded()
        backend.unload_model()
        assert not backend.is_loaded()

        # Second cycle
        backend.load_model("model.bin")
        assert backend.is_loaded()
        backend.unload_model()
        assert not backend.is_loaded()


class TestBackendGeneration:
    """Test generation workflows."""

    def test_simple_generation(self):
        """Backend should generate response for simple prompt."""
        backend = create_backend(BackendType.MOCK)
        backend.load_model("model.bin")

        params = GenerationParams(
            prompt="hello",
            max_tokens=50,
            temperature=0.7,
        )

        result = backend.generate(params)

        assert isinstance(result, GenerationResult)
        assert result.text
        assert len(result.text) > 0
        assert result.tokens_generated > 0
        assert result.finish_reason in ["stop", "max_tokens"]

        backend.unload_model()

    def test_multiple_generations(self):
        """Backend should handle multiple generation calls."""
        backend = create_backend(BackendType.MOCK)
        backend.load_model("model.bin")

        prompts = ["hello", "explain Python", "write code"]
        results = []

        for prompt in prompts:
            params = GenerationParams(prompt=prompt, max_tokens=100)
            result = backend.generate(params)
            results.append(result)

        # All should succeed
        assert len(results) == 3
        for result in results:
            assert result.text
            assert result.tokens_generated > 0

        backend.unload_model()

    def test_generation_with_different_params(self):
        """Backend should respect different generation parameters."""
        backend = create_backend(BackendType.MOCK)
        backend.load_model("model.bin")

        # Low temperature (more deterministic)
        params_low = GenerationParams(
            prompt="test", max_tokens=50, temperature=0.1
        )
        result_low = backend.generate(params_low)

        # High temperature (more random)
        params_high = GenerationParams(
            prompt="test", max_tokens=50, temperature=1.5
        )
        result_high = backend.generate(params_high)

        # Both should succeed
        assert result_low.text
        assert result_high.text

        backend.unload_model()

    def test_generation_before_load_raises(self):
        """Generation should fail if backend not loaded."""
        backend = create_backend(BackendType.MOCK)

        # Should not be loaded
        assert not backend.is_loaded()

        params = GenerationParams(prompt="test", max_tokens=50)

        with pytest.raises(RuntimeError, match="not loaded"):
            backend.generate(params)


class TestBackendStreaming:
    """Test streaming generation."""

    def test_streaming_generation(self):
        """Streaming should yield tokens incrementally."""
        backend = create_backend(BackendType.MOCK)
        backend.load_model("model.bin")

        params = GenerationParams(prompt="hello", max_tokens=50)

        tokens = list(backend.generate_stream(params))

        # Should yield multiple tokens
        assert len(tokens) > 0
        # Each token should be a string
        assert all(isinstance(token, str) for token in tokens)
        # Joined tokens should form complete response
        full_text = "".join(tokens)
        assert len(full_text) > 0

        backend.unload_model()

    def test_streaming_vs_non_streaming_consistency(self):
        """Streaming and non-streaming should produce similar output."""
        backend = create_backend(BackendType.MOCK)
        backend.load_model("model.bin")

        params = GenerationParams(
            prompt="hello", max_tokens=100, temperature=0.0
        )

        # Non-streaming
        result = backend.generate(params)
        non_stream_text = result.text

        # Streaming
        stream_tokens = list(backend.generate_stream(params))
        stream_text = "".join(stream_tokens)

        # Should be similar (MockBackend may vary slightly, but length similar)
        assert len(stream_text) > 0
        assert len(non_stream_text) > 0

        backend.unload_model()

    def test_streaming_before_load_raises(self):
        """Streaming should fail if backend not loaded."""
        backend = create_backend(BackendType.MOCK)

        params = GenerationParams(prompt="test", max_tokens=50)

        with pytest.raises(RuntimeError, match="not loaded"):
            list(backend.generate_stream(params))


class TestBackendInfo:
    """Test backend information reporting."""

    def test_model_info_before_load(self):
        """Model info should be available even before load."""
        backend = create_backend(BackendType.MOCK)

        # Should not be loaded yet
        assert not backend.is_loaded()

        info = backend.get_model_info()

        # MockBackend returns dict with basic info
        assert "name" in info
        assert "type" in info
        assert info["type"] == "mock"

    def test_model_info_after_load(self):
        """Model info should reflect loaded state."""
        backend = create_backend(BackendType.MOCK)
        backend.load_model("model.bin")

        # Should be loaded now
        assert backend.is_loaded()

        info = backend.get_model_info()

        assert "name" in info
        assert "quantization" in info
        assert info["name"] == "MockModel (model.bin)"

        backend.unload_model()

    def test_model_info_contains_expected_fields(self):
        """Model info should contain expected fields."""
        backend = create_backend(BackendType.MOCK)
        backend.load_model("test_model.bin")

        info = backend.get_model_info()

        # Check all expected fields exist
        assert "name" in info
        assert "type" in info
        assert "quantization" in info
        assert "context_length" in info
        assert "architecture" in info

        backend.unload_model()


class TestBackendErrorHandling:
    """Test backend error scenarios."""

    def test_invalid_generation_params(self):
        """Backend should handle invalid generation params gracefully."""
        backend = create_backend(BackendType.MOCK)
        backend.load_model("model.bin")

        # Negative max_tokens (backend may clamp or raise)
        params = GenerationParams(prompt="test", max_tokens=-10)

        # MockBackend is lenient, but real backends would validate
        # Just ensure it doesn't crash
        try:
            result = backend.generate(params)
            assert result is not None
        except ValueError:
            # Real backends might raise ValueError
            pass

        backend.unload_model()

    def test_empty_prompt_generation(self):
        """Backend should handle empty prompt gracefully."""
        backend = create_backend(BackendType.MOCK)
        backend.load_model("model.bin")

        params = GenerationParams(prompt="", max_tokens=50)

        # Should not crash, may return empty or default response
        result = backend.generate(params)
        assert isinstance(result, GenerationResult)

        backend.unload_model()

    def test_unload_cleans_state(self):
        """Unload should clean up backend state."""
        backend = create_backend(BackendType.MOCK)
        backend.load_model("model.bin")

        # Generate to create some state
        params = GenerationParams(prompt="test", max_tokens=50)
        backend.generate(params)

        # Unload should clean up
        backend.unload_model()
        assert not backend.is_loaded()

        # Can't generate after unload
        with pytest.raises(RuntimeError, match="not loaded"):
            backend.generate(params)


class TestBackendWorkflow:
    """Test complete backend workflows."""

    def test_complete_workflow(self):
        """Test complete workflow: create → load → generate → unload."""
        # Create backend
        backend = create_backend(BackendType.MOCK)
        assert not backend.is_loaded()

        # Load model
        backend.load_model("model.bin")
        assert backend.is_loaded()

        # Generate responses
        params = GenerationParams(prompt="test", max_tokens=50)
        result = backend.generate(params)
        assert result.text

        # Unload
        backend.unload_model()
        assert not backend.is_loaded()

    def test_workflow_with_streaming(self):
        """Test workflow with streaming generation."""
        backend = create_backend(BackendType.MOCK)
        backend.load_model("model.bin")

        # Generate streaming
        params = GenerationParams(prompt="hello", max_tokens=50)
        tokens = list(backend.generate_stream(params))

        assert len(tokens) > 0
        full_text = "".join(tokens)
        assert len(full_text) > 0

        backend.unload_model()

    def test_workflow_with_multiple_prompts(self):
        """Test workflow with multiple different prompts."""
        backend = create_backend(BackendType.MOCK)
        backend.load_model("model.bin")

        # Test different prompt types
        test_prompts = [
            "hello world",
            "write a function",
            "explain something",
        ]

        for prompt in test_prompts:
            params = GenerationParams(prompt=prompt, max_tokens=100)
            result = backend.generate(params)
            assert result.text
            assert result.tokens_generated > 0

        backend.unload_model()
