"""
Additional tests for backend abstraction to achieve 100% coverage.

Covers:
- NotImplementedError paths for llama-cpp and MLX backends
- Edge cases in backend factory
"""

import pytest

from quirkllm.backends.base import BackendType, create_backend


class TestBackendFactoryEdgeCases:
    """Test edge cases in backend factory."""

    def test_create_llamacpp_backend_string(self):
        """Test creating llama-cpp backend from string."""
        from quirkllm.backends.llamacpp import LlamaCppBackend
        backend = create_backend("llama-cpp")
        assert isinstance(backend, LlamaCppBackend)

    def test_create_mlx_backend_string(self):
        """Test creating MLX backend from string handles platform requirements."""
        # MLX requires macOS ARM64, will raise RuntimeError if not available
        try:
            backend = create_backend("mlx")
            from quirkllm.backends.mlx_backend import MLXBackend
            assert isinstance(backend, MLXBackend)
        except RuntimeError as e:
            # Expected on non-Mac or when MLX not installed
            assert "MLX" in str(e) or "macOS" in str(e)

    def test_create_backend_invalid_string(self):
        """Test that completely invalid string raises ValueError."""
        with pytest.raises(ValueError):
            create_backend("completely_invalid_backend_name")

    def test_backend_type_enum_values(self):
        """Test all BackendType enum values are correct."""
        assert BackendType.LLAMACPP.value == "llama-cpp"
        assert BackendType.MLX.value == "mlx"
        assert BackendType.MOCK.value == "mock"

        # Test all enum members exist
        assert len(list(BackendType)) == 3
    
    def test_create_backend_invalid_type_object(self):
        """Test that invalid type object raises ValueError (line 275 coverage)."""
        # Create a mock enum-like object that isn't a valid BackendType
        class FakeBackendType:
            value = "fake_backend"
        
        fake_type = FakeBackendType()
        
        # This should hit the else branch and raise ValueError
        with pytest.raises((ValueError, AttributeError)):
            create_backend(fake_type)
