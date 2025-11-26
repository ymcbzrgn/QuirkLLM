"""
QuirkLLM backend abstraction layer.

This package provides a unified interface for different inference backends:
- MockBackend: Testing backend with hardcoded responses (Phase 1)
- LlamaCppBackend: llama.cpp integration (Phase 2)
- MLXBackend: Apple MLX integration (Phase 2)
"""

from quirkllm.backends.base import (
    Backend,
    BackendType,
    GenerationParams,
    GenerationResult,
    MockBackend,
    create_backend,
)

__all__ = [
    "Backend",
    "BackendType",
    "GenerationParams",
    "GenerationResult",
    "MockBackend",
    "create_backend",
]
