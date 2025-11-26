"""
MLX Backend for QuirkLLM
Phase 2.6 - Apple Silicon optimized inference

MLX framework integration for Metal-accelerated inference on macOS.
Only available on macOS with Apple Silicon (ARM64).
"""
import platform
import sys
from typing import Any, Iterator, Optional
from pathlib import Path

from quirkllm.backends.base import Backend, GenerationParams, GenerationResult


def is_mlx_available() -> bool:
    """
    Check if MLX is available on this system.
    
    MLX requires:
    - macOS operating system
    - ARM64 architecture (Apple Silicon)
    
    Returns:
        True if MLX can be used
    """
    is_macos = platform.system() == "Darwin"
    is_arm = platform.machine() == "arm64"
    return is_macos and is_arm


def check_mlx_import() -> tuple[bool, Optional[str]]:
    """
    Try to import MLX and return status.
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        import mlx.core as mx
        import mlx_lm
        return True, None
    except ImportError as e:
        return False, str(e)


class MLXBackend(Backend):
    """
    MLX backend for Apple Silicon optimized inference.
    
    Features:
    - Metal GPU acceleration
    - Apple Silicon optimized
    - Memory-efficient inference
    - Streaming support
    
    Note: Only works on macOS with Apple Silicon (M1/M2/M3/etc.)
    """
    
    def __init__(self):
        """Initialize MLX backend."""
        # Check platform compatibility
        if not is_mlx_available():
            raise RuntimeError(
                "MLX backend requires macOS with Apple Silicon (ARM64). "
                f"Current system: {platform.system()} {platform.machine()}"
            )
        
        # Check MLX installation
        success, error = check_mlx_import()
        if not success:
            raise RuntimeError(
                f"MLX not installed or import failed: {error}\n"
                "Install with: pip install mlx mlx-lm"
            )
        
        # Import MLX modules (only after checks pass)
        import mlx.core as mx
        import mlx_lm
        
        self.mx = mx
        self.mlx_lm = mlx_lm
        
        # Model state
        self._model = None
        self._tokenizer = None
        self._model_path: Optional[str] = None
        self._model_name: str = "Unknown"
    
    def load_model(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Load an MLX-format model.
        
        Args:
            model_path: Path to MLX model directory (not GGUF)
            adapter_path: Optional LoRA adapter path
            **kwargs: Additional MLX-specific parameters
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            # Load model and tokenizer
            self._model, self._tokenizer = self.mlx_lm.load(
                model_path,
                adapter_path=adapter_path,
            )
            self._model_path = model_path
            self._model_name = Path(model_path).name
        except Exception as e:
            raise RuntimeError(f"MLX model loading failed: {str(e)}") from e
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None and self._tokenizer is not None
    
    def generate(self, params: GenerationParams) -> GenerationResult:
        """
        Generate text from prompt.
        
        Args:
            params: Generation parameters
            
        Returns:
            GenerationResult with generated text and metadata
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize prompt
            prompt_tokens = self._tokenizer.encode(params.prompt)
            prompt_token_count = len(prompt_tokens)
            
            # Generate with MLX
            # Note: mlx-lm API may vary, this is a simplified version
            response = self.mlx_lm.generate(
                model=self._model,
                tokenizer=self._tokenizer,
                prompt=params.prompt,
                max_tokens=params.max_tokens,
                temp=params.temperature,
                top_p=params.top_p,
                verbose=False,
            )
            
            # Extract response text
            if isinstance(response, str):
                generated_text = response
            else:
                # Handle different response formats
                generated_text = str(response)
            
            # Count generated tokens (approximate)
            generated_tokens = self._tokenizer.encode(generated_text)
            generated_token_count = len(generated_tokens)
            
            return GenerationResult(
                text=generated_text,
                tokens_generated=generated_token_count,
                tokens_prompt=prompt_token_count,
                finish_reason="stop",  # MLX may not provide this
                model_name=self._model_name,
            )
        except Exception as e:
            raise RuntimeError(f"MLX generation failed: {str(e)}") from e
    
    def generate_stream(self, params: GenerationParams) -> Iterator[str]:
        """
        Generate text with streaming.
        
        Args:
            params: Generation parameters
            
        Yields:
            Text chunks as they are generated
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # MLX streaming generation
            # Note: API may vary, checking for streaming support
            if hasattr(self.mlx_lm, 'stream_generate'):
                stream = self.mlx_lm.stream_generate(
                    model=self._model,
                    tokenizer=self._tokenizer,
                    prompt=params.prompt,
                    max_tokens=params.max_tokens,
                    temp=params.temperature,
                    top_p=params.top_p,
                )
                
                for chunk in stream:
                    if chunk:
                        yield chunk
            else:
                # Fallback: generate all at once and split
                result = self.generate(params)
                words = result.text.split()
                for i, word in enumerate(words):
                    yield f" {word}" if i > 0 else word
        except Exception as e:
            raise RuntimeError(f"MLX streaming generation failed: {str(e)}") from e
    
    def get_model_info(self) -> dict[str, str | int]:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        if not self.is_loaded():
            return {
                "name": "No model loaded",
                "type": "mlx",
                "size_mb": 0,
                "context_length": 0,
                "quantization": "unknown",
                "architecture": "unknown",
            }
        
        # Estimate size from model path
        size_mb = 0
        if self._model_path and Path(self._model_path).exists():
            # Sum all files in model directory
            model_dir = Path(self._model_path)
            total_bytes = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
            size_mb = total_bytes // (1024 * 1024)
        
        # Try to get context length from config
        context_length = 2048  # Default
        try:
            config_path = Path(self._model_path) / "config.json"
            if config_path.exists():
                import json
                with open(config_path) as f:
                    config = json.load(f)
                    context_length = config.get("max_position_embeddings", 2048)
        except:
            pass
        
        return {
            "name": self._model_name,
            "type": "mlx",
            "size_mb": size_mb,
            "context_length": context_length,
            "quantization": "mlx-optimized",  # MLX uses its own quantization
            "architecture": "mlx-metal",
        }
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            # MLX uses reference counting, just clear references
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._model_path = None
            self._model_name = "Unknown"
