"""
Llama.cpp Backend for QuirkLLM
Phase 2 - Day 17-19

llama-cpp-python entegrasyonu, profile-based quantization, GPU offload.
"""
from typing import Any, Iterator
from pathlib import Path
from llama_cpp import Llama
from quirkllm.backends.base import Backend, GenerationParams, GenerationResult


class LlamaCppBackend(Backend):
    """
    llama-cpp-python backend for local inference.
    
    Features:
    - GGUF model support
    - Profile-based quantization (Q4_K_M, Q8_0)
    - GPU offload (if available)
    - KV-cache management
    - Streaming support
    """
    
    def __init__(self):
        """Initialize llama-cpp backend."""
        self._model: Llama | None = None
        self._model_path: str | None = None
        self._model_name: str = "Unknown"
    
    def load_model(
        self, 
        model_path: str,
        n_ctx: int = 32768,
        n_batch: int = 512,
        n_gpu_layers: int = 0,
        use_mmap: bool = True,
        use_mlock: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Load a GGUF model with llama-cpp.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size (profile-based: 16K/32K/64K/128K)
            n_batch: Batch size for prompt processing (profile-based: 1/4/8/16)
            n_gpu_layers: GPU offload layers (0 = CPU only, -1 = all layers)
            use_mmap: Memory-map model file (recommended for large models)
            use_mlock: Lock model in RAM (prevents swapping)
            **kwargs: Additional llama-cpp parameters
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            self._model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_gpu_layers=n_gpu_layers,
                use_mmap=use_mmap,
                use_mlock=use_mlock,
                verbose=False,
                **kwargs
            )
            self._model_path = model_path
            self._model_name = Path(model_path).stem
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}") from e
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
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
        
        # Prepare stop sequences
        stop = params.stop_sequences or []
        
        try:
            # Call llama-cpp generate
            result = self._model.create_completion(
                prompt=params.prompt,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                stop=stop,
                stream=False,
            )
            
            # Extract response
            text = result["choices"][0]["text"]
            finish_reason = result["choices"][0]["finish_reason"]
            
            # Token counts
            tokens_prompt = result["usage"]["prompt_tokens"]
            tokens_generated = result["usage"]["completion_tokens"]
            
            return GenerationResult(
                text=text,
                tokens_generated=tokens_generated,
                tokens_prompt=tokens_prompt,
                finish_reason=finish_reason,
                model_name=self._model_name,
            )
        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}") from e
    
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
        
        # Prepare stop sequences
        stop = params.stop_sequences or []
        
        try:
            # Call llama-cpp streaming generate
            stream = self._model.create_completion(
                prompt=params.prompt,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                stop=stop,
                stream=True,
            )
            
            # Yield text chunks
            for chunk in stream:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("text", "")
                    if delta:
                        yield delta
        except Exception as e:
            raise RuntimeError(f"Streaming generation failed: {str(e)}") from e
    
    def get_model_info(self) -> dict[str, str | int]:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        if not self.is_loaded():
            return {
                "name": "No model loaded",
                "type": "llama-cpp",
                "size_mb": 0,
                "context_length": 0,
                "quantization": "unknown",
                "architecture": "unknown",
            }
        
        # Extract info from model
        ctx_size = self._model.n_ctx() if hasattr(self._model, 'n_ctx') else 0
        
        # Estimate size from file
        size_mb = 0
        if self._model_path and Path(self._model_path).exists():
            size_mb = Path(self._model_path).stat().st_size // (1024 * 1024)
        
        # Infer quantization from filename
        quantization = "unknown"
        if self._model_path:
            name_lower = self._model_path.lower()
            if "q4_k_m" in name_lower:
                quantization = "Q4_K_M"
            elif "q8_0" in name_lower:
                quantization = "Q8_0"
            elif "q4_0" in name_lower:
                quantization = "Q4_0"
            elif "q5_k_m" in name_lower:
                quantization = "Q5_K_M"
        
        return {
            "name": self._model_name,
            "type": "llama-cpp",
            "size_mb": size_mb,
            "context_length": ctx_size,
            "quantization": quantization,
            "architecture": "llama",  # Generic, could parse from metadata
        }
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            # llama-cpp-python doesn't have explicit unload, just delete reference
            del self._model
            self._model = None
            self._model_path = None
            self._model_name = "Unknown"
