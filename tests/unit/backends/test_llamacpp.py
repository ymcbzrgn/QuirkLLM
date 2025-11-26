"""
Unit tests for LlamaCppBackend (Phase 2)
Mock llama-cpp-python library for testing.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from quirkllm.backends.llamacpp import LlamaCppBackend
from quirkllm.backends.base import GenerationParams, BackendType, create_backend


class TestLlamaCppBackend:
    """Test suite for LlamaCppBackend"""
    
    def test_init(self):
        """Backend başlangıç durumu"""
        backend = LlamaCppBackend()
        assert backend.is_loaded() is False
        assert backend._model is None
        assert backend._model_path is None
    
    @patch('quirkllm.backends.llamacpp.Llama')
    @patch('quirkllm.backends.llamacpp.Path')
    def test_load_model_success(self, mock_path, mock_llama):
        """Model başarıyla yüklenebilmeli"""
        # Mock Path.exists()
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stem = "test-model-q4"
        mock_path.return_value = mock_path_instance
        
        # Mock Llama instance
        mock_model = Mock()
        mock_llama.return_value = mock_model
        
        backend = LlamaCppBackend()
        backend.load_model(
            "/fake/path/model.gguf",
            n_ctx=32768,
            n_batch=512,
            n_gpu_layers=0
        )
        
        assert backend.is_loaded() is True
        assert backend._model == mock_model
        mock_llama.assert_called_once_with(
            model_path="/fake/path/model.gguf",
            n_ctx=32768,
            n_batch=512,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
    
    @patch('quirkllm.backends.llamacpp.Path')
    def test_load_model_file_not_found(self, mock_path):
        """Dosya yoksa FileNotFoundError fırlatmalı"""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        backend = LlamaCppBackend()
        with pytest.raises(FileNotFoundError, match="Model not found"):
            backend.load_model("/fake/nonexistent.gguf")
    
    @patch('quirkllm.backends.llamacpp.Llama')
    @patch('quirkllm.backends.llamacpp.Path')
    def test_load_model_error_handling(self, mock_path, mock_llama):
        """Model yükleme hatası RuntimeError fırlatmalı"""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        mock_llama.side_effect = Exception("CUDA out of memory")
        
        backend = LlamaCppBackend()
        with pytest.raises(RuntimeError, match="Model loading failed"):
            backend.load_model("/fake/path/model.gguf")
    
    def test_generate_not_loaded(self):
        """Model yüklü değilse RuntimeError fırlatmalı"""
        backend = LlamaCppBackend()
        params = GenerationParams(prompt="test")
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.generate(params)
    
    @patch('quirkllm.backends.llamacpp.Llama')
    @patch('quirkllm.backends.llamacpp.Path')
    def test_generate_success(self, mock_path, mock_llama):
        """Generate başarıyla çalışmalı"""
        # Setup
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stem = "test-model"
        mock_path.return_value = mock_path_instance
        
        mock_model = Mock()
        mock_model.create_completion.return_value = {
            "choices": [{
                "text": "Generated response",
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        }
        mock_llama.return_value = mock_model
        
        # Load and generate
        backend = LlamaCppBackend()
        backend.load_model("/fake/model.gguf")
        
        params = GenerationParams(
            prompt="Hello",
            max_tokens=100,
            temperature=0.7
        )
        result = backend.generate(params)
        
        assert result.text == "Generated response"
        assert result.tokens_prompt == 10
        assert result.tokens_generated == 5
        assert result.finish_reason == "stop"
        assert result.model_name == "test-model"
        
        mock_model.create_completion.assert_called_once_with(
            prompt="Hello",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop=[],
            stream=False,
        )
    
    @patch('quirkllm.backends.llamacpp.Llama')
    @patch('quirkllm.backends.llamacpp.Path')
    def test_generate_with_stop_sequences(self, mock_path, mock_llama):
        """Stop sequences parametresi iletilmeli"""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stem = "test-model"
        mock_path.return_value = mock_path_instance
        
        mock_model = Mock()
        mock_model.create_completion.return_value = {
            "choices": [{"text": "Response", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3}
        }
        mock_llama.return_value = mock_model
        
        backend = LlamaCppBackend()
        backend.load_model("/fake/model.gguf")
        
        params = GenerationParams(
            prompt="Test",
            stop_sequences=["###", "\n\n"]
        )
        backend.generate(params)
        
        call_args = mock_model.create_completion.call_args
        assert call_args[1]["stop"] == ["###", "\n\n"]
    
    def test_generate_stream_not_loaded(self):
        """Model yüklü değilse RuntimeError fırlatmalı"""
        backend = LlamaCppBackend()
        params = GenerationParams(prompt="test", stream=True)
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            list(backend.generate_stream(params))
    
    @patch('quirkllm.backends.llamacpp.Llama')
    @patch('quirkllm.backends.llamacpp.Path')
    def test_generate_stream_success(self, mock_path, mock_llama):
        """Streaming generate başarıyla çalışmalı"""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stem = "test-model"
        mock_path.return_value = mock_path_instance
        
        # Mock streaming response
        mock_model = Mock()
        mock_model.create_completion.return_value = iter([
            {"choices": [{"text": "Hello"}]},
            {"choices": [{"text": " world"}]},
            {"choices": [{"text": "!"}]},
        ])
        mock_llama.return_value = mock_model
        
        backend = LlamaCppBackend()
        backend.load_model("/fake/model.gguf")
        
        params = GenerationParams(prompt="Test", stream=True)
        chunks = list(backend.generate_stream(params))
        
        assert chunks == ["Hello", " world", "!"]
        mock_model.create_completion.assert_called_once_with(
            prompt="Test",
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop=[],
            stream=True,
        )
    
    @patch('quirkllm.backends.llamacpp.Llama')
    @patch('quirkllm.backends.llamacpp.Path')
    def test_get_model_info_loaded(self, mock_path, mock_llama):
        """Model info loaded durumda"""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stem = "deepseek-q4_k_m"
        mock_path_instance.stat.return_value.st_size = 1024 * 1024 * 500  # 500MB
        mock_path.return_value = mock_path_instance
        
        mock_model = Mock()
        mock_model.n_ctx.return_value = 32768
        mock_llama.return_value = mock_model
        
        backend = LlamaCppBackend()
        backend.load_model("/fake/deepseek-q4_k_m.gguf")
        
        info = backend.get_model_info()
        
        assert info["name"] == "deepseek-q4_k_m"
        assert info["type"] == "llama-cpp"
        assert info["size_mb"] == 500
        assert info["context_length"] == 32768
        assert info["quantization"] == "Q4_K_M"
        assert info["architecture"] == "llama"
    
    def test_get_model_info_not_loaded(self):
        """Model yüklü değilse default değerler"""
        backend = LlamaCppBackend()
        info = backend.get_model_info()
        
        assert info["name"] == "No model loaded"
        assert info["type"] == "llama-cpp"
        assert info["size_mb"] == 0
        assert info["context_length"] == 0
        assert info["quantization"] == "unknown"
    
    @patch('quirkllm.backends.llamacpp.Llama')
    @patch('quirkllm.backends.llamacpp.Path')
    def test_unload_model(self, mock_path, mock_llama):
        """Model unload memory temizlemeli"""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stem = "test-model"
        mock_path.return_value = mock_path_instance
        
        mock_model = Mock()
        mock_llama.return_value = mock_model
        
        backend = LlamaCppBackend()
        backend.load_model("/fake/model.gguf")
        assert backend.is_loaded() is True
        
        backend.unload_model()
        assert backend.is_loaded() is False
        assert backend._model is None
        assert backend._model_path is None
    
    def test_create_backend_llamacpp(self):
        """Factory llama-cpp backend oluşturmalı"""
        backend = create_backend(BackendType.LLAMACPP)
        assert isinstance(backend, LlamaCppBackend)
        assert backend.is_loaded() is False
