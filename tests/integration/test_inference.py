"""
Integration test for first inference (Phase 2)
Simple prompt → response flow without actual model download.
"""
import pytest
from unittest.mock import patch, MagicMock
from quirkllm.backends.llamacpp import LlamaCppBackend
from quirkllm.backends.base import GenerationParams, BackendType, create_backend
from quirkllm.model.downloader import ModelDownloader


class TestFirstInference:
    """Integration tests for model download + inference flow"""
    
    @patch('quirkllm.backends.llamacpp.Llama')
    @patch('quirkllm.backends.llamacpp.Path')
    def test_full_inference_flow(self, mock_path, mock_llama):
        """
        Complete flow: backend creation → model loading → inference
        """
        # Mock model path
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stem = "deepseek-coder-1.3b-q4"
        mock_path.return_value = mock_path_instance
        
        # Mock llama model
        mock_model = MagicMock()
        mock_model.create_completion.return_value = {
            "choices": [{
                "text": "\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 35
            }
        }
        mock_llama.return_value = mock_model
        
        # Create backend
        backend = create_backend(BackendType.LLAMACPP)
        assert not backend.is_loaded()
        
        # Load model
        backend.load_model(
            "/fake/deepseek-coder-1.3b-q4.gguf",
            n_ctx=32768,
            n_batch=512,
            n_gpu_layers=0
        )
        assert backend.is_loaded()
        
        # Generate code
        params = GenerationParams(
            prompt="Write a Python function to calculate fibonacci numbers:",
            max_tokens=150,
            temperature=0.2,
            stop_sequences=["###", "\n\n\n"]
        )
        result = backend.generate(params)
        
        # Verify
        assert "fibonacci" in result.text
        assert result.tokens_prompt == 15
        assert result.tokens_generated == 35
        assert result.finish_reason == "stop"
        assert "deepseek" in result.model_name.lower()
    
    @patch('quirkllm.backends.llamacpp.Llama')
    @patch('quirkllm.backends.llamacpp.Path')
    def test_streaming_inference(self, mock_path, mock_llama):
        """
        Streaming inference flow
        """
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stem = "test-model"
        mock_path.return_value = mock_path_instance
        
        # Mock streaming response
        mock_model = MagicMock()
        mock_model.create_completion.return_value = iter([
            {"choices": [{"text": "def"}]},
            {"choices": [{"text": " hello"}]},
            {"choices": [{"text": "():"}]},
            {"choices": [{"text": "\n"}]},
            {"choices": [{"text": "    return"}]},
            {"choices": [{"text": ' "Hello"'}]},
        ])
        mock_llama.return_value = mock_model
        
        # Create and load
        backend = create_backend(BackendType.LLAMACPP)
        backend.load_model("/fake/model.gguf")
        
        # Stream
        params = GenerationParams(
            prompt="Write hello function:",
            stream=True
        )
        chunks = list(backend.generate_stream(params))
        
        # Verify
        full_text = "".join(chunks)
        assert "def hello():" in full_text
        assert "return" in full_text
        assert len(chunks) == 6
    
    @patch('quirkllm.backends.llamacpp.Llama')
    @patch('quirkllm.backends.llamacpp.Path')
    def test_profile_based_inference(self, mock_path, mock_llama):
        """
        Profile-based parametreler ile inference
        """
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stem = "model"
        mock_path.return_value = mock_path_instance
        
        mock_model = MagicMock()
        mock_llama.return_value = mock_model
        
        backend = create_backend(BackendType.LLAMACPP)
        
        # Survival profile (16K ctx, batch 1)
        backend.load_model(
            "/fake/model.gguf",
            n_ctx=16384,
            n_batch=1,
            n_gpu_layers=0
        )
        
        # Verify Llama called with correct params
        mock_llama.assert_called_once()
        call_kwargs = mock_llama.call_args[1]
        assert call_kwargs["n_ctx"] == 16384
        assert call_kwargs["n_batch"] == 1
        assert call_kwargs["n_gpu_layers"] == 0
    
    @patch('quirkllm.model.downloader.hf_hub_download')
    def test_model_download_integration(self, mock_download, tmp_path):
        """
        Model downloader integration test
        """
        # Mock download
        fake_model_path = str(tmp_path / "model.gguf")
        with open(fake_model_path, "wb") as f:
            f.write(b"fake model data")
        
        mock_download.return_value = fake_model_path
        
        # Download model
        downloader = ModelDownloader(cache_dir=str(tmp_path))
        model_path = downloader.download(
            "test/model",
            filename="model.gguf"
        )
        
        # Verify
        assert model_path == fake_model_path
        assert downloader.verify(model_path) is True
        
        # Model path exists for backend
        import os
        assert os.path.exists(model_path)
