"""
Unit tests for MLX Backend (Phase 2.6)
Mock-based tests for platform-specific backend.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import platform
from quirkllm.backends.mlx_backend import (
    MLXBackend,
    is_mlx_available,
    check_mlx_import,
)
from quirkllm.backends.base import GenerationParams, BackendType, create_backend


class TestPlatformDetection:
    """Test suite for platform detection"""
    
    @patch('platform.system')
    @patch('platform.machine')
    def test_is_mlx_available_mac_arm(self, mock_machine, mock_system):
        """MLX should be available on macOS ARM64"""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"
        
        assert is_mlx_available() is True
    
    @patch('platform.system')
    @patch('platform.machine')
    def test_is_mlx_available_mac_intel(self, mock_machine, mock_system):
        """MLX should NOT be available on macOS Intel"""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "x86_64"
        
        assert is_mlx_available() is False
    
    @patch('platform.system')
    @patch('platform.machine')
    def test_is_mlx_available_linux_arm(self, mock_machine, mock_system):
        """MLX should NOT be available on Linux ARM"""
        mock_system.return_value = "Linux"
        mock_machine.return_value = "aarch64"
        
        assert is_mlx_available() is False
    
    @patch('platform.system')
    @patch('platform.machine')
    def test_is_mlx_available_windows(self, mock_machine, mock_system):
        """MLX should NOT be available on Windows"""
        mock_system.return_value = "Windows"
        mock_machine.return_value = "AMD64"
        
        assert is_mlx_available() is False


class TestMLXImport:
    """Test suite for MLX import checking"""
    
    def test_check_mlx_import_not_installed(self):
        """Should return False when MLX not installed"""
        # On non-Mac or when MLX not installed, import will fail
        success, error = check_mlx_import()
        
        # Either success (if on Mac with MLX) or failure expected
        assert isinstance(success, bool)
        if not success:
            assert error is not None
            assert isinstance(error, str)


class TestMLXBackend:
    """Test suite for MLXBackend"""
    
    @patch('quirkllm.backends.mlx_backend.is_mlx_available')
    def test_init_wrong_platform(self, mock_available):
        """Should raise error on non-Mac/non-ARM platform"""
        mock_available.return_value = False
        
        with pytest.raises(RuntimeError, match="requires macOS with Apple Silicon"):
            MLXBackend()
    
    @patch('quirkllm.backends.mlx_backend.check_mlx_import')
    @patch('quirkllm.backends.mlx_backend.is_mlx_available')
    def test_init_mlx_not_installed(self, mock_available, mock_import):
        """Should raise error when MLX not installed"""
        mock_available.return_value = True
        mock_import.return_value = (False, "No module named 'mlx'")
        
        with pytest.raises(RuntimeError, match="MLX not installed"):
            MLXBackend()
    
    @patch('quirkllm.backends.mlx_backend.check_mlx_import')
    @patch('quirkllm.backends.mlx_backend.is_mlx_available')
    def test_init_success(self, mock_available, mock_import):
        """Should initialize when platform and MLX are correct"""
        mock_available.return_value = True
        mock_import.return_value = (True, None)
        
        # Mock MLX modules
        with patch.dict('sys.modules', {
            'mlx': MagicMock(),
            'mlx.core': MagicMock(),
            'mlx_lm': MagicMock(),
        }):
            backend = MLXBackend()
            assert backend.is_loaded() is False
    
    @patch('quirkllm.backends.mlx_backend.Path')
    @patch('quirkllm.backends.mlx_backend.check_mlx_import')
    @patch('quirkllm.backends.mlx_backend.is_mlx_available')
    def test_load_model_not_found(self, mock_available, mock_import, mock_path):
        """Should raise FileNotFoundError for missing model"""
        mock_available.return_value = True
        mock_import.return_value = (True, None)
        
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        with patch.dict('sys.modules', {
            'mlx': MagicMock(),
            'mlx.core': MagicMock(),
            'mlx_lm': MagicMock(),
        }):
            backend = MLXBackend()
            
            with pytest.raises(FileNotFoundError, match="Model not found"):
                backend.load_model("/fake/model")
    
    @patch('quirkllm.backends.mlx_backend.Path')
    @patch('quirkllm.backends.mlx_backend.check_mlx_import')
    @patch('quirkllm.backends.mlx_backend.is_mlx_available')
    def test_load_model_success(self, mock_available, mock_import, mock_path):
        """Should load model successfully"""
        mock_available.return_value = True
        mock_import.return_value = (True, None)
        
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.name = "test-mlx-model"
        mock_path.return_value = mock_path_instance
        
        mock_mlx_lm = MagicMock()
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
        
        with patch.dict('sys.modules', {
            'mlx': MagicMock(),
            'mlx.core': MagicMock(),
            'mlx_lm': mock_mlx_lm,
        }):
            backend = MLXBackend()
            backend.mlx_lm = mock_mlx_lm
            backend.load_model("/fake/model")
            
            assert backend.is_loaded() is True
            assert backend._model == mock_model
            assert backend._tokenizer == mock_tokenizer
    
    @patch('quirkllm.backends.mlx_backend.check_mlx_import')
    @patch('quirkllm.backends.mlx_backend.is_mlx_available')
    def test_generate_not_loaded(self, mock_available, mock_import):
        """Should raise error when model not loaded"""
        mock_available.return_value = True
        mock_import.return_value = (True, None)
        
        with patch.dict('sys.modules', {
            'mlx': MagicMock(),
            'mlx.core': MagicMock(),
            'mlx_lm': MagicMock(),
        }):
            backend = MLXBackend()
            params = GenerationParams(prompt="test")
            
            with pytest.raises(RuntimeError, match="Model not loaded"):
                backend.generate(params)
    
    @patch('quirkllm.backends.mlx_backend.Path')
    @patch('quirkllm.backends.mlx_backend.check_mlx_import')
    @patch('quirkllm.backends.mlx_backend.is_mlx_available')
    def test_generate_success(self, mock_available, mock_import, mock_path):
        """Should generate text successfully"""
        mock_available.return_value = True
        mock_import.return_value = (True, None)
        
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.name = "test-model"
        mock_path.return_value = mock_path_instance
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = [
            [1, 2, 3, 4, 5],  # prompt tokens
            [10, 11, 12, 13],  # response tokens
        ]
        
        # Mock MLX generation
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (Mock(), mock_tokenizer)
        mock_mlx_lm.generate.return_value = "Generated response text"
        
        with patch.dict('sys.modules', {
            'mlx': MagicMock(),
            'mlx.core': MagicMock(),
            'mlx_lm': mock_mlx_lm,
        }):
            backend = MLXBackend()
            backend.mlx_lm = mock_mlx_lm
            backend.load_model("/fake/model")
            
            params = GenerationParams(prompt="Hello", max_tokens=100)
            result = backend.generate(params)
            
            assert result.text == "Generated response text"
            assert result.tokens_prompt == 5
            assert result.tokens_generated == 4
            assert result.model_name == "test-model"
    
    @patch('quirkllm.backends.mlx_backend.check_mlx_import')
    @patch('quirkllm.backends.mlx_backend.is_mlx_available')
    def test_generate_stream_not_loaded(self, mock_available, mock_import):
        """Should raise error when streaming without loaded model"""
        mock_available.return_value = True
        mock_import.return_value = (True, None)
        
        with patch.dict('sys.modules', {
            'mlx': MagicMock(),
            'mlx.core': MagicMock(),
            'mlx_lm': MagicMock(),
        }):
            backend = MLXBackend()
            params = GenerationParams(prompt="test", stream=True)
            
            with pytest.raises(RuntimeError, match="Model not loaded"):
                list(backend.generate_stream(params))
    
    @patch('quirkllm.backends.mlx_backend.Path')
    @patch('quirkllm.backends.mlx_backend.check_mlx_import')
    @patch('quirkllm.backends.mlx_backend.is_mlx_available')
    def test_get_model_info_loaded(self, mock_available, mock_import, mock_path):
        """Should return model info when loaded"""
        mock_available.return_value = True
        mock_import.return_value = (True, None)
        
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.name = "mlx-model"
        mock_path_instance.rglob.return_value = []
        mock_path.return_value = mock_path_instance
        
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (Mock(), Mock())
        
        with patch.dict('sys.modules', {
            'mlx': MagicMock(),
            'mlx.core': MagicMock(),
            'mlx_lm': mock_mlx_lm,
        }):
            backend = MLXBackend()
            backend.mlx_lm = mock_mlx_lm
            backend.load_model("/fake/model")
            
            info = backend.get_model_info()
            
            assert info["name"] == "mlx-model"
            assert info["type"] == "mlx"
            assert info["architecture"] == "mlx-metal"
            assert "quantization" in info
    
    @patch('quirkllm.backends.mlx_backend.check_mlx_import')
    @patch('quirkllm.backends.mlx_backend.is_mlx_available')
    def test_get_model_info_not_loaded(self, mock_available, mock_import):
        """Should return default info when not loaded"""
        mock_available.return_value = True
        mock_import.return_value = (True, None)
        
        with patch.dict('sys.modules', {
            'mlx': MagicMock(),
            'mlx.core': MagicMock(),
            'mlx_lm': MagicMock(),
        }):
            backend = MLXBackend()
            info = backend.get_model_info()
            
            assert info["name"] == "No model loaded"
            assert info["type"] == "mlx"
            assert info["size_mb"] == 0
    
    @patch('quirkllm.backends.mlx_backend.Path')
    @patch('quirkllm.backends.mlx_backend.check_mlx_import')
    @patch('quirkllm.backends.mlx_backend.is_mlx_available')
    def test_unload_model(self, mock_available, mock_import, mock_path):
        """Should unload model and free memory"""
        mock_available.return_value = True
        mock_import.return_value = (True, None)
        
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.name = "model"
        mock_path.return_value = mock_path_instance
        
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (Mock(), Mock())
        
        with patch.dict('sys.modules', {
            'mlx': MagicMock(),
            'mlx.core': MagicMock(),
            'mlx_lm': mock_mlx_lm,
        }):
            backend = MLXBackend()
            backend.mlx_lm = mock_mlx_lm
            backend.load_model("/fake/model")
            
            assert backend.is_loaded() is True
            
            backend.unload_model()
            
            assert backend.is_loaded() is False
            assert backend._model is None
            assert backend._tokenizer is None
    
    @patch('quirkllm.backends.mlx_backend.check_mlx_import')
    @patch('quirkllm.backends.mlx_backend.is_mlx_available')
    def test_create_backend_mlx(self, mock_available, mock_import):
        """Factory should create MLX backend"""
        mock_available.return_value = True
        mock_import.return_value = (True, None)
        
        with patch.dict('sys.modules', {
            'mlx': MagicMock(),
            'mlx.core': MagicMock(),
            'mlx_lm': MagicMock(),
        }):
            backend = create_backend(BackendType.MLX)
            assert isinstance(backend, MLXBackend)
