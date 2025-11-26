"""
Unit tests for ModelDownloader (Phase 2)
Mocked HuggingFace Hub API, resume support, progress bar.
"""
import os
import hashlib
import pytest
from unittest.mock import patch, MagicMock
from quirkllm.model.downloader import ModelDownloader


class TestModelDownloader:
    """Test suite for ModelDownloader"""
    
    def test_init_creates_cache_dir(self, tmp_path):
        """Cache dizini otomatik oluşturulmalı"""
        cache_dir = tmp_path / "models"
        dl = ModelDownloader(cache_dir=str(cache_dir))
        assert os.path.exists(cache_dir)
        assert dl.cache_dir == str(cache_dir)
    
    def test_init_default_cache_dir(self):
        """Default cache dir ~/.quirkllm/models/ olmalı"""
        dl = ModelDownloader()
        expected = os.path.expanduser("~/.quirkllm/models/")
        assert dl.cache_dir == expected
    
    @patch('quirkllm.model.downloader.hf_hub_download')
    def test_download_success(self, mock_download, tmp_path):
        """Model başarıyla indirilmeli"""
        mock_path = str(tmp_path / "model.gguf")
        mock_download.return_value = mock_path
        
        dl = ModelDownloader(cache_dir=str(tmp_path))
        result = dl.download("test/model", filename="model.gguf")
        
        assert result == mock_path
        mock_download.assert_called_once_with(
            repo_id="test/model",
            filename="model.gguf",
            revision=None,
            cache_dir=str(tmp_path),
            resume_download=True,
        )
    
    @patch('quirkllm.model.downloader.hf_hub_download')
    def test_download_with_revision(self, mock_download, tmp_path):
        """Revision parametresi doğru iletilmeli"""
        mock_path = str(tmp_path / "model.gguf")
        mock_download.return_value = mock_path
        
        dl = ModelDownloader(cache_dir=str(tmp_path))
        result = dl.download("test/model", filename="model.gguf", revision="main")
        
        mock_download.assert_called_once_with(
            repo_id="test/model",
            filename="model.gguf",
            revision="main",
            cache_dir=str(tmp_path),
            resume_download=True,
        )
    
    @patch('quirkllm.model.downloader.hf_hub_download')
    def test_download_error_handling(self, mock_download, tmp_path):
        """İndirme hatası RuntimeError fırlatmalı"""
        mock_download.side_effect = Exception("Network error")
        
        dl = ModelDownloader(cache_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="Model indirme hatası"):
            dl.download("test/model", filename="model.gguf")
    
    def test_verify_file_not_exists(self, tmp_path):
        """Olmayan dosya False dönmeli"""
        dl = ModelDownloader(cache_dir=str(tmp_path))
        assert dl.verify(str(tmp_path / "nonexistent.gguf")) is False
    
    def test_verify_without_hash(self, tmp_path):
        """Hash verilmezse sadece dosya varlığı kontrol edilmeli"""
        model_path = tmp_path / "model.gguf"
        model_path.write_text("dummy model data")
        
        dl = ModelDownloader(cache_dir=str(tmp_path))
        assert dl.verify(str(model_path)) is True
    
    def test_verify_with_correct_hash(self, tmp_path):
        """Doğru hash verilirse True dönmeli"""
        model_path = tmp_path / "model.gguf"
        content = b"dummy model data"
        model_path.write_bytes(content)
        
        expected_hash = hashlib.sha256(content).hexdigest()
        
        dl = ModelDownloader(cache_dir=str(tmp_path))
        assert dl.verify(str(model_path), expected_hash) is True
    
    def test_verify_with_wrong_hash(self, tmp_path):
        """Yanlış hash verilirse False dönmeli"""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"dummy model data")
        
        wrong_hash = "0" * 64  # Yanlış hash
        
        dl = ModelDownloader(cache_dir=str(tmp_path))
        assert dl.verify(str(model_path), wrong_hash) is False
    
    @patch('quirkllm.model.downloader.HfApi')
    def test_list_available_files(self, mock_api_class, tmp_path):
        """Model repo dosyaları listelenebilmeli"""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        
        # Mock repo info
        mock_file1 = MagicMock()
        mock_file1.rfilename = "model.gguf"
        mock_file2 = MagicMock()
        mock_file2.rfilename = "config.json"
        
        mock_repo_info = MagicMock()
        mock_repo_info.siblings = [mock_file1, mock_file2]
        mock_api.repo_info.return_value = mock_repo_info
        
        dl = ModelDownloader(cache_dir=str(tmp_path))
        files = dl.list_available_files("test/model")
        
        assert files == ["model.gguf", "config.json"]
        mock_api.repo_info.assert_called_once_with(repo_id="test/model", repo_type="model")
    
    @patch('quirkllm.model.downloader.HfApi')
    def test_list_available_files_error(self, mock_api_class, tmp_path):
        """Dosya listesi hatası RuntimeError fırlatmalı"""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.repo_info.side_effect = Exception("API error")
        
        dl = ModelDownloader(cache_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="Model dosyaları listelenemedi"):
            dl.list_available_files("test/model")
