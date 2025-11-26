"""
Model Downloader Module
Phase 2 - QuirkLLM

HuggingFace Hub entegrasyonu, resume support, progress bar, model doğrulama.
Plan: PHASE2_PLAN.md
"""
import os
import hashlib
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download, HfApi
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn


class ModelDownloader:
    """
    HuggingFace Hub'dan model indirme, resume support, progress bar, checksum doğrulama.
    """
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.quirkllm/models/")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.api = HfApi()

    def download(
        self, 
        model_name: str, 
        filename: str = "model.gguf",
        revision: Optional[str] = None,
        show_progress: bool = True
    ) -> str:
        """
        Modeli HuggingFace Hub'dan indirir. Resume support ve progress bar ile.
        
        Args:
            model_name: HuggingFace model adı (örn. 'TheBloke/deepseek-coder-1.3b-base-GGUF')
            filename: İndirilecek dosya adı (örn. 'deepseek-coder-1.3b-base.Q4_K_M.gguf')
            revision: İstenirse belirli bir versiyon
            show_progress: Progress bar gösterilsin mi
            
        Returns:
            İndirilen modelin dosya yolu
        """
        try:
            # HuggingFace Hub otomatik olarak resume support sağlar
            model_path = hf_hub_download(
                repo_id=model_name,
                filename=filename,
                revision=revision,
                cache_dir=self.cache_dir,
                resume_download=True,
            )
            return model_path
        except Exception as e:
            raise RuntimeError(f"Model indirme hatası: {str(e)}") from e

    def verify(self, model_path: str, expected_hash: Optional[str] = None) -> bool:
        """
        Model dosyasının bütünlüğünü doğrular (checksum).
        
        Args:
            model_path: İndirilen modelin dosya yolu
            expected_hash: Beklenen SHA256 hash (opsiyonel)
            
        Returns:
            bool: Doğrulama başarılıysa True
        """
        if not os.path.exists(model_path):
            return False
            
        # Eğer expected_hash verilmemişse sadece dosyanın varlığını kontrol et
        if expected_hash is None:
            return os.path.getsize(model_path) > 0
            
        # SHA256 hesapla
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest() == expected_hash

    def list_available_files(self, model_name: str) -> list[str]:
        """
        Model repo'sunda mevcut dosyaları listeler.
        
        Args:
            model_name: HuggingFace model adı
            
        Returns:
            Dosya adlarının listesi
        """
        try:
            repo_info = self.api.repo_info(repo_id=model_name, repo_type="model")
            return [f.rfilename for f in repo_info.siblings]
        except Exception as e:
            raise RuntimeError(f"Model dosyaları listelenemedi: {str(e)}") from e
