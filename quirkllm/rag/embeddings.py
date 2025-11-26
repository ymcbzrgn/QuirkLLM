"""
Code Embeddings Generator for QuirkLLM (Phase 3.5)

Generates semantic embeddings for code using sentence-transformers.
Profile-based model selection for optimal performance.
"""

from typing import List, Optional
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# Model configurations by profile
PROFILE_MODELS = {
    "survival": "all-MiniLM-L6-v2",      # 80MB, fast, 384-dim
    "comfort": "all-MiniLM-L6-v2",       # Same as survival for now
    "power": "all-mpnet-base-v2",        # 420MB, better quality, 768-dim
    "beast": "all-mpnet-base-v2",        # Same as power
}

# Default embedding dimension (matches LanceDB schema)
DEFAULT_EMBEDDING_DIM = 384


class EmbeddingGenerator:
    """
    Generate semantic embeddings for code and queries.
    
    Uses sentence-transformers with profile-based model selection.
    """
    
    def __init__(
        self,
        profile: str = "comfort",
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize embedding generator.
        
        Args:
            profile: RAM profile ("survival", "comfort", "power", "beast")
            model_name: Override model name (default: profile-based)
            cache_dir: Model cache directory (default: ~/.cache/huggingface)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.profile = profile.lower()
        self.model_name = model_name or PROFILE_MODELS.get(
            self.profile,
            PROFILE_MODELS["comfort"]
        )
        
        # Set cache directory
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._load_model(cache_dir)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def _load_model(self, cache_dir: Optional[str] = None) -> SentenceTransformer:
        """
        Load sentence-transformer model.
        
        Args:
            cache_dir: Model cache directory
        
        Returns:
            Loaded SentenceTransformer model
        """
        try:
            model = SentenceTransformer(
                self.model_name,
                cache_folder=cache_dir
            )
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def embed_code(
        self,
        code: str,
        language: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embedding for a single code snippet.
        
        Args:
            code: Code snippet to embed
            language: Programming language (optional, for future use)
        
        Returns:
            Embedding vector as numpy array
        """
        if not code or not code.strip():
            # Return zero vector for empty code
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        try:
            # Add language prefix if provided (improves context)
            text = f"[{language}] {code}" if language else code
            
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"Error embedding code: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _prepare_texts(self, codes: List[str], languages: Optional[List[str]]) -> List[str]:
        """Prepare texts with language prefixes."""
        if languages and len(languages) == len(codes):
            return [
                f"[{lang}] {code}" if lang else code
                for code, lang in zip(codes, languages)
            ]
        return codes
    
    def _filter_valid_texts(self, texts: List[str]) -> tuple:
        """Filter out empty texts and return valid indices and texts."""
        valid_indices = [i for i, text in enumerate(texts) if text and text.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        return valid_indices, valid_texts
    
    def _map_embeddings_to_result(
        self,
        embeddings: np.ndarray,
        valid_indices: List[int],
        total_count: int
    ) -> np.ndarray:
        """Map embeddings back to original indices with zeros for empty texts."""
        result = np.zeros((total_count, self.embedding_dim), dtype=np.float32)
        for idx, valid_idx in enumerate(valid_indices):
            result[valid_idx] = embeddings[idx]
        return result
    
    def embed_batch(
        self,
        codes: List[str],
        languages: Optional[List[str]] = None,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for multiple code snippets efficiently.
        
        Args:
            codes: List of code snippets
            languages: Optional list of languages (same length as codes)
            batch_size: Batch size for encoding (default: 32)
        
        Returns:
            Array of embeddings, shape (n_codes, embedding_dim)
        """
        if not codes:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        try:
            # Prepare texts with language prefixes
            texts = self._prepare_texts(codes, languages)
            
            # Filter out empty texts
            valid_indices, valid_texts = self._filter_valid_texts(texts)
            
            if not valid_texts:
                return np.zeros((len(codes), self.embedding_dim), dtype=np.float32)
            
            # Generate embeddings
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Map embeddings back to original indices
            result = self._map_embeddings_to_result(embeddings, valid_indices, len(codes))
            
            return result.astype(np.float32)
        except Exception as e:
            print(f"Error embedding batch: {e}")
            return np.zeros((len(codes), self.embedding_dim), dtype=np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a natural language query.
        
        Args:
            query: User query (e.g., "function to calculate fibonacci")
        
        Returns:
            Embedding vector as numpy array
        """
        if not query or not query.strip():
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        try:
            # Add query prefix to help model understand intent
            text = f"query: {query}"
            
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"Error embedding query: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def get_embedding_dim(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim
    
    def get_model_name(self) -> str:
        """
        Get loaded model name.
        
        Returns:
            Model name
        """
        return self.model_name


def chunk_code(
    code: str,
    max_lines: int = 200,
    overlap_lines: int = 50
) -> List[tuple]:
    """
    Split code into overlapping chunks.
    
    Args:
        code: Code to chunk
        max_lines: Maximum lines per chunk
        overlap_lines: Number of overlapping lines between chunks
    
    Returns:
        List of tuples (chunk_text, start_line, end_line)
    """
    if not code:
        return []
    
    lines = code.split('\n')
    total_lines = len(lines)
    
    if total_lines <= max_lines:
        # No chunking needed
        return [(code, 1, total_lines)]
    
    chunks = []
    start = 0
    
    while start < total_lines:
        end = min(start + max_lines, total_lines)
        chunk_lines = lines[start:end]
        chunk_text = '\n'.join(chunk_lines)
        
        chunks.append((
            chunk_text,
            start + 1,  # 1-indexed
            end
        ))
        
        # Move to next chunk with overlap
        start += max_lines - overlap_lines
        
        # Break if we've covered everything
        if end >= total_lines:
            break
    
    return chunks


def compute_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray
) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity score (0-1)
    """
    # Normalize vectors
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Compute cosine similarity
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    # Clamp to [0, 1] range
    return float(np.clip(similarity, 0.0, 1.0))
