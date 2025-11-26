"""
Unit tests for Code Embeddings Generator (Phase 3.5)

Tests embedding generation:
- Model loading based on profile
- Single code embedding
- Batch code embedding
- Query embedding
- Code chunking
- Similarity computation
"""

import pytest
import numpy as np
from quirkllm.rag.embeddings import (
    EmbeddingGenerator,
    chunk_code,
    compute_similarity,
    PROFILE_MODELS,
)


@pytest.fixture
def embedder():
    """Create an embedding generator with default profile."""
    return EmbeddingGenerator(profile="survival")


@pytest.fixture
def sample_code():
    """Sample Python code for testing."""
    return """
def hello_world():
    \"\"\"Print hello world.\"\"\"
    print("Hello, World!")
    return True

if __name__ == "__main__":
    hello_world()
"""


class TestModelLoading:
    """Test suite for model loading"""

    def test_load_default_model(self):
        """Default model should load successfully"""
        gen = EmbeddingGenerator()
        
        assert gen is not None
        assert gen.model is not None
        assert gen.embedding_dim > 0

    def test_load_survival_profile(self):
        """Survival profile should load lightweight model"""
        gen = EmbeddingGenerator(profile="survival")
        
        assert gen.model_name == PROFILE_MODELS["survival"]
        assert gen.embedding_dim == 384  # MiniLM dimension

    def test_load_comfort_profile(self):
        """Comfort profile should load appropriate model"""
        gen = EmbeddingGenerator(profile="comfort")
        
        assert gen.model_name == PROFILE_MODELS["comfort"]

    def test_load_power_profile(self):
        """Power profile should load larger model"""
        gen = EmbeddingGenerator(profile="power")
        
        assert gen.model_name == PROFILE_MODELS["power"]

    def test_custom_model_name(self):
        """Should accept custom model name"""
        gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        
        assert gen.model_name == "all-MiniLM-L6-v2"

    def test_get_embedding_dimension(self, embedder):
        """Should return correct embedding dimension"""
        dim = embedder.get_embedding_dim()
        
        assert dim == 384
        assert isinstance(dim, int)

    def test_get_model_name(self, embedder):
        """Should return model name"""
        name = embedder.get_model_name()
        
        assert isinstance(name, str)
        assert len(name) > 0


class TestSingleCodeEmbedding:
    """Test suite for single code embedding"""

    def test_embed_simple_code(self, embedder, sample_code):
        """Should embed simple code successfully"""
        embedding = embedder.embed_code(sample_code)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32
        assert not np.all(embedding == 0)  # Should not be zero vector

    def test_embed_with_language(self, embedder):
        """Should embed code with language context"""
        code = "const x = 1;"
        embedding = embedder.embed_code(code, language="javascript")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_embed_empty_code(self, embedder):
        """Empty code should return zero vector"""
        embedding = embedder.embed_code("")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert np.all(embedding == 0)  # Zero vector

    def test_embed_whitespace_only(self, embedder):
        """Whitespace-only code should return zero vector"""
        embedding = embedder.embed_code("   \n\t  ")
        
        assert np.all(embedding == 0)

    def test_embed_different_languages(self, embedder):
        """Should handle different programming languages"""
        python_code = "def foo(): pass"
        js_code = "function foo() {}"
        
        py_emb = embedder.embed_code(python_code, language="python")
        js_emb = embedder.embed_code(js_code, language="javascript")
        
        assert py_emb.shape == js_emb.shape
        assert not np.array_equal(py_emb, js_emb)  # Different embeddings

    def test_embed_long_code(self, embedder):
        """Should handle long code snippets"""
        long_code = "\n".join([f"line_{i} = {i}" for i in range(1000)])
        embedding = embedder.embed_code(long_code)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)


class TestBatchEmbedding:
    """Test suite for batch embedding"""

    def test_embed_batch_simple(self, embedder):
        """Should embed batch of code snippets"""
        codes = [
            "def foo(): pass",
            "def bar(): return 1",
            "class Test: pass"
        ]
        
        embeddings = embedder.embed_batch(codes)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    def test_embed_batch_with_languages(self, embedder):
        """Should embed batch with language context"""
        codes = [
            "def foo(): pass",
            "function bar() {}",
            "const baz = () => {}"
        ]
        languages = ["python", "javascript", "javascript"]
        
        embeddings = embedder.embed_batch(codes, languages=languages)
        
        assert embeddings.shape == (3, 384)

    def test_embed_empty_batch(self, embedder):
        """Empty batch should return empty array"""
        embeddings = embedder.embed_batch([])
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0, 384)

    def test_embed_batch_with_empty_strings(self, embedder):
        """Batch with empty strings should handle gracefully"""
        codes = ["def foo(): pass", "", "def bar(): pass"]
        embeddings = embedder.embed_batch(codes)
        
        assert embeddings.shape == (3, 384)
        # Second embedding should be zero vector
        assert np.all(embeddings[1] == 0)
        # First and third should not be zero
        assert not np.all(embeddings[0] == 0)
        assert not np.all(embeddings[2] == 0)

    def test_embed_batch_custom_batch_size(self, embedder):
        """Should respect custom batch size"""
        codes = [f"def func_{i}(): pass" for i in range(10)]
        embeddings = embedder.embed_batch(codes, batch_size=5)
        
        assert embeddings.shape == (10, 384)

    def test_embed_large_batch(self, embedder):
        """Should handle large batches efficiently"""
        codes = [f"# Comment {i}" for i in range(100)]
        embeddings = embedder.embed_batch(codes)
        
        assert embeddings.shape == (100, 384)


class TestQueryEmbedding:
    """Test suite for query embedding"""

    def test_embed_simple_query(self, embedder):
        """Should embed natural language query"""
        query = "function to calculate fibonacci numbers"
        embedding = embedder.embed_query(query)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert not np.all(embedding == 0)

    def test_embed_empty_query(self, embedder):
        """Empty query should return zero vector"""
        embedding = embedder.embed_query("")
        
        assert np.all(embedding == 0)

    def test_embed_technical_query(self, embedder):
        """Should handle technical queries"""
        query = "React component with useState hook"
        embedding = embedder.embed_query(query)
        
        assert embedding.shape == (384,)
        assert not np.all(embedding == 0)

    def test_query_vs_code_embedding(self, embedder):
        """Query and code embeddings should be in same space"""
        query = "print hello world"
        code = "print('Hello, World!')"
        
        query_emb = embedder.embed_query(query)
        code_emb = embedder.embed_code(code)
        
        assert query_emb.shape == code_emb.shape
        # Should have some similarity
        similarity = compute_similarity(query_emb, code_emb)
        assert similarity > 0.3  # Reasonable similarity threshold


class TestCodeChunking:
    """Test suite for code chunking"""

    def test_chunk_small_code(self):
        """Small code should not be chunked"""
        code = "\n".join([f"line {i}" for i in range(50)])
        chunks = chunk_code(code, max_lines=200)
        
        assert len(chunks) == 1
        assert chunks[0][1] == 1  # Start line
        assert chunks[0][2] == 50  # End line

    def test_chunk_large_code(self):
        """Large code should be chunked"""
        code = "\n".join([f"line {i}" for i in range(500)])
        chunks = chunk_code(code, max_lines=200, overlap_lines=50)
        
        assert len(chunks) > 1
        # Check overlap
        assert chunks[1][1] < chunks[0][2]  # Second chunk starts before first ends

    def test_chunk_with_overlap(self):
        """Chunks should have specified overlap"""
        code = "\n".join([f"line {i}" for i in range(300)])
        chunks = chunk_code(code, max_lines=100, overlap_lines=20)
        
        assert len(chunks) >= 2
        # Verify overlap
        if len(chunks) >= 2:
            gap = chunks[1][1] - chunks[0][2]
            assert gap <= -19  # Should have overlap

    def test_chunk_empty_code(self):
        """Empty code should return empty list"""
        chunks = chunk_code("")
        
        assert chunks == []

    def test_chunk_exact_max_lines(self):
        """Code with exactly max_lines should not be chunked"""
        code = "\n".join([f"line {i}" for i in range(200)])
        chunks = chunk_code(code, max_lines=200)
        
        assert len(chunks) == 1


class TestSimilarityComputation:
    """Test suite for similarity computation"""

    def test_similarity_identical_vectors(self):
        """Identical vectors should have similarity 1.0"""
        vec = np.random.randn(384).astype(np.float32)
        similarity = compute_similarity(vec, vec)
        
        assert 0.99 <= similarity <= 1.01  # Allow floating point error

    def test_similarity_orthogonal_vectors(self):
        """Orthogonal vectors should have low similarity"""
        vec1 = np.array([1.0, 0.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0, 0.0])
        similarity = compute_similarity(vec1, vec2)
        
        assert similarity < 0.1

    def test_similarity_opposite_vectors(self):
        """Opposite vectors should have similarity 0"""
        vec1 = np.array([1.0, 0.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0, 0.0])
        similarity = compute_similarity(vec1, vec2)
        
        assert similarity == 0.0  # Clamped to 0

    def test_similarity_zero_vector(self):
        """Zero vector should return 0 similarity"""
        vec1 = np.random.randn(384).astype(np.float32)
        vec2 = np.zeros(384, dtype=np.float32)
        similarity = compute_similarity(vec1, vec2)
        
        assert similarity == 0.0

    def test_similarity_range(self):
        """Similarity should be in [0, 1] range"""
        vec1 = np.random.randn(384).astype(np.float32)
        vec2 = np.random.randn(384).astype(np.float32)
        similarity = compute_similarity(vec1, vec2)
        
        assert 0.0 <= similarity <= 1.0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_very_long_single_line(self, embedder):
        """Should handle very long single line"""
        long_line = "x = " + " + ".join([str(i) for i in range(10000)])
        embedding = embedder.embed_code(long_line)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_unicode_code(self, embedder):
        """Should handle unicode in code"""
        code = "# 中文注释\ndef 函数():\n    print('你好')"
        embedding = embedder.embed_code(code)
        
        assert not np.all(embedding == 0)

    def test_special_characters(self, embedder):
        """Should handle special characters"""
        code = "const emoji = '🚀💻🎉';"
        embedding = embedder.embed_code(code)
        
        assert embedding.shape == (384,)

    def test_mixed_languages_in_batch(self, embedder):
        """Should handle mixed languages in batch"""
        codes = [
            "def foo(): pass",
            "function bar() {}",
            "fn baz() -> i32 { 0 }",
            "public class Test {}"
        ]
        languages = ["python", "javascript", "rust", "java"]
        
        embeddings = embedder.embed_batch(codes, languages=languages)
        
        assert embeddings.shape == (4, 384)
        # All embeddings should be different
        for i in range(4):
            for j in range(i + 1, 4):
                assert not np.array_equal(embeddings[i], embeddings[j])


class TestIntegration:
    """Integration tests"""

    def test_end_to_end_workflow(self, embedder, sample_code):
        """Complete workflow: embed code, query, compute similarity"""
        # Embed code
        code_embedding = embedder.embed_code(sample_code, language="python")
        
        # Embed related query
        query = "function that prints hello world"
        query_embedding = embedder.embed_query(query)
        
        # Compute similarity
        similarity = compute_similarity(code_embedding, query_embedding)
        
        # Should have reasonable similarity
        assert 0.3 <= similarity <= 0.9  # Not too low, not perfect match

    def test_batch_consistency(self, embedder):
        """Batch embedding should match individual embeddings"""
        codes = ["def foo(): pass", "def bar(): pass"]
        
        # Individual embeddings
        emb1 = embedder.embed_code(codes[0])
        emb2 = embedder.embed_code(codes[1])
        
        # Batch embedding
        batch_emb = embedder.embed_batch(codes)
        
        # Should be very close (allow small numerical differences)
        assert np.allclose(batch_emb[0], emb1, atol=1e-5)
        assert np.allclose(batch_emb[1], emb2, atol=1e-5)

    def test_chunking_and_embedding(self, embedder):
        """Chunk large code and embed all chunks"""
        large_code = "\n".join([f"def func_{i}(): pass" for i in range(300)])
        
        # Chunk code
        chunks = chunk_code(large_code, max_lines=100)
        
        # Embed all chunks
        chunk_texts = [chunk[0] for chunk in chunks]
        embeddings = embedder.embed_batch(chunk_texts)
        
        assert len(chunks) == embeddings.shape[0]
        assert all(not np.all(emb == 0) for emb in embeddings)
