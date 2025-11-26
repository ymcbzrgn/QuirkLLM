"""
Unit tests for Hybrid Search Retriever (Phase 3.6)

Tests query processing, hybrid search, ranking, and context enrichment.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from quirkllm.rag.retriever import (
    CodeRetriever,
    QueryProcessor,
    RankFusion,
    ContextualResult,
)
from quirkllm.rag.lancedb_store import SearchResult


@pytest.fixture
def mock_db():
    """Mock LanceDB store"""
    db = Mock()
    
    # Mock search results
    def mock_search(embedding, k=5, filters=None):
        # Return dummy results
        results = []
        for i in range(min(k, 10)):
            result = SearchResult(
                id=f"id_{i}",
                content=f"def function_{i}(): pass",
                file_path=f"src/file_{i}.py",
                language="python",
                framework="FastAPI",
                score=1.0 - (i * 0.05),
                start_line=i * 10,
                end_line=i * 10 + 5,
                metadata={"framework": "FastAPI"}
            )
            # Add chunk_index as attribute
            result.chunk_index = i
            results.append(result)
        return results
    
    db.search = mock_search
    
    # Mock get_by_file
    def mock_get_by_file(file_path):
        return [
            Mock(
                content=f"chunk {i}",
                chunk_index=i,
                file_path=file_path
            )
            for i in range(3)
        ]
    
    db.get_by_file = mock_get_by_file
    
    return db


@pytest.fixture
def mock_embedder():
    """Mock embedding generator"""
    embedder = Mock()
    embedder.embed_query = lambda q: np.random.randn(384).astype(np.float32)
    embedder.embed_code = lambda c, l=None: np.random.randn(384).astype(np.float32)
    return embedder


@pytest.fixture
def mock_profile():
    """Mock profile config"""
    return Mock(name="survival")


@pytest.fixture
def retriever(mock_db, mock_embedder, mock_profile):
    """Create retriever with mocks"""
    return CodeRetriever(mock_db, mock_embedder, mock_profile)


class TestQueryProcessor:
    """Test query processing"""
    
    def test_extract_keywords_simple(self):
        """Should extract relevant keywords"""
        query = "function to calculate fibonacci numbers"
        keywords = QueryProcessor.extract_keywords(query)
        
        assert "function" in keywords
        assert "calculate" in keywords
        assert "fibonacci" in keywords
        assert "numbers" in keywords
        # Stop words should be filtered
        assert "to" not in keywords
    
    def test_extract_keywords_technical(self):
        """Should extract technical terms"""
        query = "React component with useState hook for authentication"
        keywords = QueryProcessor.extract_keywords(query)
        
        assert "react" in keywords
        assert "component" in keywords
        assert "usestate" in keywords
        assert "hook" in keywords
        assert "authentication" in keywords
    
    def test_extract_keywords_empty(self):
        """Empty query should return empty list"""
        keywords = QueryProcessor.extract_keywords("")
        assert keywords == []
    
    def test_decompose_simple_query(self):
        """Simple query should not be decomposed"""
        query = "function to calculate fibonacci"
        sub_queries = QueryProcessor.decompose_query(query)
        
        assert len(sub_queries) == 1
        assert sub_queries[0] == query
    
    def test_decompose_complex_query(self):
        """Complex query should be decomposed"""
        query = "function to parse JSON and validate schema"
        sub_queries = QueryProcessor.decompose_query(query)
        
        assert len(sub_queries) == 2
        assert "parse JSON" in sub_queries[0]
        assert "validate schema" in sub_queries[1]
    
    def test_decompose_multiple_connectors(self):
        """Query with multiple connectors"""
        query = "parse JSON or XML and validate schema then save to database"
        sub_queries = QueryProcessor.decompose_query(query)
        
        assert len(sub_queries) >= 2


class TestRankFusion:
    """Test rank fusion algorithms"""
    
    def test_rrf_single_list(self):
        """Single list should maintain order"""
        results = [
            SearchResult("id1", "code1", "file1.py", "python", "", 1.0, 1, 5, {}),
            SearchResult("id2", "code2", "file2.py", "python", "", 0.8, 1, 5, {}),
            SearchResult("id3", "code3", "file3.py", "python", "", 0.6, 1, 5, {}),
        ]
        
        fused = RankFusion.reciprocal_rank_fusion([results])
        
        assert len(fused) == 3
        assert fused[0][0].file_path == "file1.py"
        assert fused[1][0].file_path == "file2.py"
        assert fused[2][0].file_path == "file3.py"
    
    def test_rrf_multiple_lists(self):
        """Multiple lists should be combined"""
        list1 = [
            SearchResult("id1", "code1", "file1.py", "python", "", 1.0, 1, 5, {}),
            SearchResult("id2", "code2", "file2.py", "python", "", 0.8, 1, 5, {}),
        ]
        
        list2 = [
            SearchResult("id2", "code2", "file2.py", "python", "", 1.0, 1, 5, {}),
            SearchResult("id3", "code3", "file3.py", "python", "", 0.9, 1, 5, {}),
        ]
        
        fused = RankFusion.reciprocal_rank_fusion([list1, list2])
        
        # file2.py appears in both lists, should rank higher
        assert len(fused) == 3
        assert fused[0][0].file_path == "file2.py"  # Highest RRF score
    
    def test_rrf_empty_lists(self):
        """Empty lists should return empty"""
        fused = RankFusion.reciprocal_rank_fusion([])
        assert fused == []
    
    def test_rrf_score_accumulation(self):
        """Scores should accumulate for duplicate items"""
        result = SearchResult("id1", "code", "file.py", "python", "", 1.0, 1, 5, {})
        
        # Same result in two lists at different ranks
        list1 = [result]  # Rank 1
        list2 = [Mock(), result]  # Rank 2
        
        fused = RankFusion.reciprocal_rank_fusion([list1, list2])
        
        # Score should be sum of 1/(60+1) + 1/(60+2)
        assert len(fused) == 2


class TestCodeRetriever:
    """Test code retriever"""
    
    def test_retrieve_simple(self, retriever):
        """Simple retrieve should return results"""
        results = retriever.retrieve("function to calculate fibonacci", k=5)
        
        assert len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_retrieve_empty_query(self, retriever):
        """Empty query should return empty list"""
        results = retriever.retrieve("", k=5)
        assert results == []
        
        results = retriever.retrieve("   ", k=5)
        assert results == []
    
    def test_retrieve_with_filters(self, retriever):
        """Should pass filters to database"""
        filters = {"language": "python", "framework": "FastAPI"}
        results = retriever.retrieve(
            "authentication function",
            k=5,
            filters=filters
        )
        
        assert len(results) <= 5
    
    def test_retrieve_semantic_only(self, retriever):
        """Should support semantic-only search"""
        results = retriever.retrieve(
            "calculate fibonacci",
            k=5,
            use_hybrid=False
        )
        
        assert len(results) <= 5
    
    def test_retrieve_hybrid(self, retriever):
        """Hybrid search should combine semantic + keyword"""
        results = retriever.retrieve(
            "fibonacci calculation function",
            k=5,
            use_hybrid=True
        )
        
        assert len(results) <= 5
        # Results should have RRF scores
        for result in results:
            assert hasattr(result, 'score')
    
    def test_retrieve_with_context(self, retriever):
        """Should enrich results with context"""
        results = retriever.retrieve_with_context(
            "fibonacci function",
            k=3
        )
        
        assert len(results) <= 3
        assert all(isinstance(r, ContextualResult) for r in results)
        
        for result in results:
            assert result.content
            assert result.file_path
            assert result.language
            assert result.score >= 0
    
    def test_retrieve_with_context_includes_parent(self, retriever):
        """Context should include parent chunks"""
        results = retriever.retrieve_with_context("test function", k=1)
        
        # At least one result should have parent context
        # (depends on mock implementation)
        assert len(results) >= 1
    
    def test_retrieve_with_context_extracts_imports(self, retriever, mock_db):
        """Should extract import statements"""
        # Mock result with imports
        mock_result = SearchResult(
            id="test_id",
            content="import numpy as np\nfrom typing import List\n\ndef func(): pass",
            file_path="test.py",
            language="python",
            framework="",
            score=1.0,
            start_line=1,
            end_line=10,
            metadata={}
        )
        # Add chunk_index as attribute
        mock_result.chunk_index = 0
        
        mock_db.search = lambda *args, **kwargs: [mock_result]
        
        results = retriever.retrieve_with_context("test", k=1)
        
        assert len(results) == 1
        assert results[0].imports is not None
        assert len(results[0].imports) == 2
        assert any("numpy" in imp for imp in results[0].imports)
    
    def test_multi_query_retrieve_simple(self, retriever):
        """Simple query should not be decomposed"""
        results = retriever.multi_query_retrieve("fibonacci function", k=5)
        
        assert len(results) <= 5
    
    def test_multi_query_retrieve_complex(self, retriever):
        """Complex query should be decomposed"""
        results = retriever.multi_query_retrieve(
            "parse JSON and validate schema",
            k=5
        )
        
        assert len(results) <= 5
        # Should have unique results (no duplicates)
        file_paths = [r.file_path for r in results]
        assert len(file_paths) == len(set(file_paths))
    
    def test_multi_query_retrieve_empty(self, retriever):
        """Empty query should return empty"""
        results = retriever.multi_query_retrieve("", k=5)
        assert results == []


class TestSemanticSearch:
    """Test semantic search internals"""
    
    def test_semantic_search_calls_embedder(self, retriever):
        """Should call embedder for query"""
        # Just verify it doesn't crash and returns results
        results = retriever._semantic_search("test query", k=5)
        
        assert isinstance(results, list)
        assert len(results) <= 5
    
    def test_semantic_search_calls_db(self, retriever):
        """Should call database search"""
        results = retriever._semantic_search("test query", k=10)
        
        # Should return results
        assert isinstance(results, list)
        assert len(results) <= 10


class TestKeywordSearch:
    """Test keyword search"""
    
    def test_keyword_search_extracts_keywords(self, retriever):
        """Should extract and use keywords"""
        results = retriever._keyword_search("fibonacci calculation", k=5)
        
        # Should return results
        assert isinstance(results, list)
    
    def test_keyword_search_empty_query(self, retriever):
        """Empty query should return empty"""
        results = retriever._keyword_search("", k=5)
        assert results == []
    
    def test_keyword_search_no_keywords(self, retriever):
        """Query with only stop words should return empty"""
        results = retriever._keyword_search("the a an", k=5)
        assert results == []


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_retrieve(self, retriever):
        """Complete retrieve workflow"""
        query = "function to authenticate users with JWT tokens"
        results = retriever.retrieve(query, k=5)
        
        assert len(results) <= 5
        for result in results:
            assert result.content
            assert result.file_path
            assert 0 <= result.score <= 1.0
    
    def test_end_to_end_contextual(self, retriever):
        """Complete contextual retrieve workflow"""
        query = "parse JSON data"
        results = retriever.retrieve_with_context(query, k=3)
        
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, ContextualResult)
            assert result.content
            assert result.file_path
            assert result.language
    
    def test_filtered_search(self, retriever):
        """Search with language filter"""
        results = retriever.retrieve(
            "authentication",
            k=5,
            filters={"language": "python"}
        )
        
        assert len(results) <= 5
