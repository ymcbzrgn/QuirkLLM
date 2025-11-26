"""
Unit tests for LanceDB Vector Store (Phase 3.4)

Tests vector database operations:
- Database initialization
- Adding code chunks (single and batch)
- Semantic search with embeddings
- Metadata filtering
- Project-level deletion
- Database statistics
"""

import pytest
import numpy as np
from pathlib import Path
from quirkllm.rag.lancedb_store import (
    LanceDBStore,
    CodeChunk,
    SearchResult,
)


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database."""
    db_path = tmp_path / "test_db"
    store = LanceDBStore(str(db_path))
    yield store
    # Cleanup
    store.clear_all()


@pytest.fixture
def sample_embedding():
    """Create a sample 384-dimensional embedding."""
    return np.random.randn(384).astype(np.float32)


@pytest.fixture
def sample_chunk(sample_embedding):
    """Create a sample code chunk."""
    return CodeChunk(
        id="chunk_001",
        content="def hello():\n    print('Hello, World!')",
        embedding=sample_embedding,
        file_path="src/main.py",
        language="python",
        framework="flask",
        project_root="/path/to/project",
        chunk_index=0,
        total_chunks=1,
        start_line=1,
        end_line=2,
        metadata={"imports": ["print"]}
    )


class TestDatabaseInitialization:
    """Test suite for database initialization"""

    def test_initialize_database(self, temp_db):
        """Database should initialize successfully"""
        assert temp_db is not None
        assert temp_db.db_path.exists()
        assert "code_chunks" in temp_db.db.table_names()

    def test_database_path_creation(self, tmp_path):
        """Database path should be created if it doesn't exist"""
        db_path = tmp_path / "new" / "nested" / "path"
        store = LanceDBStore(str(db_path))
        
        assert db_path.exists()
        assert store.db is not None

    def test_default_database_path(self):
        """Should use default path if none provided"""
        store = LanceDBStore()
        
        expected_path = Path.home() / ".quirkllm" / "rag"
        assert store.db_path == expected_path


class TestAddCodeChunk:
    """Test suite for adding code chunks"""

    def test_add_single_chunk(self, temp_db, sample_chunk):
        """Single chunk should be added successfully"""
        result = temp_db.add_code_chunk(sample_chunk)
        
        assert result is True
        stats = temp_db.get_stats()
        assert stats["total_chunks"] == 1

    def test_add_multiple_chunks_individually(self, temp_db, sample_embedding):
        """Multiple chunks should be added one by one"""
        chunks = [
            CodeChunk(
                id=f"chunk_{i:03d}",
                content=f"# Code {i}",
                embedding=sample_embedding,
                file_path=f"file_{i}.py",
                language="python",
            )
            for i in range(5)
        ]
        
        for chunk in chunks:
            temp_db.add_code_chunk(chunk)
        
        stats = temp_db.get_stats()
        assert stats["total_chunks"] == 5

    def test_chunk_with_metadata(self, temp_db, sample_embedding):
        """Chunk with metadata should be stored correctly"""
        chunk = CodeChunk(
            id="meta_chunk",
            content="const x = 1;",
            embedding=sample_embedding,
            file_path="src/app.ts",
            language="typescript",
            framework="next.js",
            metadata={
                "imports": ["react", "next"],
                "exports": ["default"],
                "complexity": 5
            }
        )
        
        result = temp_db.add_code_chunk(chunk)
        assert result is True
        
        # Retrieve and verify metadata
        results = temp_db.get_by_file("src/app.ts")
        assert len(results) == 1
        assert results[0].metadata["imports"] == ["react", "next"]
        assert results[0].metadata["complexity"] == 5


class TestBatchOperations:
    """Test suite for batch operations"""

    def test_add_documents_batch(self, temp_db, sample_embedding):
        """Batch of documents should be added efficiently"""
        chunks = [
            CodeChunk(
                id=f"batch_{i}",
                content=f"Code snippet {i}",
                embedding=sample_embedding,
                file_path=f"src/file{i}.py",
                language="python",
            )
            for i in range(10)
        ]
        
        count = temp_db.add_documents(chunks)
        
        assert count == 10
        stats = temp_db.get_stats()
        assert stats["total_chunks"] == 10

    def test_add_empty_batch(self, temp_db):
        """Empty batch should return 0"""
        count = temp_db.add_documents([])
        assert count == 0

    def test_batch_with_different_languages(self, temp_db, sample_embedding):
        """Batch with mixed languages should be handled"""
        chunks = [
            CodeChunk(id="py_chunk", content="# Python", embedding=sample_embedding, 
                     file_path="a.py", language="python"),
            CodeChunk(id="js_chunk", content="// JavaScript", embedding=sample_embedding,
                     file_path="b.js", language="javascript"),
            CodeChunk(id="ts_chunk", content="// TypeScript", embedding=sample_embedding,
                     file_path="c.ts", language="typescript"),
        ]
        
        count = temp_db.add_documents(chunks)
        assert count == 3
        
        stats = temp_db.get_stats()
        assert stats["by_language"]["python"] == 1
        assert stats["by_language"]["javascript"] == 1
        assert stats["by_language"]["typescript"] == 1


class TestSemanticSearch:
    """Test suite for semantic search"""

    def test_search_with_results(self, temp_db, sample_embedding):
        """Search should return relevant results"""
        # Add some chunks
        chunks = [
            CodeChunk(
                id=f"search_{i}",
                content=f"Function {i}",
                embedding=sample_embedding + i * 0.01,  # Slightly different embeddings
                file_path=f"file{i}.py",
                language="python",
            )
            for i in range(5)
        ]
        temp_db.add_documents(chunks)
        
        # Search with similar embedding
        results = temp_db.search(sample_embedding, k=3)
        
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(hasattr(r, 'score') for r in results)

    def test_search_with_filters(self, temp_db, sample_embedding):
        """Search should respect metadata filters"""
        # Add chunks with different languages
        chunks = [
            CodeChunk(id="py1", content="Python 1", embedding=sample_embedding,
                     file_path="a.py", language="python"),
            CodeChunk(id="py2", content="Python 2", embedding=sample_embedding,
                     file_path="b.py", language="python"),
            CodeChunk(id="js1", content="JavaScript 1", embedding=sample_embedding,
                     file_path="c.js", language="javascript"),
        ]
        temp_db.add_documents(chunks)
        
        # Search with language filter
        results = temp_db.search(
            sample_embedding,
            k=10,
            filter_conditions={"language": "python"}
        )
        
        assert len(results) == 2
        assert all(r.language == "python" for r in results)

    def test_search_with_k_limit(self, temp_db, sample_embedding):
        """Search should respect k limit"""
        # Add 10 chunks
        chunks = [
            CodeChunk(id=f"k_{i}", content=f"Code {i}", embedding=sample_embedding,
                     file_path=f"f{i}.py", language="python")
            for i in range(10)
        ]
        temp_db.add_documents(chunks)
        
        # Search with k=3
        results = temp_db.search(sample_embedding, k=3)
        
        assert len(results) <= 3

    def test_search_empty_database(self, temp_db, sample_embedding):
        """Search on empty database should return empty list"""
        results = temp_db.search(sample_embedding, k=10)
        assert results == []


class TestDeletion:
    """Test suite for deletion operations"""

    def test_delete_by_project(self, temp_db, sample_embedding):
        """Should delete all chunks from a specific project"""
        # Add chunks from two projects
        project1_chunks = [
            CodeChunk(id=f"p1_{i}", content=f"P1 Code {i}", embedding=sample_embedding,
                     file_path=f"p1/file{i}.py", language="python", project_root="/project1")
            for i in range(3)
        ]
        project2_chunks = [
            CodeChunk(id=f"p2_{i}", content=f"P2 Code {i}", embedding=sample_embedding,
                     file_path=f"p2/file{i}.py", language="python", project_root="/project2")
            for i in range(2)
        ]
        
        temp_db.add_documents(project1_chunks + project2_chunks)
        assert temp_db.get_stats()["total_chunks"] == 5
        
        # Delete project1
        deleted = temp_db.delete_by_project("/project1")
        
        assert deleted == 3
        assert temp_db.get_stats()["total_chunks"] == 2

    def test_delete_nonexistent_project(self, temp_db):
        """Deleting non-existent project should return 0"""
        deleted = temp_db.delete_by_project("/nonexistent")
        assert deleted == 0

    def test_clear_all(self, temp_db, sample_embedding):
        """Clear all should remove all data"""
        # Add some data
        chunks = [
            CodeChunk(id=f"clear_{i}", content=f"Code {i}", embedding=sample_embedding,
                     file_path=f"f{i}.py", language="python")
            for i in range(5)
        ]
        temp_db.add_documents(chunks)
        assert temp_db.get_stats()["total_chunks"] == 5
        
        # Clear all
        result = temp_db.clear_all()
        
        assert result is True
        assert temp_db.get_stats()["total_chunks"] == 0


class TestStatistics:
    """Test suite for database statistics"""

    def test_get_stats_empty(self, temp_db):
        """Stats for empty database should return zeros"""
        stats = temp_db.get_stats()
        
        assert stats["total_chunks"] == 0
        assert stats["by_language"] == {}
        assert stats["by_framework"] == {}
        assert stats["by_project"] == {}

    def test_get_stats_with_data(self, temp_db, sample_embedding):
        """Stats should reflect database contents"""
        chunks = [
            CodeChunk(id="stat1", content="Code 1", embedding=sample_embedding,
                     file_path="a.py", language="python", framework="django", project_root="/proj1"),
            CodeChunk(id="stat2", content="Code 2", embedding=sample_embedding,
                     file_path="b.py", language="python", framework="flask", project_root="/proj1"),
            CodeChunk(id="stat3", content="Code 3", embedding=sample_embedding,
                     file_path="c.js", language="javascript", framework="react", project_root="/proj2"),
        ]
        temp_db.add_documents(chunks)
        
        stats = temp_db.get_stats()
        
        assert stats["total_chunks"] == 3
        assert stats["by_language"]["python"] == 2
        assert stats["by_language"]["javascript"] == 1
        assert stats["by_framework"]["django"] == 1
        assert stats["by_framework"]["flask"] == 1
        assert stats["by_framework"]["react"] == 1
        assert stats["by_project"]["/proj1"] == 2
        assert stats["by_project"]["/proj2"] == 1


class TestFileOperations:
    """Test suite for file-based operations"""

    def test_get_by_file(self, temp_db, sample_embedding):
        """Should retrieve all chunks for a specific file"""
        # Add chunks from multiple files
        chunks = [
            CodeChunk(id="f1_c1", content="File1 Chunk1", embedding=sample_embedding,
                     file_path="src/main.py", language="python", start_line=1, end_line=10),
            CodeChunk(id="f1_c2", content="File1 Chunk2", embedding=sample_embedding,
                     file_path="src/main.py", language="python", start_line=11, end_line=20),
            CodeChunk(id="f2_c1", content="File2 Chunk1", embedding=sample_embedding,
                     file_path="src/utils.py", language="python", start_line=1, end_line=15),
        ]
        temp_db.add_documents(chunks)
        
        # Get chunks for main.py
        results = temp_db.get_by_file("src/main.py")
        
        assert len(results) == 2
        assert all(r.file_path == "src/main.py" for r in results)

    def test_get_by_file_nonexistent(self, temp_db):
        """Getting non-existent file should return empty list"""
        results = temp_db.get_by_file("nonexistent.py")
        assert results == []


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_large_embedding_dimension(self, temp_db):
        """Should handle embeddings of correct dimension"""
        # 384 is the expected dimension for all-MiniLM-L6-v2
        embedding = np.random.randn(384).astype(np.float32)
        chunk = CodeChunk(
            id="large_emb",
            content="Test",
            embedding=embedding,
            file_path="test.py",
            language="python"
        )
        
        result = temp_db.add_code_chunk(chunk)
        assert result is True

    def test_unicode_content(self, temp_db, sample_embedding):
        """Should handle unicode in code content"""
        chunk = CodeChunk(
            id="unicode",
            content="# 中文注释\ndef 函数():\n    print('你好')",
            embedding=sample_embedding,
            file_path="中文.py",
            language="python"
        )
        
        result = temp_db.add_code_chunk(chunk)
        assert result is True
        
        results = temp_db.get_by_file("中文.py")
        assert len(results) == 1
        assert "中文" in results[0].content

    def test_special_characters_in_path(self, temp_db, sample_embedding):
        """Should handle special characters in file paths"""
        chunk = CodeChunk(
            id="special",
            content="code",
            embedding=sample_embedding,
            file_path="src/components/@special/file-name.tsx",
            language="typescript"
        )
        
        result = temp_db.add_code_chunk(chunk)
        assert result is True

    def test_empty_content(self, temp_db, sample_embedding):
        """Should handle empty content"""
        chunk = CodeChunk(
            id="empty",
            content="",
            embedding=sample_embedding,
            file_path="empty.py",
            language="python"
        )
        
        result = temp_db.add_code_chunk(chunk)
        assert result is True

    def test_very_long_content(self, temp_db, sample_embedding):
        """Should handle very long code content"""
        long_content = "\n".join([f"line {i}" for i in range(10000)])
        chunk = CodeChunk(
            id="long",
            content=long_content,
            embedding=sample_embedding,
            file_path="long.py",
            language="python"
        )
        
        result = temp_db.add_code_chunk(chunk)
        assert result is True


class TestDataIntegrity:
    """Test data integrity and consistency"""

    def test_chunk_retrieval_matches_insertion(self, temp_db, sample_embedding):
        """Retrieved chunk should match inserted chunk"""
        original = CodeChunk(
            id="integrity_test",
            content="def test(): pass",
            embedding=sample_embedding,
            file_path="test.py",
            language="python",
            framework="pytest",
            project_root="/test/project",
            chunk_index=5,
            total_chunks=10,
            start_line=100,
            end_line=105,
            metadata={"test": "value", "number": 42}
        )
        
        temp_db.add_code_chunk(original)
        results = temp_db.get_by_file("test.py")
        
        assert len(results) == 1
        retrieved = results[0]
        
        assert retrieved.id == original.id
        assert retrieved.content == original.content
        assert retrieved.file_path == original.file_path
        assert retrieved.language == original.language
        assert retrieved.framework == original.framework
        assert retrieved.start_line == original.start_line
        assert retrieved.end_line == original.end_line
        assert retrieved.metadata == original.metadata
