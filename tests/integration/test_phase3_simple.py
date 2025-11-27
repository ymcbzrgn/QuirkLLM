"""
Integration tests for Phase 3 - Simplified version focusing on working tests.

Tests:
- Multi-file edit transactions with rollback
- Performance benchmarks for project analysis
"""

import shutil
import tempfile
import time
from pathlib import Path

import pytest

from quirkllm.analyzer.project_analyzer import analyze_project
from quirkllm.file_ops.file_manager import FileEdit, FileManager


@pytest.fixture
def temp_project():
    """Create a temporary test project."""
    project_dir = tempfile.mkdtemp(prefix="test_project_")
    project_path = Path(project_dir)
    
    # Create minimal project structure
    (project_path / "README.md").write_text("# Test Project\n")
    
    src_dir = project_path / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("def main():\n    print('Hello')\n")
    
    yield project_path
    
    # Cleanup
    shutil.rmtree(project_dir)


class TestMultiFileEditWorkflow:
    """Test file operations with multi-file transactions."""
    
    def test_atomic_multi_file_edit_success(self, temp_project):
        """Test successful multi-file edit creates backups and applies changes."""
        file_manager = FileManager(str(temp_project))
        
        # Create test files
        file1 = temp_project / "file1.txt"
        file2 = temp_project / "file2.txt"
        file1.write_text("Original content 1")
        file2.write_text("Original content 2")
        
        # Create edits
        edits = [
            FileEdit(
                file_path=str(file1),
                new_content="Modified content 1",
                reason="Test edit 1",
                create_backup=True
            ),
            FileEdit(
                file_path=str(file2),
                new_content="Modified content 2",
                reason="Test edit 2",
                create_backup=True
            )
        ]
        
        # Execute transaction
        backups = file_manager.multi_file_edit(edits, atomic=True)
        
        assert len(backups) == 2
        assert file1.read_text() == "Modified content 1"
        assert file2.read_text() == "Modified content 2"
        
        # Verify backups exist
        for backup in backups:
            assert backup is not None
            assert Path(backup.backup_path).exists()
    
    def test_atomic_multi_file_edit_rollback(self, temp_project):
        """Test multi-file edit rolls back on failure."""
        file_manager = FileManager(str(temp_project))
        
        # Create test files
        file1 = temp_project / "file1.txt"
        file2 = temp_project / "file2.txt"
        file1.write_text("Original content 1")
        file2.write_text("Original content 2")
        
        # Create edits with one that will fail (invalid path)
        edits = [
            FileEdit(
                file_path=str(file1),
                new_content="Modified content 1",
                reason="Test edit 1",
                create_backup=True
            ),
            FileEdit(
                file_path="/nonexistent/path/file.txt",  # This will fail
                new_content="Modified content 2",
                reason="Test edit 2",
                create_backup=True
            )
        ]
        
        # Execute transaction (should rollback)
        with pytest.raises(Exception):
            file_manager.multi_file_edit(edits, atomic=True)
        
        # Verify file1 was rolled back
        assert file1.read_text() == "Original content 1"
    
    def test_backup_and_rollback_integration(self, temp_project):
        """Test complete backup and rollback workflow."""
        file_manager = FileManager(str(temp_project))
        
        # Create and modify a file
        test_file = temp_project / "test.txt"
        original_content = "Version 1"
        test_file.write_text(original_content)
        
        # Edit 1
        backup1 = file_manager.write_file(
            str(test_file),
            "Version 2",
            create_backup=True,
            reason="First edit"
        )
        assert test_file.read_text() == "Version 2"
        
        # Edit 2
        _ = file_manager.write_file(
            str(test_file),
            "Version 3",
            create_backup=True,
            reason="Second edit"
        )
        assert test_file.read_text() == "Version 3"
        
        # List backups
        backups = file_manager.list_backups(str(test_file))
        assert len(backups) >= 2
        
        # Rollback to first backup
        file_manager.rollback_file(str(test_file), backup1.id)
        assert test_file.read_text() == "Version 1"


class TestPerformanceBenchmarks:
    """Performance benchmarks for Phase 3 operations."""
    
    def test_small_project_analysis_performance(self, temp_project):
        """Test project analysis completes in <5s for small projects."""
        start_time = time.time()
        
        result = analyze_project(str(temp_project))
        
        elapsed = time.time() - start_time
        
        # Should complete quickly (project is small)
        assert elapsed < 5.0, f"Analysis took {elapsed:.2f}s, target is <5s"
        
        # Basic validation
        assert result is not None
        print(f"\nProject analysis completed in {elapsed:.3f}s")


class TestRAGSystem:
    """Test RAG system components with correct API."""

    def test_chunk_code_basic(self):
        """Test chunk_code splits code correctly."""
        from quirkllm.rag.embeddings import chunk_code

        code = """def hello():
    print("Hello")

def world():
    print("World")
"""
        chunks = chunk_code(code)

        assert len(chunks) >= 1
        # Each chunk is (text, start_line, end_line)
        assert all(len(c) == 3 for c in chunks)
        assert all(isinstance(c[0], str) for c in chunks)

    def test_embedding_generator_initialization(self):
        """Test EmbeddingGenerator initializes correctly."""
        from quirkllm.rag.embeddings import EmbeddingGenerator

        embedder = EmbeddingGenerator("survival")

        assert embedder.get_model_name() == "all-MiniLM-L6-v2"
        assert embedder.get_embedding_dim() == 384

    def test_embed_query(self):
        """Test query embedding generation."""
        from quirkllm.rag.embeddings import EmbeddingGenerator

        embedder = EmbeddingGenerator("survival")
        query_emb = embedder.embed_query("test query")

        assert query_emb.shape == (384,)

    def test_embed_batch(self):
        """Test batch embedding generation."""
        from quirkllm.rag.embeddings import EmbeddingGenerator

        embedder = EmbeddingGenerator("survival")
        texts = ["hello world", "foo bar"]
        embeddings = embedder.embed_batch(texts)

        assert len(embeddings) == 2
        assert all(len(emb) == 384 for emb in embeddings)

    def test_lancedb_store_crud(self):
        """Test LanceDBStore basic CRUD operations."""
        import tempfile
        from quirkllm.rag.lancedb_store import LanceDBStore, CodeChunk
        from quirkllm.rag.embeddings import EmbeddingGenerator

        embedder = EmbeddingGenerator("survival")

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LanceDBStore(tmpdir)

            # Create chunk
            query_emb = embedder.embed_query("test code")
            chunk = CodeChunk(
                id="test1",
                content="def hello(): pass",
                embedding=query_emb.tolist(),
                file_path="test.py",
                start_line=1,
                end_line=1,
                language="python",
                metadata={"project": "test"}
            )

            # Add
            result = store.add_code_chunk(chunk)
            assert result is True

            # Search
            results = store.search(query_emb, k=1)
            assert len(results) >= 1

    def test_end_to_end_rag_pipeline(self):
        """Test complete RAG pipeline: chunk -> embed -> store -> search."""
        import tempfile
        from quirkllm.rag.embeddings import EmbeddingGenerator, chunk_code
        from quirkllm.rag.lancedb_store import LanceDBStore, CodeChunk

        # Sample code
        code = """
def calculate_sum(a, b):
    return a + b

def calculate_product(a, b):
    return a * b
"""
        # 1. Chunk code
        chunks = chunk_code(code)
        assert len(chunks) >= 1

        # 2. Generate embeddings
        embedder = EmbeddingGenerator("survival")
        chunk_texts = [c[0] for c in chunks]
        embeddings = embedder.embed_batch(chunk_texts)

        # 3. Store in LanceDB
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LanceDBStore(tmpdir)

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                code_chunk = CodeChunk(
                    id=f"chunk_{i}",
                    content=chunk[0],
                    embedding=emb.tolist(),
                    file_path="math_utils.py",
                    start_line=chunk[1],
                    end_line=chunk[2],
                    language="python",
                    metadata={}
                )
                store.add_code_chunk(code_chunk)

            # 4. Search
            query = "function to add numbers"
            query_emb = embedder.embed_query(query)
            results = store.search(query_emb, k=3)

            assert len(results) >= 1
            # Should find relevant code
            assert any("sum" in r.content.lower() or "add" in r.content.lower()
                      for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
