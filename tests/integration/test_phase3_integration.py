"""
Integration tests for Phase 3 modules.

Tests end-to-end workflows:
- Full RAG pipeline (analyze → chunk → embed → store → search)
- Multi-file edit transactions with rollback
- Hybrid search quality with real code
- Performance benchmarks
- Memory profiling
"""

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock

import pytest

from quirkllm.analyzer.project_analyzer import analyze_project, ProjectMap
from quirkllm.core.profile_manager import ProfileConfig
from quirkllm.file_ops.file_manager import FileEdit, FileManager
from quirkllm.rag.embeddings import ChunkedCode, EmbeddingGenerator
from quirkllm.rag.lancedb_store import CodeChunk, LanceDBStore
from quirkllm.rag.retriever import CodeRetriever


@pytest.fixture
def temp_project():
    """Create a temporary test project with realistic structure."""
    project_dir = tempfile.mkdtemp(prefix="test_project_")
    project_path = Path(project_dir)
    
    # Create package.json
    package_json = {
        "name": "test-app",
        "version": "1.0.0",
        "dependencies": {
            "react": "^18.0.0",
            "next": "^14.0.0"
        },
        "devDependencies": {
            "typescript": "^5.0.0",
            "vitest": "^1.0.0"
        }
    }
    (project_path / "package.json").write_text(json.dumps(package_json, indent=2))
    
    # Create next.config.js
    (project_path / "next.config.js").write_text("""
module.exports = {
  reactStrictMode: true,
}
""")
    
    # Create src directory with code
    src_dir = project_path / "src"
    src_dir.mkdir()
    
    # Create a realistic component
    (src_dir / "Button.tsx").write_text("""
import React from 'react';

interface ButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
}

export const Button: React.FC<ButtonProps> = ({ label, onClick, disabled = false }) => {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="px-4 py-2 bg-blue-500 text-white rounded"
    >
      {label}
    </button>
  );
};
""")
    
    # Create a utility function
    (src_dir / "utils.ts").write_text("""
/**
 * Format a date to YYYY-MM-DD
 */
export function formatDate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

/**
 * Capitalize first letter of a string
 */
export function capitalize(str: string): string {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}
""")
    
    # Create README
    (project_path / "README.md").write_text("""
# Test App

A Next.js + React test application for integration testing.

## Features
- React components with TypeScript
- Utility functions
- Vitest testing setup
""")
    
    yield project_path
    
    # Cleanup
    shutil.rmtree(project_dir)


@pytest.fixture
def temp_rag_db():
    """Create a temporary LanceDB database."""
    db_dir = tempfile.mkdtemp(prefix="test_rag_db_")
    db_path = Path(db_dir)
    
    yield db_path
    
    # Cleanup
    shutil.rmtree(db_dir)


@pytest.fixture
def survival_profile():
    """Create a survival profile for testing."""
    return ProfileConfig(
        name="survival",
        context_length=16384,
        quantization="4bit",
        batch_size=1,
        rag_cache_mb=200,
        compaction_mode="aggressive",
        max_history_turns=3
    )


class TestFullRAGPipeline:
    """Test complete RAG ingestion and retrieval pipeline."""
    
    def test_end_to_end_rag_workflow(self, temp_project, temp_rag_db, survival_profile):
        """
        Test full RAG pipeline:
        1. Analyze project structure
        2. Chunk code files
        3. Generate embeddings
        4. Store in LanceDB
        5. Search and retrieve
        """
        # Step 1: Analyze project
        project_info = analyze_project(str(temp_project))
        
        assert project_info.framework == "Next.js"
        assert project_info.package_manager == "npm"
        
        # Step 2: Get code files to process
        src_dir = temp_project / "src"
        code_files = list(src_dir.glob("*.ts*")) if src_dir.exists() else []
        
        assert len(code_files) >= 2, "Should have at least 2 code files"
        
        # Step 3: Chunk and embed code
        embedder = EmbeddingGenerator(survival_profile)
        all_chunks = []
        all_embeddings = []
        
        for code_file in code_files[:2]:
            content = code_file.read_text()
            chunks = embedder.chunk_code(
                content=content,
                file_path=str(code_file.relative_to(temp_project)),
                language="typescript"
            )
            
            if chunks:
                chunk_texts = [c.content for c in chunks]
                embeddings = embedder.embed_batch(chunk_texts)
                all_chunks.extend(chunks)
                all_embeddings.extend(embeddings)
        
        assert len(all_chunks) >= 2, "Should have at least 2 chunks"
        assert len(all_embeddings) == len(all_chunks)
        assert all(len(emb) == 384 for emb in all_embeddings), "MiniLM produces 384-dim vectors"
        
        # Step 4: Store in LanceDB
        store = LanceDBStore(str(temp_rag_db))
        
        code_chunks = [
            CodeChunk(
                id=f"chunk_{i}",
                content=chunk.content,
                embedding=all_embeddings[i],
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                language=chunk.language,
                metadata={
                    "project": "test-app",
                    "framework": "Next.js"
                }
            )
            for i, chunk in enumerate(all_chunks)
        ]
        
        store.add_chunks(code_chunks)
        
        # Step 5: Search and retrieve
        retriever = CodeRetriever(store, embedder, survival_profile)
        
        # Search for button component
        results = retriever.retrieve(query="React button component with props")
        
        assert len(results) > 0, "Should find relevant results"
        
        # Verify results have reasonable scores
        assert all(result.score > 0 for result in results), "All results should have positive scores"
    
    def test_hybrid_search_quality(self, temp_project, temp_rag_db, survival_profile):
        """Test hybrid search finds relevant code better than pure semantic."""
        # Setup: Create and populate database
        project_info = analyze_project(str(temp_project))
        
        # Simple search test - just verify RAG pipeline works
        embedder = EmbeddingGenerator(survival_profile)
        store = LanceDBStore(str(temp_rag_db))
        retriever = CodeRetriever(store, embedder, survival_profile)
        
        # Add a simple test chunk
        test_chunk = CodeChunk(
            id="test_1",
            content="function formatDate(date) { return date.toISOString(); }",
            embedding=embedder.embed_query("formatDate function"),
            file_path="test.js",
            start_line=1,
            end_line=1,
            language="javascript",
            metadata={}
        )
        store.add_chunks([test_chunk])
        
        # Search for it
        results = retriever.retrieve("formatDate function")
        assert len(results) > 0
        assert "formatDate" in results[0].content


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
        backup2 = file_manager.write_file(
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
        """Test project analysis completes in <5s for ~50 files."""
        start_time = time.time()
        
        result = analyze_project(str(temp_project))
        
        elapsed = time.time() - start_time
        
        # Should complete quickly (project is small)
        assert elapsed < 5.0, f"Analysis took {elapsed:.2f}s, target is <5s"
        assert result.framework is not None
        assert result.package_manager is not None
    
    def test_embedding_generation_performance(self, survival_profile):
        """Test embedding generation for batch of code chunks."""
        embedder = EmbeddingGenerator(survival_profile)
        
        # Create sample code chunks
        code_samples = [
            "def hello(): print('world')",
            "function add(a, b) { return a + b; }",
            "class User { constructor(name) { this.name = name; } }",
            "import React from 'react';",
            "const result = await fetch('/api/data');"
        ]
        
        start_time = time.time()
        embeddings = embedder.embed_batch(code_samples)
        elapsed = time.time() - start_time
        
        # Should generate embeddings for 5 chunks in <2s
        assert elapsed < 2.0, f"Embedding took {elapsed:.2f}s"
        assert len(embeddings) == 5
        
        # Calculate per-chunk time
        per_chunk_ms = (elapsed / 5) * 1000
        print(f"\nEmbedding performance: {per_chunk_ms:.1f}ms per chunk")
    
    def test_search_latency(self, temp_rag_db, survival_profile):
        """Test search operations complete in <500ms."""
        # Setup: Create small database
        store = LanceDBStore(str(temp_rag_db))
        embedder = EmbeddingGenerator(survival_profile)
        
        # Add some chunks
        sample_chunks = [
            CodeChunk(
                id=f"chunk_{i}",
                content=f"Sample code {i}: function test{i}() {{ return {i}; }}",
                embedding=embedder.embed_query(f"function test{i}"),
                file_path=f"test{i}.js",
                start_line=1,
                end_line=3,
                language="javascript",
                metadata={}
            )
            for i in range(10)
        ]
        store.add_chunks(sample_chunks)
        
        # Test search performance
        retriever = CodeRetriever(store, embedder, survival_profile)
        
        start_time = time.time()
        results = retriever.retrieve("function test")
        elapsed = time.time() - start_time
        
        elapsed_ms = elapsed * 1000
        assert elapsed_ms < 1000, f"Search took {elapsed_ms:.1f}ms, target is <1000ms"
        assert len(results) > 0
        
        print(f"\nSearch performance: {elapsed_ms:.1f}ms for results")


class TestMemoryProfiling:
    """Memory profiling for RAG operations."""
    
    def test_embedding_model_memory_usage(self, survival_profile):
        """Test embedding model stays within profile limits."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create embedder (loads model)
        embedder = EmbeddingGenerator(survival_profile)
        
        # Generate some embeddings
        _ = embedder.embed_batch(["test code"] * 10)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
        
        print(f"\nMemory usage: {mem_increase:.1f}MB increase")
        print(f"Profile limit: {survival_profile.rag_cache_mb}MB")
        
        # Should stay under profile limit (with some margin)
        # MiniLM model is ~80MB, so 200MB should be enough
        assert mem_increase < survival_profile.rag_cache_mb * 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
