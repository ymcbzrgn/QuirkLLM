"""
Tests for LanceDB document storage functionality.

Test Categories:
1. Document Chunk Operations (3)
2. Document Search (2)
3. Delete by Source (2)
4. Document Stats (1)

Total: 8 tests
"""

import tempfile
import numpy as np
import pytest

from quirkllm.rag.lancedb_store import (
    LanceDBStore,
    DocumentChunk,
    DocumentSearchResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(db_path=tmpdir)
        yield store


@pytest.fixture
def sample_embedding():
    """Create a sample 384-dim embedding."""
    return np.random.rand(384).astype(np.float32)


@pytest.fixture
def sample_document_chunk(sample_embedding):
    """Create a sample DocumentChunk."""
    return DocumentChunk(
        id="test_chunk_1",
        content="This is sample content for testing.",
        embedding=sample_embedding,
        source_id="src_abc123",
        source_type="web",
        source_url="https://example.com/doc",
        title="Test Document",
        page_num=0,
        chunk_index=0,
        total_chunks=1,
        metadata={"test": True},
    )


# =============================================================================
# 1. Document Chunk Operations Tests (3)
# =============================================================================


class TestDocumentChunkOperations:
    """Tests for document chunk CRUD operations."""

    def test_add_document_chunks_single(self, temp_db, sample_document_chunk):
        """Test adding a single document chunk."""
        result = temp_db.add_document_chunks([sample_document_chunk])

        assert result == 1

    def test_add_document_chunks_multiple(self, temp_db, sample_embedding):
        """Test adding multiple document chunks."""
        chunks = []
        for i in range(5):
            chunk = DocumentChunk(
                id=f"chunk_{i}",
                content=f"Content for chunk {i}",
                embedding=sample_embedding,
                source_id="src_multi",
                source_type="pdf",
                source_url="/path/to/doc.pdf",
                title="Multi Chunk Doc",
                page_num=i,
                chunk_index=i,
                total_chunks=5,
                metadata={},
            )
            chunks.append(chunk)

        result = temp_db.add_document_chunks(chunks)

        assert result == 5

    def test_add_document_chunks_empty_list(self, temp_db):
        """Test adding empty list returns 0."""
        result = temp_db.add_document_chunks([])

        assert result == 0

    def test_documents_table_creation(self, temp_db):
        """Test that documents table is created on init."""
        table_names = temp_db.db.table_names()

        assert "documents" in table_names


# =============================================================================
# 2. Document Search Tests (2)
# =============================================================================


class TestDocumentSearch:
    """Tests for document search functionality."""

    def test_search_documents_basic(self, temp_db, sample_embedding):
        """Test basic document search."""
        # Add some documents
        chunks = [
            DocumentChunk(
                id=f"search_chunk_{i}",
                content=f"Searchable content {i}",
                embedding=sample_embedding,
                source_id="search_src",
                source_type="web",
                source_url="https://example.com",
                title="Search Test",
                page_num=0,
                chunk_index=i,
                total_chunks=3,
                metadata={},
            )
            for i in range(3)
        ]
        temp_db.add_document_chunks(chunks)

        # Search
        query = np.random.rand(384).astype(np.float32)
        results = temp_db.search_documents(query, k=5)

        assert len(results) <= 5
        assert all(isinstance(r, DocumentSearchResult) for r in results)

    def test_search_documents_filter_by_type(self, temp_db, sample_embedding):
        """Test document search with source_type filter."""
        # Add web document
        web_chunk = DocumentChunk(
            id="web_doc",
            content="Web content",
            embedding=sample_embedding,
            source_id="web_src",
            source_type="web",
            source_url="https://example.com",
            title="Web Doc",
            page_num=0,
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )

        # Add PDF document
        pdf_chunk = DocumentChunk(
            id="pdf_doc",
            content="PDF content",
            embedding=sample_embedding,
            source_id="pdf_src",
            source_type="pdf",
            source_url="/path/to/doc.pdf",
            title="PDF Doc",
            page_num=1,
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )

        temp_db.add_document_chunks([web_chunk, pdf_chunk])

        # Search only web documents
        query = sample_embedding
        results = temp_db.search_documents(query, k=10, source_type="web")

        # All results should be web type
        for result in results:
            assert result.source_type == "web"


# =============================================================================
# 3. Delete by Source Tests (2)
# =============================================================================


class TestDeleteBySource:
    """Tests for delete_by_source_id functionality."""

    def test_delete_by_source_id_success(self, temp_db, sample_embedding):
        """Test deleting documents by source_id."""
        # Add documents with specific source_id
        chunks = [
            DocumentChunk(
                id=f"delete_chunk_{i}",
                content=f"Content to delete {i}",
                embedding=sample_embedding,
                source_id="source_to_delete",
                source_type="web",
                source_url="https://delete.example.com",
                title="Delete Test",
                page_num=0,
                chunk_index=i,
                total_chunks=3,
                metadata={},
            )
            for i in range(3)
        ]
        temp_db.add_document_chunks(chunks)

        # Delete by source_id
        deleted = temp_db.delete_by_source_id("source_to_delete")

        assert deleted == 3

    def test_delete_by_source_id_not_found(self, temp_db):
        """Test deleting non-existent source_id returns 0."""
        deleted = temp_db.delete_by_source_id("nonexistent_source")

        assert deleted == 0


# =============================================================================
# 4. Document Stats Tests (1)
# =============================================================================


class TestDocumentStats:
    """Tests for document statistics."""

    def test_get_document_stats(self, temp_db, sample_embedding):
        """Test getting document statistics."""
        # Add mixed documents
        web_chunks = [
            DocumentChunk(
                id=f"stats_web_{i}",
                content=f"Web content {i}",
                embedding=sample_embedding,
                source_id="stats_web_src",
                source_type="web",
                source_url="https://stats.example.com",
                title="Stats Web",
                page_num=0,
                chunk_index=i,
                total_chunks=2,
                metadata={},
            )
            for i in range(2)
        ]

        pdf_chunks = [
            DocumentChunk(
                id=f"stats_pdf_{i}",
                content=f"PDF content {i}",
                embedding=sample_embedding,
                source_id="stats_pdf_src",
                source_type="pdf",
                source_url="/path/stats.pdf",
                title="Stats PDF",
                page_num=i,
                chunk_index=i,
                total_chunks=3,
                metadata={},
            )
            for i in range(3)
        ]

        temp_db.add_document_chunks(web_chunks + pdf_chunks)

        stats = temp_db.get_document_stats()

        assert stats["total_chunks"] == 5
        assert stats["by_type"]["web"] == 2
        assert stats["by_type"]["pdf"] == 3
