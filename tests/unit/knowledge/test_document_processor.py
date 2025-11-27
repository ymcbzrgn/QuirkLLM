"""
Tests for DocumentProcessor class.

Test Categories:
1. Initialization Tests (2)
2. Content Normalization Tests (3)
3. Chunking Tests (4)
4. Processing Tests (3)
5. Stats Tests (1)

Total: 13 tests
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

from quirkllm.knowledge.document_processor import (
    DocumentProcessor,
    DocumentType,
    Document,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_embedder():
    """Create a mock embedding generator."""
    mock = Mock()
    mock.embed_code = Mock(return_value=np.random.rand(384).astype(np.float32))
    return mock


@pytest.fixture
def mock_store():
    """Create a mock LanceDB store."""
    mock = Mock()
    mock.add_document_chunks = Mock(return_value=5)
    return mock


@pytest.fixture
def processor_with_mocks(mock_embedder, mock_store):
    """Create processor with mocked dependencies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("quirkllm.knowledge.document_processor.EmbeddingGenerator") as mock_embed_cls:
            with patch("quirkllm.knowledge.document_processor.LanceDBStore") as mock_store_cls:
                mock_embed_cls.return_value = mock_embedder
                mock_store_cls.return_value = mock_store
                processor = DocumentProcessor(profile="survival", db_path=tmpdir)
                processor.embedder = mock_embedder
                processor.store = mock_store
                yield processor


@pytest.fixture
def real_processor():
    """Create a real processor with temp database (for integration-like tests)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("quirkllm.knowledge.document_processor.EmbeddingGenerator") as mock_embed_cls:
            mock_embedder = Mock()
            mock_embedder.embed_code = Mock(return_value=np.random.rand(384).astype(np.float32))
            mock_embed_cls.return_value = mock_embedder
            processor = DocumentProcessor(profile="survival", db_path=tmpdir)
            yield processor


# =============================================================================
# 1. Initialization Tests (2)
# =============================================================================


class TestProcessorInitialization:
    """Tests for DocumentProcessor initialization."""

    def test_processor_init_default_profile(self):
        """Test processor initializes with default profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("quirkllm.knowledge.document_processor.EmbeddingGenerator") as mock_embed:
                with patch("quirkllm.knowledge.document_processor.LanceDBStore"):
                    processor = DocumentProcessor(db_path=tmpdir)

                    assert processor.profile == "survival"
                    mock_embed.assert_called_once_with(profile="survival")

    def test_processor_init_custom_profile(self):
        """Test processor initializes with custom profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("quirkllm.knowledge.document_processor.EmbeddingGenerator") as mock_embed:
                with patch("quirkllm.knowledge.document_processor.LanceDBStore"):
                    processor = DocumentProcessor(profile="power", db_path=tmpdir)

                    assert processor.profile == "power"
                    mock_embed.assert_called_once_with(profile="power")


# =============================================================================
# 2. Content Normalization Tests (3)
# =============================================================================


class TestContentNormalization:
    """Tests for content normalization."""

    def test_normalize_content_whitespace(self, processor_with_mocks):
        """Test whitespace normalization."""
        content = "Hello   world\n\n\n\nMultiple    spaces"
        result = processor_with_mocks.normalize_content(content)

        assert "   " not in result  # No triple spaces
        assert "\n\n\n" not in result  # No more than 2 newlines

    def test_normalize_content_empty(self, processor_with_mocks):
        """Test normalizing empty content."""
        assert processor_with_mocks.normalize_content("") == ""
        assert processor_with_mocks.normalize_content(None) == ""

    def test_normalize_content_unicode(self, processor_with_mocks):
        """Test unicode normalization."""
        content = "Türkçe içerik: şçöğüı"
        result = processor_with_mocks.normalize_content(content)

        assert "Türkçe" in result
        assert "şçöğüı" in result


# =============================================================================
# 3. Chunking Tests (4)
# =============================================================================


class TestChunking:
    """Tests for content chunking."""

    def test_chunk_content_web_page(self, processor_with_mocks):
        """Test chunking web page content."""
        content = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = processor_with_mocks.chunk_content(
            content,
            DocumentType.WEB_PAGE,
            chunk_size=50,
            overlap=10
        )

        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_content_code(self, processor_with_mocks):
        """Test chunking code content."""
        content = "def foo():\n    pass\n\ndef bar():\n    return 42"
        chunks = processor_with_mocks.chunk_content(
            content,
            DocumentType.CODE,
            chunk_size=30,
            overlap=5
        )

        assert len(chunks) > 0

    def test_chunk_content_empty(self, processor_with_mocks):
        """Test chunking empty content returns empty list."""
        chunks = processor_with_mocks.chunk_content(
            "",
            DocumentType.WEB_PAGE,
            chunk_size=500,
            overlap=50
        )

        assert chunks == []

    def test_chunk_content_small_content(self, processor_with_mocks):
        """Test small content returns single chunk."""
        content = "Small text."
        chunks = processor_with_mocks.chunk_content(
            content,
            DocumentType.WEB_PAGE,
            chunk_size=500,
            overlap=50
        )

        assert len(chunks) == 1
        assert chunks[0] == content


# =============================================================================
# 4. Processing Tests (3)
# =============================================================================


class TestProcessing:
    """Tests for document processing."""

    def test_process_web_document(self, processor_with_mocks):
        """Test processing a web document."""
        chunks_added = processor_with_mocks.process_web_page(
            url="https://example.com/docs",
            content="This is documentation content.\n\nAnother paragraph.",
            title="Example Docs",
            metadata={"crawl_depth": 1}
        )

        # Should call store.add_document_chunks
        processor_with_mocks.store.add_document_chunks.assert_called()
        assert chunks_added > 0

    def test_process_pdf_document(self, processor_with_mocks):
        """Test processing a PDF document."""
        pages = [
            {
                "page_num": 1,
                "content": "Page 1 content",
                "tables": ["| A | B |"],
                "code_blocks": ["def foo(): pass"],
            },
            {
                "page_num": 2,
                "content": "Page 2 content",
                "tables": [],
                "code_blocks": [],
            },
        ]

        chunks_added = processor_with_mocks.process_pdf(
            file_path=Path("/path/to/doc.pdf"),
            pages=pages,
            metadata={"title": "Test PDF"}
        )

        assert chunks_added > 0

    def test_process_empty_content(self, processor_with_mocks):
        """Test processing empty content returns 0."""
        result = processor_with_mocks.process_web_page(
            url="https://empty.com",
            content="",
            title="Empty"
        )

        assert result == 0


# =============================================================================
# 5. Stats Tests (1)
# =============================================================================


class TestStats:
    """Tests for processing statistics."""

    def test_get_stats_after_processing(self, processor_with_mocks):
        """Test statistics after processing."""
        # Process some documents
        processor_with_mocks.process_web_page(
            url="https://test1.com",
            content="Test content 1",
            title="Test 1"
        )
        processor_with_mocks.process_web_page(
            url="https://test2.com",
            content="Test content 2",
            title="Test 2"
        )

        stats = processor_with_mocks.get_stats()

        assert "documents_processed" in stats
        assert "chunks_created" in stats
        assert "web_pages_processed" in stats
        assert stats["profile"] == "survival"


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_source_id_generation(self, processor_with_mocks):
        """Test source ID generation is deterministic."""
        url = "https://example.com/page"

        id1 = processor_with_mocks._generate_source_id(url)
        id2 = processor_with_mocks._generate_source_id(url)

        assert id1 == id2
        assert len(id1) == 16  # 16 hex chars

    def test_different_urls_different_ids(self, processor_with_mocks):
        """Test different URLs produce different IDs."""
        id1 = processor_with_mocks._generate_source_id("https://a.com")
        id2 = processor_with_mocks._generate_source_id("https://b.com")

        assert id1 != id2
