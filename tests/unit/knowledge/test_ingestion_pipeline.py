"""
Tests for IngestionPipeline - Unified Knowledge Eater interface.

Test Categories:
1. Initialization Tests (2)
2. URL Ingestion Tests (4)
3. PDF Ingestion Tests (4)
4. Source Management Tests (4)

Total: 14 tests
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from quirkllm.knowledge.ingestion_pipeline import IngestionPipeline
from quirkllm.knowledge.knowledge_manager import KnowledgeSource


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_knowledge_source():
    """Create a mock KnowledgeSource."""
    return KnowledgeSource(
        source_id="abc123def456",
        source_type="web",
        source_path="https://docs.example.com",
        title="Example Docs",
        chunk_count=10,
        ingested_at="2025-11-27T12:00:00",
        metadata={},
    )


# =============================================================================
# 1. Initialization Tests (2)
# =============================================================================


class TestIngestionPipelineInit:
    """Tests for IngestionPipeline initialization."""

    @patch("quirkllm.knowledge.ingestion_pipeline.DocumentProcessor")
    @patch("quirkllm.knowledge.ingestion_pipeline.KnowledgeManager")
    def test_init_default_profile(self, mock_manager, mock_processor):
        """Test initialization with default profile."""
        pipeline = IngestionPipeline()

        assert pipeline.profile == "survival"
        mock_processor.assert_called_once_with(profile="survival")
        mock_manager.assert_called_once()

    @patch("quirkllm.knowledge.ingestion_pipeline.DocumentProcessor")
    @patch("quirkllm.knowledge.ingestion_pipeline.KnowledgeManager")
    def test_init_custom_profile(self, mock_manager, mock_processor):
        """Test initialization with custom profile."""
        pipeline = IngestionPipeline(profile="power")

        assert pipeline.profile == "power"
        mock_processor.assert_called_once_with(profile="power")


# =============================================================================
# 2. URL Ingestion Tests (4)
# =============================================================================


class TestIngestUrl:
    """Tests for URL ingestion."""

    @patch("quirkllm.knowledge.ingestion_pipeline.WebCrawler")
    @patch("quirkllm.knowledge.ingestion_pipeline.KnowledgeManager")
    @patch("quirkllm.knowledge.ingestion_pipeline.DocumentProcessor")
    def test_ingest_url_success(self, mock_processor, mock_manager, mock_crawler_cls):
        """Test successful URL ingestion."""
        # Setup mocks
        mock_crawler = Mock()
        mock_crawler.crawl.return_value = [
            {"url": "https://docs.example.com", "title": "Home", "content": "Hello"},
            {"url": "https://docs.example.com/api", "title": "API", "content": "World"},
        ]
        mock_crawler_cls.return_value = mock_crawler

        mock_processor_instance = Mock()
        mock_processor_instance.process_web_page.return_value = 5
        mock_processor.return_value = mock_processor_instance

        mock_manager_instance = Mock()
        mock_manager.return_value = mock_manager_instance
        mock_manager.create_source = Mock(return_value=Mock())

        # Execute
        pipeline = IngestionPipeline()
        result = pipeline.ingest_url("https://docs.example.com", max_depth=3)

        # Verify
        assert result["success"] is True
        assert result["documents"] == 2
        assert result["chunks"] == 10  # 5 chunks per page * 2 pages

        mock_crawler_cls.assert_called_once_with(
            base_url="https://docs.example.com",
            max_depth=3,
        )
        mock_crawler.crawl.assert_called_once_with(show_progress=False)
        assert mock_processor_instance.process_web_page.call_count == 2

    @patch("quirkllm.knowledge.ingestion_pipeline.WebCrawler")
    @patch("quirkllm.knowledge.ingestion_pipeline.KnowledgeManager")
    @patch("quirkllm.knowledge.ingestion_pipeline.DocumentProcessor")
    def test_ingest_url_no_pages(self, mock_processor, mock_manager, mock_crawler_cls):
        """Test URL ingestion when no pages are crawled."""
        mock_crawler = Mock()
        mock_crawler.crawl.return_value = []
        mock_crawler_cls.return_value = mock_crawler

        pipeline = IngestionPipeline()
        result = pipeline.ingest_url("https://empty.example.com")

        assert result["success"] is False
        assert "No pages crawled" in result["error"]

    @patch("quirkllm.knowledge.ingestion_pipeline.WebCrawler")
    @patch("quirkllm.knowledge.ingestion_pipeline.KnowledgeManager")
    @patch("quirkllm.knowledge.ingestion_pipeline.DocumentProcessor")
    def test_ingest_url_exception(self, mock_processor, mock_manager, mock_crawler_cls):
        """Test URL ingestion handles exceptions."""
        mock_crawler_cls.side_effect = Exception("Network error")

        pipeline = IngestionPipeline()
        result = pipeline.ingest_url("https://failing.example.com")

        assert result["success"] is False
        assert "Network error" in result["error"]

    @patch("quirkllm.knowledge.ingestion_pipeline.WebCrawler")
    @patch("quirkllm.knowledge.ingestion_pipeline.KnowledgeManager")
    @patch("quirkllm.knowledge.ingestion_pipeline.DocumentProcessor")
    def test_ingest_url_min_depth(self, mock_processor, mock_manager, mock_crawler_cls):
        """Test URL ingestion enforces minimum depth of 2."""
        mock_crawler = Mock()
        mock_crawler.crawl.return_value = [
            {"url": "https://docs.example.com", "title": "Home", "content": "Test"}
        ]
        mock_crawler_cls.return_value = mock_crawler

        mock_processor_instance = Mock()
        mock_processor_instance.process_web_page.return_value = 1
        mock_processor.return_value = mock_processor_instance

        mock_manager_instance = Mock()
        mock_manager.return_value = mock_manager_instance
        mock_manager.create_source = Mock(return_value=Mock())

        pipeline = IngestionPipeline()
        result = pipeline.ingest_url("https://docs.example.com", max_depth=1)

        # Should enforce minimum depth of 2
        mock_crawler_cls.assert_called_once_with(
            base_url="https://docs.example.com",
            max_depth=2,
        )


# =============================================================================
# 3. PDF Ingestion Tests (4)
# =============================================================================


class TestIngestPdf:
    """Tests for PDF ingestion."""

    @patch("quirkllm.knowledge.ingestion_pipeline.PDFParser")
    @patch("quirkllm.knowledge.ingestion_pipeline.KnowledgeManager")
    @patch("quirkllm.knowledge.ingestion_pipeline.DocumentProcessor")
    def test_ingest_pdf_success(self, mock_processor, mock_manager, mock_parser_cls, temp_dir):
        """Test successful PDF ingestion."""
        # Create a fake PDF file
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake pdf content")

        # Setup mocks
        mock_parser = Mock()
        mock_parser.parse.return_value = [
            {"page_num": 1, "content": "Page 1 content"},
            {"page_num": 2, "content": "Page 2 content"},
        ]
        mock_parser_cls.return_value = mock_parser

        mock_processor_instance = Mock()
        mock_processor_instance.process_pdf.return_value = 8
        mock_processor.return_value = mock_processor_instance

        mock_manager_instance = Mock()
        mock_manager.return_value = mock_manager_instance
        mock_manager.create_source = Mock(return_value=Mock())

        # Execute
        pipeline = IngestionPipeline()
        result = pipeline.ingest_pdf(str(pdf_file))

        # Verify
        assert result["success"] is True
        assert result["pages"] == 2
        assert result["chunks"] == 8

        mock_parser_cls.assert_called_once()
        mock_parser.parse.assert_called_once_with(show_progress=False)
        mock_processor_instance.process_pdf.assert_called_once()

    def test_ingest_pdf_file_not_found(self):
        """Test PDF ingestion with missing file."""
        pipeline = IngestionPipeline()
        result = pipeline.ingest_pdf("/nonexistent/file.pdf")

        assert result["success"] is False
        assert "File not found" in result["error"]

    def test_ingest_pdf_not_pdf_extension(self, temp_dir):
        """Test PDF ingestion with non-PDF file."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("Not a PDF")

        pipeline = IngestionPipeline()
        result = pipeline.ingest_pdf(str(txt_file))

        assert result["success"] is False
        assert "Not a PDF file" in result["error"]

    @patch("quirkllm.knowledge.ingestion_pipeline.PDFParser")
    @patch("quirkllm.knowledge.ingestion_pipeline.KnowledgeManager")
    @patch("quirkllm.knowledge.ingestion_pipeline.DocumentProcessor")
    def test_ingest_pdf_no_pages(self, mock_processor, mock_manager, mock_parser_cls, temp_dir):
        """Test PDF ingestion when no pages extracted."""
        pdf_file = temp_dir / "empty.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 empty")

        mock_parser = Mock()
        mock_parser.parse.return_value = []
        mock_parser_cls.return_value = mock_parser

        pipeline = IngestionPipeline()
        result = pipeline.ingest_pdf(str(pdf_file))

        assert result["success"] is False
        assert "No pages extracted" in result["error"]


# =============================================================================
# 4. Source Management Tests (4)
# =============================================================================


class TestSourceManagement:
    """Tests for source listing, stats, and removal."""

    @patch("quirkllm.knowledge.ingestion_pipeline.KnowledgeManager")
    @patch("quirkllm.knowledge.ingestion_pipeline.DocumentProcessor")
    def test_list_sources(self, mock_processor, mock_manager, mock_knowledge_source):
        """Test listing knowledge sources."""
        mock_manager_instance = Mock()
        mock_manager_instance.list_sources.return_value = [mock_knowledge_source]
        mock_manager.return_value = mock_manager_instance

        pipeline = IngestionPipeline()
        sources = pipeline.list_sources()

        assert len(sources) == 1
        assert sources[0]["id"] == "abc123def456"
        assert sources[0]["type"] == "web"
        assert sources[0]["source"] == "https://docs.example.com"
        assert sources[0]["chunks"] == 10
        assert sources[0]["added"] == "2025-11-27T12:00:00"

    @patch("quirkllm.knowledge.ingestion_pipeline.KnowledgeManager")
    @patch("quirkllm.knowledge.ingestion_pipeline.DocumentProcessor")
    def test_list_sources_empty(self, mock_processor, mock_manager):
        """Test listing when no sources exist."""
        mock_manager_instance = Mock()
        mock_manager_instance.list_sources.return_value = []
        mock_manager.return_value = mock_manager_instance

        pipeline = IngestionPipeline()
        sources = pipeline.list_sources()

        assert sources == []

    @patch("quirkllm.knowledge.ingestion_pipeline.KnowledgeManager")
    @patch("quirkllm.knowledge.ingestion_pipeline.DocumentProcessor")
    def test_get_stats(self, mock_processor, mock_manager):
        """Test getting knowledge statistics."""
        mock_manager_instance = Mock()
        mock_manager_instance.get_stats.return_value = {
            "total_sources": 5,
            "total_chunks": 100,
            "by_type": {"web": 3, "pdf": 2},
            "store_stats": {"size_mb": 12.5},
        }
        mock_manager.return_value = mock_manager_instance

        pipeline = IngestionPipeline()
        stats = pipeline.get_stats()

        assert stats["total_sources"] == 5
        assert stats["total_chunks"] == 100
        assert stats["web_sources"] == 3
        assert stats["pdf_sources"] == 2
        assert stats["storage_mb"] == 12.5

    @patch("quirkllm.knowledge.ingestion_pipeline.KnowledgeManager")
    @patch("quirkllm.knowledge.ingestion_pipeline.DocumentProcessor")
    def test_remove_source_success(self, mock_processor, mock_manager, mock_knowledge_source):
        """Test successful source removal."""
        mock_manager_instance = Mock()
        mock_manager_instance.list_sources.return_value = [mock_knowledge_source]
        mock_manager_instance.get_source.return_value = mock_knowledge_source
        mock_manager_instance.forget_source.return_value = True
        mock_manager.return_value = mock_manager_instance

        pipeline = IngestionPipeline()
        result = pipeline.remove_source("abc123")  # Partial ID

        assert result["success"] is True
        assert result["chunks_deleted"] == 10
        mock_manager_instance.forget_source.assert_called_once_with("abc123def456")

    @patch("quirkllm.knowledge.ingestion_pipeline.KnowledgeManager")
    @patch("quirkllm.knowledge.ingestion_pipeline.DocumentProcessor")
    def test_remove_source_not_found(self, mock_processor, mock_manager):
        """Test removing non-existent source."""
        mock_manager_instance = Mock()
        mock_manager_instance.list_sources.return_value = []
        mock_manager.return_value = mock_manager_instance

        pipeline = IngestionPipeline()
        result = pipeline.remove_source("nonexistent")

        assert result["success"] is False
        assert "Source not found" in result["error"]
