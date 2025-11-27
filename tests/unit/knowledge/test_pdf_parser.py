"""
Tests for PDFParser class.

Test Categories:
1. Initialization Tests (3)
2. Page Extraction Tests (3)
3. Table Detection Tests (3)
4. Code Block Detection Tests (3)
5. Metadata Tests (2)
6. Error Handling Tests (3)
7. Stats Tests (2)

Total: 19 tests
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from quirkllm.knowledge.pdf_parser import PDFParser


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_pdf_path():
    """Create a temporary file with .pdf extension."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        # Write minimal PDF content (not a valid PDF, but enough for path tests)
        f.write(b"%PDF-1.4\n%%EOF")
        return Path(f.name)


@pytest.fixture
def mock_page():
    """Create a mock PDF page."""
    page = Mock()
    page.extract_text.return_value = "This is sample text from page."
    page.extract_tables.return_value = []
    page.chars = []
    page.width = 612
    page.height = 792
    return page


@pytest.fixture
def mock_page_with_table():
    """Create a mock PDF page with a table."""
    page = Mock()
    page.extract_text.return_value = "Page with table"
    page.extract_tables.return_value = [
        [["Name", "Value"], ["Foo", "Bar"], ["Baz", "Qux"]]
    ]
    page.chars = []
    page.width = 612
    page.height = 792
    return page


@pytest.fixture
def mock_page_with_code():
    """Create a mock PDF page with code (monospace font)."""
    page = Mock()
    page.extract_text.return_value = "def hello(): print('world')"
    page.extract_tables.return_value = []
    page.chars = [
        {"text": "d", "fontname": "Courier"},
        {"text": "e", "fontname": "Courier"},
        {"text": "f", "fontname": "Courier"},
        {"text": " ", "fontname": "Courier"},
        {"text": "h", "fontname": "Courier"},
        {"text": "e", "fontname": "Courier"},
        {"text": "l", "fontname": "Courier"},
        {"text": "l", "fontname": "Courier"},
        {"text": "o", "fontname": "Courier"},
        {"text": "(", "fontname": "Courier"},
        {"text": ")", "fontname": "Courier"},
        {"text": ":", "fontname": "Courier"},
    ]
    page.width = 612
    page.height = 792
    return page


# =============================================================================
# 1. Initialization Tests (3)
# =============================================================================


class TestParserInitialization:
    """Tests for PDFParser initialization."""

    def test_parser_init_valid_pdf(self, temp_pdf_path):
        """Test parser initializes with valid PDF path."""
        parser = PDFParser(temp_pdf_path)

        assert parser.file_path == temp_pdf_path
        assert parser.metadata == {}
        assert parser.results == []

    def test_parser_init_file_not_found(self):
        """Test parser raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            PDFParser("/nonexistent/path/file.pdf")

        assert "PDF not found" in str(exc_info.value)

    def test_parser_init_not_pdf_extension(self, tmp_path):
        """Test parser raises ValueError for non-PDF file."""
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("Not a PDF")

        with pytest.raises(ValueError) as exc_info:
            PDFParser(txt_file)

        assert "Not a PDF file" in str(exc_info.value)

    def test_parser_init_accepts_string_path(self, temp_pdf_path):
        """Test parser accepts string path."""
        parser = PDFParser(str(temp_pdf_path))

        assert parser.file_path == temp_pdf_path


# =============================================================================
# 2. Page Extraction Tests (3)
# =============================================================================


class TestPageExtraction:
    """Tests for page extraction."""

    def test_extract_page_basic(self, temp_pdf_path, mock_page):
        """Test basic page extraction."""
        parser = PDFParser(temp_pdf_path)

        result = parser.extract_page(mock_page, 1)

        assert result["page_num"] == 1
        assert "sample text" in result["content"]
        assert "tables" in result
        assert "code_blocks" in result
        assert result["metadata"]["width"] == 612

    def test_parse_single_page_pdf(self, temp_pdf_path, mock_page):
        """Test parsing single page PDF."""
        parser = PDFParser(temp_pdf_path)

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {"Title": "Test Doc"}
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            results = parser.parse(show_progress=False)

        assert len(results) == 1
        assert results[0]["page_num"] == 1

    def test_parse_multi_page_pdf(self, temp_pdf_path, mock_page):
        """Test parsing multi-page PDF."""
        parser = PDFParser(temp_pdf_path)

        # Create 3 pages
        pages = [Mock(), Mock(), Mock()]
        for i, page in enumerate(pages):
            page.extract_text.return_value = f"Page {i+1} content"
            page.extract_tables.return_value = []
            page.chars = []
            page.width = 612
            page.height = 792

        mock_pdf = Mock()
        mock_pdf.pages = pages
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            results = parser.parse(show_progress=False)

        assert len(results) == 3
        assert results[0]["page_num"] == 1
        assert results[1]["page_num"] == 2
        assert results[2]["page_num"] == 3

    def test_parse_empty_page_handling(self, temp_pdf_path):
        """Test handling of empty pages."""
        parser = PDFParser(temp_pdf_path)

        empty_page = Mock()
        empty_page.extract_text.return_value = None  # Empty page
        empty_page.extract_tables.return_value = []
        empty_page.chars = []
        empty_page.width = 612
        empty_page.height = 792

        mock_pdf = Mock()
        mock_pdf.pages = [empty_page]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            results = parser.parse(show_progress=False)

        assert len(results) == 1
        assert results[0]["content"] == ""


# =============================================================================
# 3. Table Detection Tests (3)
# =============================================================================


class TestTableDetection:
    """Tests for table extraction and conversion."""

    def test_extract_tables_simple(self, temp_pdf_path, mock_page_with_table):
        """Test simple table extraction."""
        parser = PDFParser(temp_pdf_path)

        tables = parser.extract_tables(mock_page_with_table)

        assert len(tables) == 1
        assert "| Name | Value |" in tables[0]
        assert "| Foo | Bar |" in tables[0]

    def test_extract_tables_no_tables(self, temp_pdf_path, mock_page):
        """Test extraction when no tables present."""
        parser = PDFParser(temp_pdf_path)

        tables = parser.extract_tables(mock_page)

        assert tables == []

    def test_table_to_markdown_format(self, temp_pdf_path):
        """Test table to markdown conversion."""
        parser = PDFParser(temp_pdf_path)

        table = [["A", "B"], ["1", "2"], ["3", "4"]]
        md = parser._table_to_markdown(table)

        assert "| A | B |" in md
        assert "| --- | --- |" in md
        assert "| 1 | 2 |" in md
        assert "| 3 | 4 |" in md

    def test_table_to_markdown_with_none_cells(self, temp_pdf_path):
        """Test table conversion handles None cells."""
        parser = PDFParser(temp_pdf_path)

        table = [["A", None], ["1", "2"]]
        md = parser._table_to_markdown(table)

        assert "| A |  |" in md  # None becomes empty string


# =============================================================================
# 4. Code Block Detection Tests (3)
# =============================================================================


class TestCodeBlockDetection:
    """Tests for code block detection."""

    def test_detect_code_courier_font(self, temp_pdf_path, mock_page_with_code):
        """Test code detection with Courier font."""
        parser = PDFParser(temp_pdf_path)

        code_blocks = parser.detect_code_blocks(mock_page_with_code)

        assert len(code_blocks) == 1
        assert "def hello():" in code_blocks[0]

    def test_detect_code_various_fonts(self, temp_pdf_path):
        """Test code detection with various monospace fonts."""
        parser = PDFParser(temp_pdf_path)

        # Test font detection
        assert parser._is_code_font("Courier")
        assert parser._is_code_font("Consolas")
        assert parser._is_code_font("Monaco")
        assert parser._is_code_font("Menlo-Regular")
        assert parser._is_code_font("Source Code Pro")
        assert parser._is_code_font("Fira Code Medium")

    def test_detect_code_no_code_blocks(self, temp_pdf_path, mock_page):
        """Test detection returns empty for regular text."""
        parser = PDFParser(temp_pdf_path)

        code_blocks = parser.detect_code_blocks(mock_page)

        assert code_blocks == []

    def test_is_code_font_non_code_fonts(self, temp_pdf_path):
        """Test non-code fonts are not detected."""
        parser = PDFParser(temp_pdf_path)

        assert not parser._is_code_font("Arial")
        assert not parser._is_code_font("Times New Roman")
        assert not parser._is_code_font("Helvetica")
        assert not parser._is_code_font("")


# =============================================================================
# 5. Metadata Tests (2)
# =============================================================================


class TestMetadata:
    """Tests for metadata extraction."""

    def test_get_metadata_full(self, temp_pdf_path, mock_page):
        """Test full metadata extraction."""
        parser = PDFParser(temp_pdf_path)

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {
            "Title": "Test Document",
            "Author": "Test Author",
            "Creator": "Test Creator",
            "CreationDate": "2024-01-01",
        }
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            parser.parse(show_progress=False)

        metadata = parser.get_metadata()

        assert metadata["title"] == "Test Document"
        assert metadata["author"] == "Test Author"
        assert metadata["pages"] == 1

    def test_get_metadata_minimal(self, temp_pdf_path, mock_page):
        """Test metadata with minimal info."""
        parser = PDFParser(temp_pdf_path)

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {}  # No metadata
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            parser.parse(show_progress=False)

        metadata = parser.get_metadata()

        # Should use file name as title fallback
        assert "title" in metadata
        assert metadata["author"] == "Unknown"


# =============================================================================
# 6. Error Handling Tests (3)
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_parse_encrypted_pdf_error(self, temp_pdf_path):
        """Test error on encrypted PDF."""
        parser = PDFParser(temp_pdf_path)

        with patch("pdfplumber.open") as mock_open:
            mock_open.side_effect = Exception("PDF is encrypted")

            with pytest.raises(RuntimeError) as exc_info:
                parser.parse(show_progress=False)

            assert "encrypted" in str(exc_info.value).lower()

    def test_parse_corrupted_pdf_error(self, temp_pdf_path):
        """Test error on corrupted PDF."""
        parser = PDFParser(temp_pdf_path)

        with patch("pdfplumber.open") as mock_open:
            mock_open.side_effect = Exception("File is damaged or corrupted")

            with pytest.raises(RuntimeError) as exc_info:
                parser.parse(show_progress=False)

            assert "corrupted" in str(exc_info.value).lower()

    def test_parse_generic_error(self, temp_pdf_path):
        """Test generic error handling."""
        parser = PDFParser(temp_pdf_path)

        with patch("pdfplumber.open") as mock_open:
            mock_open.side_effect = Exception("Some other error")

            with pytest.raises(RuntimeError) as exc_info:
                parser.parse(show_progress=False)

            assert "Failed to parse" in str(exc_info.value)


# =============================================================================
# 7. Stats Tests (2)
# =============================================================================


class TestStats:
    """Tests for statistics."""

    def test_get_stats_empty(self, temp_pdf_path):
        """Test stats when no pages parsed."""
        parser = PDFParser(temp_pdf_path)

        stats = parser.get_stats()

        assert stats["pages_parsed"] == 0
        assert stats["total_chars"] == 0
        assert stats["tables_found"] == 0
        assert stats["code_blocks_found"] == 0

    def test_get_stats_after_parse(self, temp_pdf_path, mock_page_with_table):
        """Test stats after parsing."""
        parser = PDFParser(temp_pdf_path)

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page_with_table, mock_page_with_table]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            parser.parse(show_progress=False)

        stats = parser.get_stats()

        assert stats["pages_parsed"] == 2
        assert stats["tables_found"] == 2  # One table per page


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_table_extraction_error_handling(self, temp_pdf_path):
        """Test table extraction handles errors gracefully."""
        parser = PDFParser(temp_pdf_path)

        page = Mock()
        page.extract_tables.side_effect = Exception("Table extraction failed")

        tables = parser.extract_tables(page)

        assert tables == []  # Should return empty, not raise

    def test_code_detection_error_handling(self, temp_pdf_path):
        """Test code detection handles errors gracefully."""
        parser = PDFParser(temp_pdf_path)

        page = Mock()
        page.chars = None  # Would cause error if not handled

        code_blocks = parser.detect_code_blocks(page)

        assert code_blocks == []

    def test_empty_table_handling(self, temp_pdf_path):
        """Test empty table conversion."""
        parser = PDFParser(temp_pdf_path)

        assert parser._table_to_markdown([]) == ""
        assert parser._table_to_markdown([[]]) == ""

    def test_parse_with_progress_bar(self, temp_pdf_path, mock_page):
        """Test parsing with progress bar enabled."""
        parser = PDFParser(temp_pdf_path)

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        # Suppress Rich console output
        parser.console = Mock()

        with patch("pdfplumber.open", return_value=mock_pdf):
            with patch("quirkllm.knowledge.pdf_parser.Progress"):
                results = parser.parse(show_progress=True)

        assert len(results) == 1

    def test_get_metadata_without_parse(self, temp_pdf_path, mock_page):
        """Test get_metadata when parse() hasn't been called."""
        parser = PDFParser(temp_pdf_path)

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {"Title": "Direct Metadata"}
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            metadata = parser.get_metadata()

        assert metadata["title"] == "Direct Metadata"

    def test_get_metadata_on_error(self, temp_pdf_path):
        """Test get_metadata fallback on error."""
        parser = PDFParser(temp_pdf_path)

        with patch("pdfplumber.open") as mock_open:
            mock_open.side_effect = Exception("Cannot open")
            metadata = parser.get_metadata()

        assert metadata["pages"] == 0
        assert "title" in metadata

    def test_code_detection_short_text_ignored(self, temp_pdf_path):
        """Test code blocks shorter than 10 chars are ignored."""
        parser = PDFParser(temp_pdf_path)

        page = Mock()
        page.chars = [
            {"text": "x", "fontname": "Courier"},
            {"text": "=", "fontname": "Courier"},
            {"text": "1", "fontname": "Courier"},
        ]

        code_blocks = parser.detect_code_blocks(page)

        assert code_blocks == []  # Too short
