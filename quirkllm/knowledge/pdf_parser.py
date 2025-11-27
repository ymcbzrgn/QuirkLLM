"""
PDF Parser - Extract and convert PDF content for RAG ingestion.

Features:
- Page-by-page extraction
- Table detection and markdown formatting
- Code block detection (monospace fonts)
- Metadata extraction (title, author, pages)
- Large PDF streaming support
- Unicode/Turkish character support

Example:
    >>> parser = PDFParser("docs/manual.pdf")
    >>> pages = parser.parse()
    >>> for page in pages:
    ...     print(f"Page {page['page_num']}: {len(page['content'])} chars")
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pdfplumber
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console


class PDFParser:
    """
    Parses PDF files and extracts text content.

    The parser extracts text page by page, detects tables and converts
    them to markdown format, and can detect code blocks based on font analysis.

    Attributes:
        file_path: Path to PDF file
        metadata: Extracted PDF metadata
        results: List of parsed pages
    """

    # Common monospace fonts for code detection
    CODE_FONTS = [
        "courier",
        "consolas",
        "monaco",
        "menlo",
        "source code",
        "fira code",
        "jetbrains",
        "mono",
        "fixed",
        "terminal",
        "inconsolata",
        "dejavu sans mono",
        "liberation mono",
    ]

    def __init__(self, file_path: Union[str, Path]) -> None:
        """
        Initialize PDF parser.

        Args:
            file_path: Path to PDF file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a PDF
        """
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {file_path}")

        self.metadata: Dict[str, Any] = {}
        self.results: List[Dict[str, Any]] = []
        self.console = Console()

        # Statistics
        self._tables_found = 0
        self._code_blocks_found = 0

    def parse(self, show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Parse PDF and return page contents.

        Args:
            show_progress: Show Rich progress bar

        Returns:
            List of {page_num, content, tables, code_blocks, metadata} dicts

        Raises:
            RuntimeError: If PDF is encrypted or corrupted
        """
        # Reset state
        self.results.clear()
        self._tables_found = 0
        self._code_blocks_found = 0

        try:
            with pdfplumber.open(self.file_path) as pdf:
                # Extract metadata first
                self.metadata = self._extract_metadata(pdf)
                total_pages = len(pdf.pages)

                if show_progress:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
                        console=self.console,
                    ) as progress:
                        task = progress.add_task(
                            f"[cyan]Parsing {self.file_path.name}...",
                            total=total_pages,
                        )
                        for i, page in enumerate(pdf.pages):
                            page_data = self.extract_page(page, i + 1)
                            self.results.append(page_data)
                            progress.update(task, completed=i + 1)
                else:
                    for i, page in enumerate(pdf.pages):
                        page_data = self.extract_page(page, i + 1)
                        self.results.append(page_data)

        except Exception as e:
            error_msg = str(e).lower()
            if "encrypt" in error_msg or "password" in error_msg:
                raise RuntimeError(f"PDF is encrypted: {self.file_path}") from e
            if "damaged" in error_msg or "corrupt" in error_msg or "invalid" in error_msg:
                raise RuntimeError(f"PDF is corrupted: {self.file_path}") from e
            raise RuntimeError(f"Failed to parse PDF: {e}") from e

        return self.results

    def extract_page(self, page: Any, page_num: int) -> Dict[str, Any]:
        """
        Extract content from single page.

        Args:
            page: pdfplumber page object
            page_num: Page number (1-indexed)

        Returns:
            Dict with page_num, content, tables, code_blocks
        """
        # Extract text
        text = page.extract_text() or ""

        # Extract tables
        tables = self.extract_tables(page)
        self._tables_found += len(tables)

        # Detect code blocks
        code_blocks = self.detect_code_blocks(page)
        self._code_blocks_found += len(code_blocks)

        return {
            "page_num": page_num,
            "content": text.strip(),
            "tables": tables,
            "code_blocks": code_blocks,
            "metadata": {
                "width": page.width,
                "height": page.height,
                "char_count": len(text),
            },
        }

    def extract_tables(self, page: Any) -> List[str]:
        """
        Extract tables and convert to markdown.

        Args:
            page: pdfplumber page object

        Returns:
            List of markdown table strings
        """
        markdown_tables = []

        try:
            tables = page.extract_tables()
            for table in tables:
                if table and len(table) > 0:
                    md_table = self._table_to_markdown(table)
                    if md_table:
                        markdown_tables.append(md_table)
        except Exception:
            # Silently skip table extraction errors
            pass

        return markdown_tables

    def detect_code_blocks(self, page: Any) -> List[str]:
        """
        Detect code blocks by font analysis.

        Looks for monospace fonts (Courier, Consolas, etc.)
        and groups consecutive characters.

        Args:
            page: pdfplumber page object

        Returns:
            List of code block strings
        """
        code_blocks = []

        try:
            chars = page.chars
            if not chars:
                return []

            # Group characters by font
            code_chars = []
            for char in chars:
                font_name = char.get("fontname", "").lower()
                if self._is_code_font(font_name):
                    code_chars.append(char.get("text", ""))

            # If we found code characters, combine them
            if code_chars:
                code_text = "".join(code_chars).strip()
                if code_text and len(code_text) > 10:  # Minimum length for code block
                    code_blocks.append(code_text)

        except Exception:
            # Silently skip code detection errors
            pass

        return code_blocks

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get PDF metadata.

        If parse() hasn't been called yet, opens PDF to extract metadata.

        Returns:
            Dict with title, author, pages, created, etc.
        """
        if self.metadata:
            return self.metadata

        try:
            with pdfplumber.open(self.file_path) as pdf:
                self.metadata = self._extract_metadata(pdf)
        except Exception:
            self.metadata = {"pages": 0, "title": self.file_path.stem}

        return self.metadata

    def _extract_metadata(self, pdf: Any) -> Dict[str, Any]:
        """
        Extract metadata from open PDF.

        Args:
            pdf: Open pdfplumber PDF object

        Returns:
            Dict with metadata
        """
        info = pdf.metadata or {}

        return {
            "title": info.get("Title", self.file_path.stem),
            "author": info.get("Author", "Unknown"),
            "creator": info.get("Creator", ""),
            "producer": info.get("Producer", ""),
            "created": info.get("CreationDate", ""),
            "modified": info.get("ModDate", ""),
            "pages": len(pdf.pages),
            "file_name": self.file_path.name,
            "file_size": self.file_path.stat().st_size,
        }

    def _table_to_markdown(self, table: List[List[Optional[str]]]) -> str:
        """
        Convert table data to markdown format.

        Args:
            table: 2D list of cell values

        Returns:
            Markdown table string
        """
        if not table or not table[0]:
            return ""

        # Clean cells (replace None with empty string)
        cleaned = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned.append(cleaned_row)

        if not cleaned:
            return ""

        # Build markdown table
        lines = []

        # Header row
        header = cleaned[0]
        lines.append("| " + " | ".join(header) + " |")

        # Separator
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        # Data rows
        for row in cleaned[1:]:
            # Pad row if needed
            while len(row) < len(header):
                row.append("")
            lines.append("| " + " | ".join(row[: len(header)]) + " |")

        return "\n".join(lines)

    def _is_code_font(self, font_name: str) -> bool:
        """
        Check if font is a monospace/code font.

        Args:
            font_name: Font name from PDF

        Returns:
            True if font is likely a code font
        """
        if not font_name:
            return False

        font_lower = font_name.lower()
        return any(code_font in font_lower for code_font in self.CODE_FONTS)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get parsing statistics.

        Returns:
            Dict with pages_parsed, total_chars, tables_found, code_blocks_found
        """
        if not self.results:
            return {
                "pages_parsed": 0,
                "total_chars": 0,
                "tables_found": 0,
                "code_blocks_found": 0,
            }

        total_chars = sum(r["metadata"]["char_count"] for r in self.results)

        return {
            "pages_parsed": len(self.results),
            "total_chars": total_chars,
            "tables_found": self._tables_found,
            "code_blocks_found": self._code_blocks_found,
        }
