"""
Document Processor - Unified ingestion pipeline for RAG.

Processes web pages and PDFs into the vector database:
1. Normalize content (clean whitespace, fix encoding)
2. Chunk content (semantic boundaries)
3. Generate embeddings (profile-based: MiniLM/mpnet)
4. Store in LanceDB with metadata

Example:
    >>> processor = DocumentProcessor(profile="comfort")
    >>> chunks = processor.process_web_page(
    ...     url="https://docs.example.com",
    ...     content="Hello world",
    ...     title="Example Doc"
    ... )
    >>> print(f"Added {chunks} chunks to RAG")
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional

from quirkllm.rag.embeddings import EmbeddingGenerator
from quirkllm.rag.lancedb_store import LanceDBStore, DocumentChunk


class DocumentType(Enum):
    """Document type enumeration."""
    WEB_PAGE = "web"
    PDF = "pdf"
    MARKDOWN = "markdown"
    CODE = "code"


@dataclass
class Document:
    """
    Unified document representation.

    Attributes:
        content: Raw text content
        doc_type: Type of document
        source: URL or file path
        title: Document title
        metadata: Additional metadata
    """
    content: str
    doc_type: DocumentType
    source: str
    title: str
    metadata: Dict[str, Any]


class DocumentProcessor:
    """
    Processes documents for RAG ingestion.

    Pipeline:
    1. Normalize -> Clean text
    2. Chunk -> Split into semantic units
    3. Embed -> Generate vectors
    4. Store -> Save to LanceDB

    Attributes:
        profile: RAM profile for embedding model selection
        embedder: EmbeddingGenerator instance
        store: LanceDBStore instance
    """

    # Chunking parameters
    DEFAULT_CHUNK_SIZE = 500  # characters
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(
        self,
        profile: str = "survival",
        db_path: Optional[str] = None
    ) -> None:
        """
        Initialize processor.

        Args:
            profile: RAM profile for embedding model selection
            db_path: Optional custom database path
        """
        self.profile = profile.lower()
        self.embedder = EmbeddingGenerator(profile=self.profile)
        self.store = LanceDBStore(db_path=db_path)

        # Statistics
        self._documents_processed = 0
        self._chunks_created = 0
        self._web_pages_processed = 0
        self._pdfs_processed = 0

    def process(self, document: Document) -> int:
        """
        Process document and add to RAG.

        Args:
            document: Document to process

        Returns:
            Number of chunks added
        """
        # Normalize content
        normalized = self.normalize_content(document.content)

        if not normalized.strip():
            return 0

        # Chunk content
        chunks = self.chunk_content(
            normalized,
            document.doc_type,
            self.DEFAULT_CHUNK_SIZE,
            self.DEFAULT_CHUNK_OVERLAP
        )

        if not chunks:
            return 0

        # Generate source_id
        source_id = self._generate_source_id(document.source)

        # Create DocumentChunk objects
        doc_chunks = []
        for i, chunk_text in enumerate(chunks):
            # Generate embedding
            embedding = self.embedder.embed_code(chunk_text)

            chunk = DocumentChunk(
                id=f"{source_id}_{i}",
                content=chunk_text,
                embedding=embedding,
                source_id=source_id,
                source_type=document.doc_type.value,
                source_url=document.source,
                title=document.title,
                page_num=document.metadata.get("page_num", 0),
                chunk_index=i,
                total_chunks=len(chunks),
                metadata=document.metadata,
            )
            doc_chunks.append(chunk)

        # Store in database
        added = self.store.add_document_chunks(doc_chunks)

        # Update statistics
        self._documents_processed += 1
        self._chunks_created += added

        return added

    def process_web_page(
        self,
        url: str,
        content: str,
        title: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Process web page content.

        Convenience method for WebCrawler results.

        Args:
            url: Page URL
            content: Page content (markdown or text)
            title: Page title
            metadata: Optional additional metadata

        Returns:
            Number of chunks added
        """
        document = Document(
            content=content,
            doc_type=DocumentType.WEB_PAGE,
            source=url,
            title=title,
            metadata=metadata or {},
        )

        result = self.process(document)
        self._web_pages_processed += 1
        return result

    def process_pdf(
        self,
        file_path: Path,
        pages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Process PDF content.

        Convenience method for PDFParser results.

        Args:
            file_path: Path to PDF file
            pages: List of page dictionaries from PDFParser
            metadata: Optional additional metadata

        Returns:
            Total number of chunks added across all pages
        """
        total_chunks = 0
        file_path = Path(file_path)

        for page in pages:
            page_content = page.get("content", "")
            page_num = page.get("page_num", 0)

            # Include tables as markdown if present
            tables = page.get("tables", [])
            if tables:
                page_content += "\n\n" + "\n\n".join(tables)

            # Include code blocks if present
            code_blocks = page.get("code_blocks", [])
            if code_blocks:
                for code in code_blocks:
                    page_content += f"\n\n```\n{code}\n```"

            page_metadata = {
                "page_num": page_num,
                "file_name": file_path.name,
                **(metadata or {}),
            }

            document = Document(
                content=page_content,
                doc_type=DocumentType.PDF,
                source=str(file_path),
                title=metadata.get("title", file_path.stem) if metadata else file_path.stem,
                metadata=page_metadata,
            )

            total_chunks += self.process(document)

        self._pdfs_processed += 1
        return total_chunks

    def chunk_content(
        self,
        content: str,
        doc_type: DocumentType,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> List[str]:
        """
        Split content into semantic chunks.

        Uses different strategies based on doc_type:
        - WEB_PAGE: Paragraph boundaries
        - PDF: Paragraph boundaries
        - CODE: Line-based with overlap
        - MARKDOWN: Heading boundaries

        Args:
            content: Text content to chunk
            doc_type: Type of document
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if not content or not content.strip():
            return []

        # Strategy based on document type
        if doc_type in (DocumentType.WEB_PAGE, DocumentType.PDF, DocumentType.MARKDOWN):
            return self._chunk_by_paragraphs(content, chunk_size, overlap)
        elif doc_type == DocumentType.CODE:
            return self._chunk_by_lines(content, chunk_size, overlap)
        else:
            return self._chunk_by_characters(content, chunk_size, overlap)

    def _chunk_by_paragraphs(
        self,
        content: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Chunk content by paragraph boundaries."""
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from end of previous
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add remaining content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _chunk_by_lines(
        self,
        content: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Chunk content by line boundaries (for code)."""
        lines = content.split('\n')
        chunks = []
        current_chunk = ""

        for line in lines:
            if len(current_chunk) + len(line) + 1 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Overlap: keep last few lines
                overlap_lines = current_chunk.split('\n')[-3:] if overlap > 0 else []
                current_chunk = '\n'.join(overlap_lines) + '\n' + line if overlap_lines else line
            else:
                current_chunk += line + '\n'

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _chunk_by_characters(
        self,
        content: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Simple character-based chunking with overlap."""
        if len(content) <= chunk_size:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size

            # Try to break at word boundary
            if end < len(content):
                # Look for last space before end
                space_pos = content.rfind(' ', start, end)
                if space_pos > start + chunk_size // 2:
                    end = space_pos

            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap if overlap > 0 else end

        return chunks

    def normalize_content(self, content: str) -> str:
        """
        Clean and normalize text.

        - Fix encoding issues
        - Normalize whitespace
        - Remove control characters

        Args:
            content: Raw text content

        Returns:
            Cleaned text
        """
        if not content:
            return ""

        # Normalize unicode
        text = unicodedata.normalize('NFKC', content)

        # Remove control characters (except newlines and tabs)
        text = ''.join(
            char if char in '\n\t' or not unicodedata.category(char).startswith('C')
            else ' '
            for char in text
        )

        # Normalize whitespace
        # Replace multiple spaces with single space (but preserve newlines)
        lines = text.split('\n')
        lines = [' '.join(line.split()) for line in lines]
        text = '\n'.join(lines)

        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dict with documents_processed, chunks_created, etc.
        """
        return {
            "documents_processed": self._documents_processed,
            "chunks_created": self._chunks_created,
            "web_pages_processed": self._web_pages_processed,
            "pdfs_processed": self._pdfs_processed,
            "profile": self.profile,
        }

    @staticmethod
    def _generate_source_id(source: str) -> str:
        """Generate unique source ID from URL or path."""
        return hashlib.sha256(source.encode()).hexdigest()[:16]
