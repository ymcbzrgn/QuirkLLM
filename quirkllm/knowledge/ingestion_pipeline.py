"""
Ingestion Pipeline - Unified interface for Knowledge Eater.

Orchestrates WebCrawler, PDFParser, DocumentProcessor, and KnowledgeManager
to provide the API expected by REPL /learn and /knowledge commands.

Example:
    >>> pipeline = IngestionPipeline()
    >>> result = pipeline.ingest_url("https://docs.example.com", max_depth=2)
    >>> print(f"Added {result['chunks']} chunks")
    >>> sources = pipeline.list_sources()
    >>> stats = pipeline.get_stats()
"""

from pathlib import Path
from typing import Any, Dict, List

from quirkllm.knowledge.web_crawler import WebCrawler
from quirkllm.knowledge.pdf_parser import PDFParser
from quirkllm.knowledge.document_processor import DocumentProcessor
from quirkllm.knowledge.knowledge_manager import KnowledgeManager


class IngestionPipeline:
    """
    Unified pipeline for ingesting web and PDF content into RAG.

    This class orchestrates all Knowledge Eater components and provides
    the API expected by REPL commands:
    - /learn --url <url> [--depth <n>]
    - /learn --pdf <path>
    - /knowledge list
    - /knowledge stats
    - /knowledge forget <id>

    Attributes:
        processor: DocumentProcessor instance for chunking/embedding
        manager: KnowledgeManager instance for source tracking
        profile: RAM profile for embedding model selection
    """

    def __init__(self, profile: str = "survival") -> None:
        """
        Initialize pipeline with all components.

        Args:
            profile: RAM profile for embedding model selection
                     (survival, comfort, power, beast)
        """
        self.profile = profile
        self.processor = DocumentProcessor(profile=profile)
        self.manager = KnowledgeManager()

    def ingest_url(self, url: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Crawl URL and ingest content into RAG.

        Creates a WebCrawler for the URL, crawls pages, processes them
        through DocumentProcessor, and registers the source.

        Args:
            url: Web URL to crawl (starting point)
            max_depth: Maximum crawl depth (minimum 2)

        Returns:
            Dict with keys:
                success: bool - Whether ingestion succeeded
                documents: int - Number of pages crawled
                chunks: int - Number of chunks created
                error: str - Error message if failed
        """
        try:
            # 1. Create crawler for this URL
            crawler = WebCrawler(
                base_url=url,
                max_depth=max(2, max_depth),
            )

            # 2. Crawl website
            pages = crawler.crawl(show_progress=False)
            if not pages:
                return {"success": False, "error": "No pages crawled"}

            # 3. Process each page through DocumentProcessor
            total_chunks = 0
            for page in pages:
                chunks = self.processor.process_web_page(
                    url=page.get("url", url),
                    content=page.get("content", ""),
                    title=page.get("title", url),
                )
                total_chunks += chunks

            # 4. Register source in KnowledgeManager
            source = KnowledgeManager.create_source(
                source_path=url,
                source_type="web",
                title=pages[0].get("title", url) if pages else url,
                chunk_count=total_chunks,
            )
            self.manager.add_source(source)

            return {
                "success": True,
                "documents": len(pages),
                "chunks": total_chunks,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def ingest_pdf(self, path: str) -> Dict[str, Any]:
        """
        Parse PDF and ingest content into RAG.

        Creates a PDFParser for the file, extracts pages, processes them
        through DocumentProcessor, and registers the source.

        Args:
            path: Path to PDF file (absolute or relative)

        Returns:
            Dict with keys:
                success: bool - Whether ingestion succeeded
                pages: int - Number of pages extracted
                chunks: int - Number of chunks created
                error: str - Error message if failed
        """
        try:
            pdf_path = Path(path).expanduser().resolve()

            if not pdf_path.exists():
                return {"success": False, "error": f"File not found: {path}"}

            if pdf_path.suffix.lower() != ".pdf":
                return {"success": False, "error": f"Not a PDF file: {path}"}

            # 1. Create parser for this PDF
            parser = PDFParser(str(pdf_path))

            # 2. Parse PDF
            pages = parser.parse(show_progress=False)
            if not pages:
                return {"success": False, "error": "No pages extracted"}

            # 3. Process through DocumentProcessor
            total_chunks = self.processor.process_pdf(
                file_path=pdf_path,
                pages=pages,
                metadata={"title": pdf_path.stem},
            )

            # 4. Register source
            source = KnowledgeManager.create_source(
                source_path=str(pdf_path),
                source_type="pdf",
                title=pdf_path.stem,
                chunk_count=total_chunks,
            )
            self.manager.add_source(source)

            return {
                "success": True,
                "pages": len(pages),
                "chunks": total_chunks,
            }
        except FileNotFoundError as e:
            return {"success": False, "error": str(e)}
        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_sources(self) -> List[Dict[str, Any]]:
        """
        List all registered knowledge sources.

        Returns:
            List of dicts with keys:
                id: str - Source ID (16-char hash)
                type: str - "web" or "pdf"
                source: str - URL or file path
                chunks: int - Number of chunks
                added: str - ISO timestamp

        Note:
            Format matches what REPL _knowledge_list expects.
        """
        sources = self.manager.list_sources()
        # Transform KnowledgeSource objects to REPL-expected dict format
        return [
            {
                "id": src.source_id,
                "type": src.source_type,
                "source": src.source_path,
                "chunks": src.chunk_count,
                "added": src.ingested_at,
            }
            for src in sources
        ]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Dict with keys:
                total_sources: int - Total number of sources
                total_chunks: int - Total chunks across all sources
                web_sources: int - Number of web sources
                pdf_sources: int - Number of PDF sources
                storage_mb: float - Approximate storage size

        Note:
            Format matches what REPL _knowledge_stats expects.
        """
        stats = self.manager.get_stats()
        by_type = stats.get("by_type", {})

        return {
            "total_sources": stats.get("total_sources", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "web_sources": by_type.get("web", 0),
            "pdf_sources": by_type.get("pdf", 0),
            "storage_mb": stats.get("store_stats", {}).get("size_mb", 0.0),
        }

    def remove_source(self, source_id: str) -> Dict[str, Any]:
        """
        Remove a knowledge source and its chunks.

        Supports partial ID matching (first 8 characters).

        Args:
            source_id: Full or partial source ID

        Returns:
            Dict with keys:
                success: bool - Whether removal succeeded
                chunks_deleted: int - Number of chunks removed
                error: str - Error message if failed
        """
        try:
            # Find full source_id if partial provided
            sources = self.manager.list_sources()
            full_id = None
            for src in sources:
                if src.source_id.startswith(source_id):
                    full_id = src.source_id
                    break

            if not full_id:
                return {"success": False, "error": f"Source not found: {source_id}"}

            # Get chunk count before deletion
            source = self.manager.get_source(full_id)
            chunks_to_delete = source.chunk_count if source else 0

            # Remove source (also deletes chunks from store via forget_source)
            success = self.manager.forget_source(full_id)

            if success:
                return {
                    "success": True,
                    "chunks_deleted": chunks_to_delete,
                }
            else:
                return {"success": False, "error": "Failed to remove source"}
        except Exception as e:
            return {"success": False, "error": str(e)}
