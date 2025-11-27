"""
Knowledge Manager - High-level knowledge source management.

Tracks all ingested sources (URLs, PDFs) and provides:
- Source listing and statistics
- Source removal (forget)
- Re-indexing capability

Example:
    >>> manager = KnowledgeManager()
    >>> sources = manager.list_sources()
    >>> for src in sources:
    ...     print(f"{src.title}: {src.chunk_count} chunks")
    >>> manager.forget_source("abc123")  # Remove source
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from quirkllm.rag.lancedb_store import LanceDBStore


@dataclass
class KnowledgeSource:
    """
    Represents an ingested knowledge source.

    Attributes:
        source_id: Unique identifier (hash of URL/path)
        source_type: "web" or "pdf"
        source_path: URL or file path
        title: Document/page title
        chunk_count: Number of chunks in RAG
        ingested_at: Timestamp string (ISO format)
        metadata: Additional info
    """
    source_id: str
    source_type: str
    source_path: str
    title: str
    chunk_count: int
    ingested_at: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeSource":
        """Create KnowledgeSource from dictionary."""
        return cls(**data)


class KnowledgeManager:
    """
    High-level knowledge management.

    Provides commands:
    - /knowledge list - Show learned sources
    - /knowledge forget <source> - Remove source
    - /knowledge stats - Show statistics
    - /knowledge reindex - Rebuild index

    Attributes:
        base_dir: Configuration directory
        store: LanceDB store instance
    """

    # Storage file name
    SOURCES_FILE = "knowledge_sources.json"

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        db_path: Optional[str] = None
    ) -> None:
        """
        Initialize manager.

        Args:
            base_dir: QuirkLLM config directory (~/.quirkllm)
            db_path: Optional custom database path
        """
        self.base_dir = Path(base_dir) if base_dir else Path.home() / ".quirkllm"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.sources_file = self.base_dir / self.SOURCES_FILE
        self.store = LanceDBStore(db_path=db_path)

        # Internal storage
        self._sources: Dict[str, KnowledgeSource] = {}

        # Load existing sources
        self._load_sources()

    def add_source(self, source: KnowledgeSource) -> None:
        """
        Register a new knowledge source.

        Called by DocumentProcessor after ingestion.

        Args:
            source: KnowledgeSource to register
        """
        self._sources[source.source_id] = source
        self._save_sources()

    def list_sources(self) -> List[KnowledgeSource]:
        """
        List all registered sources.

        Returns:
            List of KnowledgeSource objects
        """
        return list(self._sources.values())

    def get_source(self, source_id: str) -> Optional[KnowledgeSource]:
        """
        Get source by ID.

        Args:
            source_id: Source identifier

        Returns:
            KnowledgeSource or None if not found
        """
        return self._sources.get(source_id)

    def forget_source(self, source_id: str) -> bool:
        """
        Remove a knowledge source.

        Also removes associated chunks from RAG.

        Args:
            source_id: Source to forget

        Returns:
            True if removed, False if not found
        """
        if source_id not in self._sources:
            return False

        # Remove chunks from RAG
        self.store.delete_by_source_id(source_id)

        # Remove from tracking
        del self._sources[source_id]
        self._save_sources()

        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Dict with total_sources, total_chunks, by_type, etc.
        """
        total_chunks = sum(s.chunk_count for s in self._sources.values())

        by_type: Dict[str, int] = {}
        for source in self._sources.values():
            by_type[source.source_type] = by_type.get(source.source_type, 0) + 1

        # Get document stats from store
        doc_stats = self.store.get_document_stats()

        return {
            "total_sources": len(self._sources),
            "total_chunks": total_chunks,
            "by_type": by_type,
            "store_stats": doc_stats,
        }

    def reindex(self) -> int:
        """
        Rebuild entire RAG index.

        This is a placeholder - full reindexing requires re-crawling/re-parsing.
        For now, it returns the current chunk count.

        Returns:
            Number of chunks currently indexed
        """
        # Note: Full reindex would require access to original content
        # This would be implemented by re-crawling URLs / re-parsing PDFs
        # For now, we just return the current count
        return sum(s.chunk_count for s in self._sources.values())

    def _load_sources(self) -> None:
        """Load sources from JSON file."""
        if not self.sources_file.exists():
            self._sources = {}
            return

        try:
            with open(self.sources_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._sources = {
                source_id: KnowledgeSource.from_dict(source_data)
                for source_id, source_data in data.items()
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # If file is corrupted, start fresh
            print(f"Warning: Could not load knowledge sources: {e}")
            self._sources = {}

    def _save_sources(self) -> None:
        """Save sources to JSON file."""
        data = {
            source_id: source.to_dict()
            for source_id, source in self._sources.items()
        }

        with open(self.sources_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def generate_source_id(source_path: str) -> str:
        """
        Generate unique ID from source path.

        Args:
            source_path: URL or file path

        Returns:
            16-character hex string
        """
        return hashlib.sha256(source_path.encode()).hexdigest()[:16]

    @staticmethod
    def create_source(
        source_path: str,
        source_type: str,
        title: str,
        chunk_count: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeSource:
        """
        Factory method to create a KnowledgeSource.

        Args:
            source_path: URL or file path
            source_type: "web" or "pdf"
            title: Document title
            chunk_count: Number of chunks ingested
            metadata: Optional additional metadata

        Returns:
            KnowledgeSource instance
        """
        return KnowledgeSource(
            source_id=KnowledgeManager.generate_source_id(source_path),
            source_type=source_type,
            source_path=source_path,
            title=title,
            chunk_count=chunk_count,
            ingested_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )
