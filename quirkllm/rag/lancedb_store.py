"""
LanceDB Vector Store for QuirkLLM (Phase 3.4)

Vector database for semantic code search using LanceDB.
Stores code chunks with embeddings and metadata for efficient retrieval.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import lancedb
import pyarrow as pa
import numpy as np


# Default database location
DEFAULT_DB_PATH = Path.home() / ".quirkllm" / "rag"

# Schema for code chunks table
CODE_CHUNKS_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("content", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), 384)),  # 384-dim for all-MiniLM-L6-v2
    pa.field("file_path", pa.string()),
    pa.field("language", pa.string()),
    pa.field("framework", pa.string()),
    pa.field("project_root", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("total_chunks", pa.int32()),
    pa.field("start_line", pa.int32()),
    pa.field("end_line", pa.int32()),
    pa.field("metadata_json", pa.string()),  # Additional metadata as JSON
])

# Schema for documents table (web pages, PDFs)
DOCUMENTS_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("content", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), 384)),  # 384-dim for all-MiniLM-L6-v2
    pa.field("source_id", pa.string()),  # Reference to KnowledgeSource
    pa.field("source_type", pa.string()),  # "web" or "pdf"
    pa.field("source_url", pa.string()),  # URL or file path
    pa.field("title", pa.string()),
    pa.field("page_num", pa.int32()),  # Page number (PDF) or 0 (web)
    pa.field("chunk_index", pa.int32()),
    pa.field("total_chunks", pa.int32()),
    pa.field("metadata_json", pa.string()),
])


@dataclass
class CodeChunk:
    """Code chunk with metadata."""
    id: str
    content: str
    embedding: np.ndarray
    file_path: str
    language: str
    framework: str = ""
    project_root: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    start_line: int = 1
    end_line: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LanceDB insertion."""
        data = {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding,
            "file_path": self.file_path,
            "language": self.language,
            "framework": self.framework,
            "project_root": self.project_root,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "metadata_json": json.dumps(self.metadata),
        }
        return data


@dataclass
class SearchResult:
    """Search result from vector database."""
    id: str
    content: str
    file_path: str
    language: str
    framework: str
    score: float
    start_line: int
    end_line: int
    metadata: Dict[str, Any]


@dataclass
class DocumentChunk:
    """Document chunk for web/PDF content."""
    id: str
    content: str
    embedding: np.ndarray
    source_id: str
    source_type: str  # "web" or "pdf"
    source_url: str
    title: str
    page_num: int = 0
    chunk_index: int = 0
    total_chunks: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LanceDB insertion."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "title": self.title,
            "page_num": self.page_num,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "metadata_json": json.dumps(self.metadata),
        }


@dataclass
class DocumentSearchResult:
    """Search result from documents table."""
    id: str
    content: str
    source_id: str
    source_type: str
    source_url: str
    title: str
    score: float
    page_num: int
    chunk_index: int
    metadata: Dict[str, Any]


class LanceDBStore:
    """
    LanceDB vector store for code chunks.
    
    Provides semantic search capabilities for code using vector embeddings.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize LanceDB store.
        
        Args:
            db_path: Path to database directory (default: ~/.quirkllm/rag/)
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.db = lancedb.connect(str(self.db_path))
        self._init_tables()
    
    def _init_tables(self) -> None:
        """Initialize database tables if they don't exist."""
        # Create code_chunks table if it doesn't exist
        if "code_chunks" not in self.db.table_names():
            # Create empty table with schema
            self.db.create_table("code_chunks", schema=CODE_CHUNKS_SCHEMA, mode="create")

        # Create documents table if it doesn't exist (Phase 5.3)
        if "documents" not in self.db.table_names():
            self.db.create_table("documents", schema=DOCUMENTS_SCHEMA, mode="create")
    
    def add_code_chunk(self, chunk: CodeChunk) -> bool:
        """
        Add a single code chunk to the database.
        
        Args:
            chunk: CodeChunk to add
        
        Returns:
            True if successful, False otherwise
        """
        try:
            table = self.db.open_table("code_chunks")
            data = [chunk.to_dict()]
            table.add(data)
            return True
        except Exception as e:
            print(f"Error adding code chunk: {e}")
            return False
    
    def add_documents(self, chunks: List[CodeChunk]) -> int:
        """
        Add multiple code chunks in batch.
        
        Args:
            chunks: List of CodeChunk objects
        
        Returns:
            Number of chunks successfully added
        """
        if not chunks:
            return 0
        
        try:
            table = self.db.open_table("code_chunks")
            data = [chunk.to_dict() for chunk in chunks]
            table.add(data)
            return len(chunks)
        except Exception as e:
            print(f"Error adding documents: {e}")
            return 0
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Semantic search for code chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_conditions: Optional metadata filters (e.g., {"language": "python"})
        
        Returns:
            List of SearchResult objects, sorted by relevance
        """
        try:
            table = self.db.open_table("code_chunks")
            
            # Convert embedding to list
            query_vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            # Build search query
            search_query = table.search(query_vector).limit(k)
            
            # Apply filters if provided
            if filter_conditions:
                for key, value in filter_conditions.items():
                    search_query = search_query.where(f"{key} = '{value}'")
            
            # Execute search
            results = search_query.to_list()
            
            # Convert to SearchResult objects
            search_results = []
            for result in results:
                search_results.append(SearchResult(
                    id=result["id"],
                    content=result["content"],
                    file_path=result["file_path"],
                    language=result["language"],
                    framework=result["framework"],
                    score=float(result.get("_distance", 0.0)),  # LanceDB provides distance
                    start_line=result["start_line"],
                    end_line=result["end_line"],
                    metadata=json.loads(result.get("metadata_json", "{}")),
                ))
            
            return search_results
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def delete_by_project(self, project_root: str) -> int:
        """
        Delete all chunks belonging to a project.
        
        Args:
            project_root: Root path of the project
        
        Returns:
            Number of chunks deleted
        """
        try:
            table = self.db.open_table("code_chunks")
            
            # Get count before deletion
            before_count = table.count_rows()
            
            # Delete rows matching project_root
            table.delete(f"project_root = '{project_root}'")
            
            # Get count after deletion
            after_count = table.count_rows()
            
            return before_count - after_count
        except Exception as e:
            print(f"Error deleting by project: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics (total_chunks, by_language, by_framework, etc.)
        """
        try:
            table = self.db.open_table("code_chunks")
            total = table.count_rows()
            
            # Get all data for statistics (limit for large databases)
            data = table.to_pandas()
            
            stats = {
                "total_chunks": total,
                "by_language": {},
                "by_framework": {},
                "by_project": {},
            }
            
            if not data.empty:
                # Count by language
                stats["by_language"] = data["language"].value_counts().to_dict()
                
                # Count by framework
                framework_counts = data["framework"].value_counts().to_dict()
                # Remove empty strings
                stats["by_framework"] = {k: v for k, v in framework_counts.items() if k}
                
                # Count by project
                stats["by_project"] = data["project_root"].value_counts().to_dict()
            
            return stats
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                "total_chunks": 0,
                "by_language": {},
                "by_framework": {},
                "by_project": {},
            }
    
    def clear_all(self) -> bool:
        """
        Clear all data from the database (use with caution).
        
        Returns:
            True if successful
        """
        try:
            # Drop and recreate table
            self.db.drop_table("code_chunks")
            self._init_tables()
            return True
        except Exception as e:
            print(f"Error clearing database: {e}")
            return False
    
    def get_by_file(self, file_path: str) -> List[SearchResult]:
        """
        Get all chunks for a specific file.
        
        Args:
            file_path: Path to the file
        
        Returns:
            List of SearchResult objects
        """
        try:
            table = self.db.open_table("code_chunks")
            results = table.search().where(f"file_path = '{file_path}'").to_list()
            
            search_results = []
            for result in results:
                search_results.append(SearchResult(
                    id=result["id"],
                    content=result["content"],
                    file_path=result["file_path"],
                    language=result["language"],
                    framework=result["framework"],
                    score=0.0,
                    start_line=result["start_line"],
                    end_line=result["end_line"],
                    metadata=json.loads(result.get("metadata_json", "{}")),
                ))
            
            return search_results
        except Exception as e:
            print(f"Error getting by file: {e}")
            return []

    # =========================================================================
    # Document Methods (Phase 5.3 - Knowledge Eater)
    # =========================================================================

    def add_document_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Add document chunks (web pages, PDFs) to the database.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Number of chunks successfully added
        """
        if not chunks:
            return 0

        try:
            table = self.db.open_table("documents")
            data = [chunk.to_dict() for chunk in chunks]
            table.add(data)
            return len(chunks)
        except Exception as e:
            print(f"Error adding document chunks: {e}")
            return 0

    def search_documents(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        source_type: Optional[str] = None
    ) -> List[DocumentSearchResult]:
        """
        Semantic search for document chunks.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            source_type: Optional filter by "web" or "pdf"

        Returns:
            List of DocumentSearchResult objects, sorted by relevance
        """
        try:
            table = self.db.open_table("documents")

            # Convert embedding to list
            query_vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

            # Build search query
            search_query = table.search(query_vector).limit(k)

            # Apply source_type filter if provided
            if source_type:
                search_query = search_query.where(f"source_type = '{source_type}'")

            # Execute search
            results = search_query.to_list()

            # Convert to DocumentSearchResult objects
            search_results = []
            for result in results:
                search_results.append(DocumentSearchResult(
                    id=result["id"],
                    content=result["content"],
                    source_id=result["source_id"],
                    source_type=result["source_type"],
                    source_url=result["source_url"],
                    title=result["title"],
                    score=float(result.get("_distance", 0.0)),
                    page_num=result["page_num"],
                    chunk_index=result["chunk_index"],
                    metadata=json.loads(result.get("metadata_json", "{}")),
                ))

            return search_results
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []

    def delete_by_source_id(self, source_id: str) -> int:
        """
        Delete all document chunks belonging to a source.

        Used by KnowledgeManager.forget_source() to remove ingested content.

        Args:
            source_id: Source identifier (hash of URL/path)

        Returns:
            Number of chunks deleted
        """
        try:
            table = self.db.open_table("documents")

            # Get count before deletion
            before_count = table.count_rows()

            # Delete rows matching source_id
            table.delete(f"source_id = '{source_id}'")

            # Get count after deletion
            after_count = table.count_rows()

            return before_count - after_count
        except Exception as e:
            print(f"Error deleting by source_id: {e}")
            return 0

    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics for documents table.

        Returns:
            Dictionary with total_documents, by_type, by_source, etc.
        """
        try:
            table = self.db.open_table("documents")
            total = table.count_rows()

            stats = {
                "total_chunks": total,
                "by_type": {},
                "by_source": {},
            }

            if total > 0:
                data = table.to_pandas()

                # Count by source_type
                stats["by_type"] = data["source_type"].value_counts().to_dict()

                # Count by source_id
                stats["by_source"] = data["source_id"].value_counts().to_dict()

            return stats
        except Exception as e:
            print(f"Error getting document stats: {e}")
            return {
                "total_chunks": 0,
                "by_type": {},
                "by_source": {},
            }
