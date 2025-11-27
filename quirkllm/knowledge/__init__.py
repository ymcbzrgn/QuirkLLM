"""
Knowledge module - Web and PDF document ingestion for RAG.

This module provides tools for crawling web documentation and parsing PDFs,
converting them to markdown, and ingesting into the RAG vector database.

Components:
- WebCrawler: Crawl and convert web pages to markdown ✅
- PDFParser: Parse PDF files and extract text ✅
- DocumentProcessor: Unified processing pipeline ✅
- KnowledgeManager: High-level knowledge management ✅
- IngestionPipeline: Unified interface for REPL commands ✅
"""

from quirkllm.knowledge.web_crawler import WebCrawler
from quirkllm.knowledge.pdf_parser import PDFParser
from quirkllm.knowledge.document_processor import (
    DocumentProcessor,
    DocumentType,
    Document,
)
from quirkllm.knowledge.knowledge_manager import (
    KnowledgeManager,
    KnowledgeSource,
)
from quirkllm.knowledge.ingestion_pipeline import IngestionPipeline

__all__ = [
    "WebCrawler",
    "PDFParser",
    "DocumentProcessor",
    "DocumentType",
    "Document",
    "KnowledgeManager",
    "KnowledgeSource",
    "IngestionPipeline",
]
