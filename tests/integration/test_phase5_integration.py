"""
Phase 5 Integration Tests - MCP, Knowledge Eater, Ghost Mode.

Test Categories:
1. MCP Integration Tests (5)
2. Knowledge Integration Tests (5)
3. Cross-Component Tests (5)

Total: 15 tests
"""

import tempfile
from pathlib import Path
import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_pdf(temp_dir):
    """Create a temporary PDF file (minimal valid PDF)."""
    pdf_path = temp_dir / "test.pdf"
    # Minimal PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000206 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
299
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return pdf_path


# =============================================================================
# 1. MCP Integration Tests (5)
# =============================================================================


class TestMCPIntegration:
    """Integration tests for MCP components."""

    def test_mcp_server_initialization(self):
        """Test MCP server can be initialized with handlers."""
        from quirkllm.mcp.server import MCPServer

        server = MCPServer()
        assert server is not None
        assert hasattr(server, "protocol")
        assert hasattr(server, "config")
        # Verify handlers are registered
        assert "ping" in server.protocol.handlers
        assert "initialize" in server.protocol.handlers
        assert "tools/list" in server.protocol.handlers

    def test_mcp_tool_registry_has_all_tools(self):
        """Test all 8 MCP tools are registered."""
        from quirkllm.mcp.tools import ToolRegistry

        registry = ToolRegistry()
        tools = registry.list_tools()

        assert len(tools) == 8, f"Expected 8 tools, got {len(tools)}"

        tool_names = [t["name"] for t in tools]
        expected_tools = [
            "analyze_project",
            "search_code",
            "get_context",
            "read_file",
            "list_files",
            "write_file",
            "execute_command",
            "search_documents",
        ]
        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"

    def test_mcp_server_handles_ping(self):
        """Test MCP server handles ping method."""
        from quirkllm.mcp.server import MCPServer
        from quirkllm.mcp.protocol import MCPRequest

        server = MCPServer()
        request = MCPRequest(jsonrpc="2.0", method="ping", id=1)
        response = server.protocol.handle_request(request)

        assert response.id == 1
        assert response.error is None

    def test_mcp_server_handles_initialize(self):
        """Test MCP server handles initialize method."""
        from quirkllm.mcp.server import MCPServer
        from quirkllm.mcp.protocol import MCPRequest

        server = MCPServer()
        request = MCPRequest(
            jsonrpc="2.0",
            method="initialize",
            id=1,
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )
        response = server.protocol.handle_request(request)

        assert response.id == 1
        assert response.error is None
        assert "protocolVersion" in response.result

    def test_mcp_server_handles_tools_list(self):
        """Test MCP server handles tools/list method with tools registered."""
        from quirkllm.mcp.server import MCPServer
        from quirkllm.mcp.tools import ToolRegistry
        from quirkllm.mcp.protocol import MCPRequest

        server = MCPServer()
        # Set tools registry
        server.set_tools(ToolRegistry())

        # First initialize
        init_request = MCPRequest(
            jsonrpc="2.0",
            method="initialize",
            id=1,
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )
        server.protocol.handle_request(init_request)

        # Then list tools
        request = MCPRequest(jsonrpc="2.0", method="tools/list", id=2)
        response = server.protocol.handle_request(request)

        assert response.id == 2
        assert response.error is None
        assert "tools" in response.result
        assert len(response.result["tools"]) == 8


# =============================================================================
# 2. Knowledge Integration Tests (5)
# =============================================================================


class TestKnowledgeIntegration:
    """Integration tests for Knowledge Eater components."""

    def test_ingestion_pipeline_initialization(self):
        """Test IngestionPipeline can be initialized."""
        from quirkllm.knowledge import IngestionPipeline

        pipeline = IngestionPipeline()
        assert pipeline.processor is not None
        assert pipeline.manager is not None

    def test_knowledge_manager_persistence(self, temp_dir):
        """Test KnowledgeManager saves and loads sources."""
        from quirkllm.knowledge import KnowledgeManager

        # Create manager and add source
        manager1 = KnowledgeManager(base_dir=temp_dir)
        source = KnowledgeManager.create_source(
            source_path="https://test.com",
            source_type="web",
            title="Test",
            chunk_count=5,
        )
        manager1.add_source(source)

        # Create new manager (same dir) and verify persistence
        manager2 = KnowledgeManager(base_dir=temp_dir)
        sources = manager2.list_sources()

        assert len(sources) == 1
        assert sources[0].source_path == "https://test.com"
        assert sources[0].chunk_count == 5

    def test_document_processor_chunks_text(self):
        """Test DocumentProcessor creates chunks from text."""
        from quirkllm.knowledge import DocumentProcessor, DocumentType, Document

        processor = DocumentProcessor(profile="survival")
        chunks = processor.chunk_content(
            "Paragraph one.\n\nParagraph two.\n\nParagraph three.",
            DocumentType.WEB_PAGE,
            chunk_size=50,
            overlap=10,
        )

        assert len(chunks) > 0
        assert all(len(chunk) <= 100 for chunk in chunks)  # Allow some flexibility

    def test_web_crawler_initialization(self):
        """Test WebCrawler can be initialized."""
        from quirkllm.knowledge import WebCrawler

        crawler = WebCrawler(
            base_url="https://example.com",
            max_depth=2,
            max_pages=10,
        )
        assert crawler.base_url == "https://example.com"
        assert crawler.max_depth == 2
        assert crawler.max_pages == 10

    def test_pdf_parser_validates_file(self, temp_dir):
        """Test PDFParser validates file existence."""
        from quirkllm.knowledge import PDFParser

        # Non-existent file
        with pytest.raises(FileNotFoundError):
            PDFParser(str(temp_dir / "nonexistent.pdf"))


# =============================================================================
# 3. Cross-Component Tests (5)
# =============================================================================


class TestCrossComponentIntegration:
    """Integration tests for component interactions."""

    def test_ghost_mode_with_performance_monitor(self, temp_dir):
        """Test Ghost Mode works with PerformanceMonitor."""
        from quirkllm.modes.ghost_mode import GhostMode

        mode = GhostMode(watch_dir=str(temp_dir))
        mode.activate()

        # Check performance monitor started
        assert mode.perf_monitor is not None
        assert mode.perf_monitor._running is True

        # Check session stats include perf data
        stats = mode.get_session_stats()
        assert "perf_cpu_percent" in stats
        assert "perf_ram_percent" in stats
        assert "perf_throttle_count" in stats

        mode.deactivate()
        assert mode.perf_monitor._running is False

    def test_ingestion_pipeline_list_sources_format(self, temp_dir):
        """Test IngestionPipeline.list_sources returns REPL-expected format."""
        from quirkllm.knowledge import IngestionPipeline, KnowledgeManager

        # Create pipeline with custom manager
        pipeline = IngestionPipeline()
        pipeline.manager = KnowledgeManager(base_dir=temp_dir)

        # Add a source
        source = KnowledgeManager.create_source(
            source_path="https://docs.test.com",
            source_type="web",
            title="Test Docs",
            chunk_count=10,
        )
        pipeline.manager.add_source(source)

        # Get sources via pipeline (REPL format)
        sources = pipeline.list_sources()

        assert len(sources) == 1
        # Verify REPL-expected keys
        assert "id" in sources[0]
        assert "type" in sources[0]
        assert "source" in sources[0]
        assert "chunks" in sources[0]
        assert "added" in sources[0]

        # Verify values
        assert sources[0]["type"] == "web"
        assert sources[0]["source"] == "https://docs.test.com"
        assert sources[0]["chunks"] == 10

    def test_ingestion_pipeline_get_stats_format(self, temp_dir):
        """Test IngestionPipeline.get_stats returns REPL-expected format."""
        from quirkllm.knowledge import IngestionPipeline, KnowledgeManager

        pipeline = IngestionPipeline()
        pipeline.manager = KnowledgeManager(base_dir=temp_dir)

        # Add sources
        for i, stype in enumerate(["web", "web", "pdf"]):
            source = KnowledgeManager.create_source(
                source_path=f"https://test{i}.com" if stype == "web" else f"/path/test{i}.pdf",
                source_type=stype,
                title=f"Test {i}",
                chunk_count=5,
            )
            pipeline.manager.add_source(source)

        # Get stats via pipeline (REPL format)
        stats = pipeline.get_stats()

        # Verify REPL-expected keys
        assert "total_sources" in stats
        assert "total_chunks" in stats
        assert "web_sources" in stats
        assert "pdf_sources" in stats
        assert "storage_mb" in stats

        # Verify values
        assert stats["total_sources"] == 3
        assert stats["total_chunks"] == 15
        assert stats["web_sources"] == 2
        assert stats["pdf_sources"] == 1

    def test_mcp_search_documents_tool_exists(self):
        """Test MCP search_documents tool is available in registry."""
        from quirkllm.mcp.tools import ToolRegistry

        registry = ToolRegistry()

        # Find search_documents tool
        tools = registry.list_tools()
        search_tool = next((t for t in tools if t["name"] == "search_documents"), None)

        assert search_tool is not None
        assert search_tool["name"] == "search_documents"
        # Verify it has required input schema
        assert "inputSchema" in search_tool
        assert search_tool["inputSchema"]["type"] == "object"

    def test_mode_registry_has_ghost_mode(self):
        """Test Ghost Mode is registered in mode registry."""
        from quirkllm.modes import ModeType, get_registry

        registry = get_registry()
        mode = registry.create_mode(ModeType.GHOST)

        assert mode is not None
        assert mode.mode_type == ModeType.GHOST
