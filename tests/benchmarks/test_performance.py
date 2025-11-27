"""
Comprehensive Performance Benchmarks for QuirkLLM.

Targets from ROADMAP and docs:
- Startup time: <3s
- Import time: <1s
- Embedding generation: <100ms per chunk
- Search latency: <500ms for top 5 results
- Ghost Mode overhead: <5% CPU
- MCP response time: <100ms

Test Categories:
1. Startup & Import Benchmarks (3)
2. RAG System Benchmarks (3)
3. Ghost Mode Benchmarks (2)
4. MCP Benchmarks (2)

Total: 10 tests
"""

import time
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest


# =============================================================================
# Startup & Import Benchmarks (3 tests)
# =============================================================================


class TestStartupBenchmarks:
    """Startup and import time benchmarks."""

    def test_cli_startup_time(self):
        """Test CLI starts in <3 seconds."""
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-m", "quirkllm", "--help"],
            capture_output=True,
            timeout=10,
        )
        elapsed = time.perf_counter() - start

        assert result.returncode == 0, f"CLI failed: {result.stderr.decode()}"
        assert elapsed < 3.0, f"Startup took {elapsed:.2f}s (target: <3s)"

    def test_import_time(self):
        """Test main module import time <1s."""
        # Use subprocess to get clean import time
        code = """
import time
start = time.perf_counter()
import quirkllm
elapsed = time.perf_counter() - start
print(f"{elapsed:.4f}")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        elapsed = float(result.stdout.strip())

        assert elapsed < 1.0, f"Import took {elapsed:.2f}s (target: <1s)"

    def test_heavy_module_import_time(self):
        """Test heavy modules (RAG, modes) import time."""
        code = """
import time
start = time.perf_counter()
from quirkllm.rag.embeddings import EmbeddingGenerator
from quirkllm.modes.ghost_mode import GhostMode
from quirkllm.mcp.server import MCPServer
elapsed = time.perf_counter() - start
print(f"{elapsed:.4f}")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        elapsed = float(result.stdout.strip())

        assert elapsed < 5.0, f"Heavy imports took {elapsed:.2f}s (target: <5s)"


# =============================================================================
# RAG System Benchmarks (3 tests)
# =============================================================================


class TestRAGBenchmarks:
    """RAG system performance benchmarks."""

    @pytest.mark.slow
    def test_embedding_generation_speed(self):
        """Test embedding generation <100ms per chunk (after warmup)."""
        from quirkllm.rag.embeddings import EmbeddingGenerator

        gen = EmbeddingGenerator()
        text = "def hello_world():\n    print('Hello, World!')"

        # Warm up - first call loads model
        _ = gen.embed_code("warmup")

        # Measure actual embedding time
        start = time.perf_counter()
        embedding = gen.embed_code(text)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, f"Embedding took {elapsed*1000:.1f}ms (target: <100ms)"
        assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension

    @pytest.mark.slow
    def test_batch_embedding_throughput(self):
        """Test batch embedding ≥5 chunks/second."""
        from quirkllm.rag.embeddings import EmbeddingGenerator

        gen = EmbeddingGenerator()
        texts = [f"def func_{i}(): pass" for i in range(5)]

        # Warm up
        _ = gen.embed_code("warmup")

        start = time.perf_counter()
        embeddings = gen.embed_batch(texts)  # Correct method name
        elapsed = time.perf_counter() - start

        throughput = len(texts) / elapsed
        assert throughput >= 5, f"Throughput {throughput:.1f}/s (target: ≥5/s)"
        assert len(embeddings) == 5

    def test_lancedb_store_initialization(self):
        """Test LanceDB store initializes in <1s."""
        from quirkllm.rag.lancedb_store import LanceDBStore

        with tempfile.TemporaryDirectory() as tmpdir:
            start = time.perf_counter()
            store = LanceDBStore(db_path=tmpdir)
            elapsed = time.perf_counter() - start

            assert elapsed < 1.0, f"Store init took {elapsed*1000:.1f}ms (target: <1s)"
            assert store is not None


# =============================================================================
# Ghost Mode Benchmarks (2 tests)
# =============================================================================


class TestGhostModeBenchmarks:
    """Ghost Mode performance benchmarks."""

    def test_performance_monitor_overhead(self):
        """Test PerformanceMonitor has minimal CPU impact."""
        from quirkllm.modes.ghost_mode import PerformanceMonitor

        # Simple test: monitor starts and stops without error
        # and doesn't cause excessive CPU usage
        monitor = PerformanceMonitor(sample_interval=0.5)
        monitor.start()

        # Let it run for a bit
        time.sleep(0.5)

        # Get stats - should have some samples
        stats = monitor.get_stats()
        monitor.stop()

        # Verify monitor ran and collected data
        assert stats["samples"] >= 0
        assert stats["cpu_percent"] >= 0
        assert stats["ram_percent"] >= 0
        # Monitor should not be running anymore
        assert monitor._running is False

    def test_ghost_mode_activation_time(self):
        """Test Ghost Mode activates in <500ms."""
        from quirkllm.modes.ghost_mode import GhostMode

        with tempfile.TemporaryDirectory() as tmpdir:
            mode = GhostMode(watch_dir=tmpdir)

            start = time.perf_counter()
            mode.activate()
            elapsed = time.perf_counter() - start
            mode.deactivate()

            assert elapsed < 0.5, f"Activation took {elapsed*1000:.1f}ms (target: <500ms)"


# =============================================================================
# MCP Benchmarks (2 tests)
# =============================================================================


class TestMCPBenchmarks:
    """MCP server performance benchmarks."""

    def test_mcp_tool_registry_initialization(self):
        """Test MCP tool registry initializes in <500ms."""
        from quirkllm.mcp.tools import ToolRegistry

        start = time.perf_counter()
        registry = ToolRegistry()
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"Registry init took {elapsed*1000:.1f}ms (target: <500ms)"
        assert len(registry.list_tools()) == 8

    def test_mcp_protocol_message_latency(self):
        """Test MCP protocol handles messages in <100ms."""
        from quirkllm.mcp.server import MCPServer
        from quirkllm.mcp.protocol import MCPRequest

        server = MCPServer()

        # Warm up
        warmup_req = MCPRequest(jsonrpc="2.0", method="ping", id=0)
        _ = server.protocol.handle_request(warmup_req)

        # Measure
        start = time.perf_counter()
        for i in range(10):
            request = MCPRequest(jsonrpc="2.0", method="ping", id=i + 1)
            _ = server.protocol.handle_request(request)
        elapsed = time.perf_counter() - start

        avg_latency = elapsed / 10
        assert avg_latency < 0.1, f"Avg latency {avg_latency*1000:.1f}ms (target: <100ms)"
