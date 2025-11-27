"""
Tests for KnowledgeManager class.

Test Categories:
1. Initialization Tests (2)
2. Source Management Tests (4)
3. Stats Tests (2)
4. Persistence Tests (2)

Total: 10 tests
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from quirkllm.knowledge.knowledge_manager import (
    KnowledgeManager,
    KnowledgeSource,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_manager():
    """Create a manager with temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("quirkllm.knowledge.knowledge_manager.LanceDBStore"):
            manager = KnowledgeManager(base_dir=Path(tmpdir))
            yield manager


@pytest.fixture
def sample_source():
    """Create a sample knowledge source."""
    return KnowledgeSource(
        source_id="abc123",
        source_type="web",
        source_path="https://example.com/docs",
        title="Example Documentation",
        chunk_count=10,
        ingested_at="2024-01-01T12:00:00",
        metadata={"depth": 2},
    )


# =============================================================================
# 1. Initialization Tests (2)
# =============================================================================


class TestManagerInitialization:
    """Tests for KnowledgeManager initialization."""

    def test_manager_init_creates_directory(self):
        """Test manager creates base directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_knowledge_dir"

            with patch("quirkllm.knowledge.knowledge_manager.LanceDBStore"):
                manager = KnowledgeManager(base_dir=new_dir)

            assert new_dir.exists()

    def test_manager_init_loads_existing_sources(self):
        """Test manager loads existing sources on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a sources file
            sources_file = Path(tmpdir) / "knowledge_sources.json"
            sources_file.write_text('{"src1": {"source_id": "src1", "source_type": "web", "source_path": "https://test.com", "title": "Test", "chunk_count": 5, "ingested_at": "2024-01-01", "metadata": {}}}')

            with patch("quirkllm.knowledge.knowledge_manager.LanceDBStore"):
                manager = KnowledgeManager(base_dir=Path(tmpdir))

            assert len(manager.list_sources()) == 1


# =============================================================================
# 2. Source Management Tests (4)
# =============================================================================


class TestSourceManagement:
    """Tests for source CRUD operations."""

    def test_manager_add_source(self, temp_manager, sample_source):
        """Test adding a knowledge source."""
        temp_manager.add_source(sample_source)

        sources = temp_manager.list_sources()
        assert len(sources) == 1
        assert sources[0].source_id == "abc123"

    def test_manager_list_sources_empty(self, temp_manager):
        """Test listing sources when empty."""
        sources = temp_manager.list_sources()

        assert sources == []

    def test_manager_get_source_found(self, temp_manager, sample_source):
        """Test getting an existing source."""
        temp_manager.add_source(sample_source)

        found = temp_manager.get_source("abc123")

        assert found is not None
        assert found.title == "Example Documentation"

    def test_manager_get_source_not_found(self, temp_manager):
        """Test getting a non-existent source."""
        found = temp_manager.get_source("nonexistent")

        assert found is None

    def test_manager_forget_source(self, temp_manager, sample_source):
        """Test forgetting a source."""
        temp_manager.add_source(sample_source)

        result = temp_manager.forget_source("abc123")

        assert result is True
        assert temp_manager.get_source("abc123") is None

    def test_manager_forget_source_not_found(self, temp_manager):
        """Test forgetting a non-existent source."""
        result = temp_manager.forget_source("nonexistent")

        assert result is False


# =============================================================================
# 3. Stats Tests (2)
# =============================================================================


class TestStats:
    """Tests for statistics."""

    def test_manager_get_stats_empty(self, temp_manager):
        """Test stats when no sources."""
        stats = temp_manager.get_stats()

        assert stats["total_sources"] == 0
        assert stats["total_chunks"] == 0

    def test_manager_get_stats_with_sources(self, temp_manager):
        """Test stats with multiple sources."""
        web_source = KnowledgeSource(
            source_id="web1",
            source_type="web",
            source_path="https://web.com",
            title="Web Source",
            chunk_count=10,
            ingested_at="2024-01-01",
            metadata={},
        )
        pdf_source = KnowledgeSource(
            source_id="pdf1",
            source_type="pdf",
            source_path="/path/to/doc.pdf",
            title="PDF Source",
            chunk_count=20,
            ingested_at="2024-01-02",
            metadata={},
        )

        temp_manager.add_source(web_source)
        temp_manager.add_source(pdf_source)

        stats = temp_manager.get_stats()

        assert stats["total_sources"] == 2
        assert stats["total_chunks"] == 30
        assert stats["by_type"]["web"] == 1
        assert stats["by_type"]["pdf"] == 1


# =============================================================================
# 4. Persistence Tests (2)
# =============================================================================


class TestPersistence:
    """Tests for file persistence."""

    def test_sources_persist_to_file(self, temp_manager, sample_source):
        """Test sources are saved to file."""
        temp_manager.add_source(sample_source)

        assert temp_manager.sources_file.exists()

    def test_sources_survive_reload(self):
        """Test sources persist across manager instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("quirkllm.knowledge.knowledge_manager.LanceDBStore"):
                # Create first manager and add source
                manager1 = KnowledgeManager(base_dir=Path(tmpdir))
                source = KnowledgeSource(
                    source_id="persist_test",
                    source_type="web",
                    source_path="https://persist.com",
                    title="Persist Test",
                    chunk_count=5,
                    ingested_at="2024-01-01",
                    metadata={},
                )
                manager1.add_source(source)

                # Create second manager (simulating restart)
                manager2 = KnowledgeManager(base_dir=Path(tmpdir))

                sources = manager2.list_sources()
                assert len(sources) == 1
                assert sources[0].source_id == "persist_test"


# =============================================================================
# Additional Tests
# =============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    def test_generate_source_id_deterministic(self):
        """Test source ID generation is deterministic."""
        path = "https://example.com/docs"

        id1 = KnowledgeManager.generate_source_id(path)
        id2 = KnowledgeManager.generate_source_id(path)

        assert id1 == id2
        assert len(id1) == 16

    def test_generate_source_id_different_paths(self):
        """Test different paths produce different IDs."""
        id1 = KnowledgeManager.generate_source_id("https://a.com")
        id2 = KnowledgeManager.generate_source_id("https://b.com")

        assert id1 != id2

    def test_create_source_factory(self):
        """Test create_source factory method."""
        source = KnowledgeManager.create_source(
            source_path="https://factory.com",
            source_type="web",
            title="Factory Test",
            chunk_count=15,
            metadata={"test": True},
        )

        assert source.source_type == "web"
        assert source.title == "Factory Test"
        assert source.chunk_count == 15
        assert source.metadata == {"test": True}
        assert source.ingested_at is not None


class TestKnowledgeSourceDataclass:
    """Tests for KnowledgeSource dataclass."""

    def test_to_dict(self, sample_source):
        """Test converting source to dictionary."""
        data = sample_source.to_dict()

        assert data["source_id"] == "abc123"
        assert data["source_type"] == "web"
        assert data["title"] == "Example Documentation"

    def test_from_dict(self):
        """Test creating source from dictionary."""
        data = {
            "source_id": "from_dict",
            "source_type": "pdf",
            "source_path": "/path/to/file.pdf",
            "title": "From Dict",
            "chunk_count": 8,
            "ingested_at": "2024-01-01",
            "metadata": {},
        }

        source = KnowledgeSource.from_dict(data)

        assert source.source_id == "from_dict"
        assert source.source_type == "pdf"
