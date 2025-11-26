"""
Integration tests for Phase 3 - Simplified version focusing on working tests.

Tests:
- Multi-file edit transactions with rollback
- Performance benchmarks for project analysis
"""

import shutil
import tempfile
import time
from pathlib import Path

import pytest

from quirkllm.analyzer.project_analyzer import analyze_project
from quirkllm.file_ops.file_manager import FileEdit, FileManager


@pytest.fixture
def temp_project():
    """Create a temporary test project."""
    project_dir = tempfile.mkdtemp(prefix="test_project_")
    project_path = Path(project_dir)
    
    # Create minimal project structure
    (project_path / "README.md").write_text("# Test Project\n")
    
    src_dir = project_path / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("def main():\n    print('Hello')\n")
    
    yield project_path
    
    # Cleanup
    shutil.rmtree(project_dir)


class TestMultiFileEditWorkflow:
    """Test file operations with multi-file transactions."""
    
    def test_atomic_multi_file_edit_success(self, temp_project):
        """Test successful multi-file edit creates backups and applies changes."""
        file_manager = FileManager(str(temp_project))
        
        # Create test files
        file1 = temp_project / "file1.txt"
        file2 = temp_project / "file2.txt"
        file1.write_text("Original content 1")
        file2.write_text("Original content 2")
        
        # Create edits
        edits = [
            FileEdit(
                file_path=str(file1),
                new_content="Modified content 1",
                reason="Test edit 1",
                create_backup=True
            ),
            FileEdit(
                file_path=str(file2),
                new_content="Modified content 2",
                reason="Test edit 2",
                create_backup=True
            )
        ]
        
        # Execute transaction
        backups = file_manager.multi_file_edit(edits, atomic=True)
        
        assert len(backups) == 2
        assert file1.read_text() == "Modified content 1"
        assert file2.read_text() == "Modified content 2"
        
        # Verify backups exist
        for backup in backups:
            assert backup is not None
            assert Path(backup.backup_path).exists()
    
    def test_atomic_multi_file_edit_rollback(self, temp_project):
        """Test multi-file edit rolls back on failure."""
        file_manager = FileManager(str(temp_project))
        
        # Create test files
        file1 = temp_project / "file1.txt"
        file2 = temp_project / "file2.txt"
        file1.write_text("Original content 1")
        file2.write_text("Original content 2")
        
        # Create edits with one that will fail (invalid path)
        edits = [
            FileEdit(
                file_path=str(file1),
                new_content="Modified content 1",
                reason="Test edit 1",
                create_backup=True
            ),
            FileEdit(
                file_path="/nonexistent/path/file.txt",  # This will fail
                new_content="Modified content 2",
                reason="Test edit 2",
                create_backup=True
            )
        ]
        
        # Execute transaction (should rollback)
        with pytest.raises(Exception):
            file_manager.multi_file_edit(edits, atomic=True)
        
        # Verify file1 was rolled back
        assert file1.read_text() == "Original content 1"
    
    def test_backup_and_rollback_integration(self, temp_project):
        """Test complete backup and rollback workflow."""
        file_manager = FileManager(str(temp_project))
        
        # Create and modify a file
        test_file = temp_project / "test.txt"
        original_content = "Version 1"
        test_file.write_text(original_content)
        
        # Edit 1
        backup1 = file_manager.write_file(
            str(test_file),
            "Version 2",
            create_backup=True,
            reason="First edit"
        )
        assert test_file.read_text() == "Version 2"
        
        # Edit 2
        _ = file_manager.write_file(
            str(test_file),
            "Version 3",
            create_backup=True,
            reason="Second edit"
        )
        assert test_file.read_text() == "Version 3"
        
        # List backups
        backups = file_manager.list_backups(str(test_file))
        assert len(backups) >= 2
        
        # Rollback to first backup
        file_manager.rollback_file(str(test_file), backup1.id)
        assert test_file.read_text() == "Version 1"


class TestPerformanceBenchmarks:
    """Performance benchmarks for Phase 3 operations."""
    
    def test_small_project_analysis_performance(self, temp_project):
        """Test project analysis completes in <5s for small projects."""
        start_time = time.time()
        
        result = analyze_project(str(temp_project))
        
        elapsed = time.time() - start_time
        
        # Should complete quickly (project is small)
        assert elapsed < 5.0, f"Analysis took {elapsed:.2f}s, target is <5s"
        
        # Basic validation
        assert result is not None
        print(f"\nProject analysis completed in {elapsed:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
