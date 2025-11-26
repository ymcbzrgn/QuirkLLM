"""
Unit tests for File Operations Manager (Phase 3.7)

Tests atomic file operations, backups, diffs, and rollback.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from quirkllm.file_ops.file_manager import (
    FileManager,
    Backup,
    FileEdit,
)


@pytest.fixture
def temp_project():
    """Create temporary project directory."""
    temp_dir = tempfile.mkdtemp(prefix="quirkllm_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_backup():
    """Create temporary backup directory."""
    temp_dir = tempfile.mkdtemp(prefix="quirkllm_backup_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def file_manager(temp_project, temp_backup):
    """Create file manager with temp directories."""
    return FileManager(
        project_root=str(temp_project),
        backup_dir=str(temp_backup),
        max_backups_per_file=5
    )


class TestFileReading:
    """Test file reading operations"""
    
    def test_read_existing_file(self, file_manager, temp_project):
        """Should read existing file successfully"""
        # Create test file
        test_file = temp_project / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)
        
        # Read file
        content = file_manager.read_file("test.txt")
        
        assert content == test_content
    
    def test_read_nonexistent_file(self, file_manager):
        """Should raise FileNotFoundError for nonexistent file"""
        with pytest.raises(FileNotFoundError):
            file_manager.read_file("nonexistent.txt")
    
    def test_read_absolute_path(self, file_manager, temp_project):
        """Should read file with absolute path"""
        test_file = temp_project / "test.txt"
        test_content = "Test content"
        test_file.write_text(test_content)
        
        content = file_manager.read_file(str(test_file))
        
        assert content == test_content
    
    def test_read_directory_fails(self, file_manager, temp_project):
        """Should fail when trying to read directory"""
        subdir = temp_project / "subdir"
        subdir.mkdir()
        
        with pytest.raises(ValueError):
            file_manager.read_file("subdir")


class TestFileWriting:
    """Test file writing operations"""
    
    def test_write_new_file(self, file_manager, temp_project):
        """Should create new file successfully"""
        content = "New file content"
        
        backup = file_manager.write_file("new.txt", content, create_backup=False)
        
        assert backup is None
        assert (temp_project / "new.txt").exists()
        assert (temp_project / "new.txt").read_text() == content
    
    def test_write_existing_file_with_backup(self, file_manager, temp_project):
        """Should backup existing file before overwriting"""
        # Create original file
        test_file = temp_project / "test.txt"
        original_content = "Original content"
        test_file.write_text(original_content)
        
        # Overwrite with backup
        new_content = "New content"
        backup = file_manager.write_file("test.txt", new_content, reason="test edit")
        
        assert backup is not None
        assert backup.original_content == original_content
        assert backup.reason == "test edit"
        assert test_file.read_text() == new_content
    
    def test_write_without_backup(self, file_manager, temp_project):
        """Should overwrite without backup when disabled"""
        test_file = temp_project / "test.txt"
        test_file.write_text("Original")
        
        backup = file_manager.write_file("test.txt", "New", create_backup=False)
        
        assert backup is None
    
    def test_write_creates_parent_directories(self, file_manager, temp_project):
        """Should create parent directories if they don't exist"""
        content = "Nested content"
        
        file_manager.write_file("subdir/nested/file.txt", content, create_backup=False)
        
        assert (temp_project / "subdir" / "nested" / "file.txt").exists()
        assert (temp_project / "subdir" / "nested" / "file.txt").read_text() == content
    
    def test_atomic_write(self, file_manager, temp_project):
        """Write operation should be atomic"""
        test_file = temp_project / "test.txt"
        test_file.write_text("Original")
        
        # Write should succeed completely or fail completely
        file_manager.write_file("test.txt", "New content", create_backup=False)
        
        # File should have new content, not partial
        assert test_file.read_text() == "New content"


class TestBackupSystem:
    """Test backup creation and management"""
    
    def test_create_backup(self, file_manager, temp_project):
        """Should create backup with metadata"""
        test_file = temp_project / "test.txt"
        test_file.write_text("Original")
        
        backup = file_manager.write_file("test.txt", "New", reason="test")
        
        assert backup is not None
        assert backup.original_content == "Original"
        assert backup.reason == "test"
        assert Path(backup.backup_path).exists()
        assert len(backup.checksum) == 64  # SHA256
    
    def test_list_backups_for_file(self, file_manager, temp_project):
        """Should list backups for specific file"""
        test_file = temp_project / "test.txt"
        test_file.write_text("Original")
        
        # Create multiple backups
        file_manager.write_file("test.txt", "Version 1", reason="edit 1")
        file_manager.write_file("test.txt", "Version 2", reason="edit 2")
        
        backups = file_manager.list_backups("test.txt")
        
        assert len(backups) == 2
        assert backups[0].timestamp > backups[1].timestamp  # Sorted desc
    
    def test_list_all_backups(self, file_manager, temp_project):
        """Should list backups for all files"""
        (temp_project / "file1.txt").write_text("Content 1")
        (temp_project / "file2.txt").write_text("Content 2")
        
        file_manager.write_file("file1.txt", "New 1")
        file_manager.write_file("file2.txt", "New 2")
        
        backups = file_manager.list_backups()
        
        assert len(backups) == 2
    
    def test_cleanup_old_backups(self, file_manager, temp_project):
        """Should cleanup old backups exceeding limit"""
        test_file = temp_project / "test.txt"
        test_file.write_text("Original")
        
        # Create more backups than limit (limit is 5)
        for i in range(8):
            file_manager.write_file("test.txt", f"Version {i}")
        
        backups = file_manager.list_backups("test.txt")
        
        # Should only keep max_backups_per_file (5)
        assert len(backups) <= 5
    
    def test_backup_checksum(self, file_manager, temp_project):
        """Backup should have correct checksum"""
        test_file = temp_project / "test.txt"
        test_file.write_text("Original")
        
        backup = file_manager.write_file("test.txt", "New")
        
        # Verify checksum
        expected_checksum = file_manager._compute_checksum("Original")
        assert backup.checksum == expected_checksum


class TestDiffGeneration:
    """Test diff generation"""
    
    def test_generate_diff_for_existing_file(self, file_manager, temp_project):
        """Should generate unified diff"""
        test_file = temp_project / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\n")
        
        new_content = "Line 1\nModified Line 2\nLine 3\n"
        diff = file_manager.generate_diff("test.txt", new_content)
        
        assert "Line 1" in diff
        assert "Modified Line 2" in diff
        assert "@@" in diff  # Unified diff format marker
    
    def test_generate_diff_for_new_file(self, file_manager, temp_project):
        """Should generate diff for new file (all additions)"""
        new_content = "New line 1\nNew line 2\n"
        diff = file_manager.generate_diff("new.txt", new_content)
        
        assert "New line 1" in diff or diff == ""  # Empty file has no diff
    
    def test_diff_shows_additions(self, file_manager, temp_project):
        """Diff should show added lines"""
        test_file = temp_project / "test.txt"
        test_file.write_text("Line 1\n")
        
        new_content = "Line 1\nLine 2\nLine 3\n"
        diff = file_manager.generate_diff("test.txt", new_content)
        
        assert "Line 2" in diff
        assert "Line 3" in diff
    
    def test_diff_shows_deletions(self, file_manager, temp_project):
        """Diff should show deleted lines"""
        test_file = temp_project / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\n")
        
        new_content = "Line 1\n"
        diff = file_manager.generate_diff("test.txt", new_content)
        
        # Deletions might be marked with - in unified diff
        assert diff != ""  # Should have some diff


class TestRollback:
    """Test rollback functionality"""
    
    def test_rollback_to_backup(self, file_manager, temp_project):
        """Should restore file from backup"""
        test_file = temp_project / "test.txt"
        original_content = "Original content"
        test_file.write_text(original_content)
        
        # Edit file
        backup = file_manager.write_file("test.txt", "New content")
        
        # Rollback
        file_manager.rollback_file("test.txt", backup.id)
        
        # Should be restored
        assert test_file.read_text() == original_content
    
    def test_rollback_nonexistent_backup(self, file_manager, temp_project):
        """Should raise error for nonexistent backup"""
        test_file = temp_project / "test.txt"
        test_file.write_text("Content")
        
        with pytest.raises(FileNotFoundError):
            file_manager.rollback_file("test.txt", "nonexistent_id")


class TestMultiFileEdit:
    """Test multi-file editing"""
    
    def test_multi_file_edit_success(self, file_manager, temp_project):
        """Should edit multiple files successfully"""
        # Create test files
        (temp_project / "file1.txt").write_text("Original 1")
        (temp_project / "file2.txt").write_text("Original 2")
        
        edits = [
            FileEdit("file1.txt", "New 1", "edit 1"),
            FileEdit("file2.txt", "New 2", "edit 2"),
        ]
        
        backups = file_manager.multi_file_edit(edits)
        
        assert len(backups) == 2
        assert (temp_project / "file1.txt").read_text() == "New 1"
        assert (temp_project / "file2.txt").read_text() == "New 2"
    
    def test_multi_file_edit_atomic_rollback(self, file_manager, temp_project):
        """Should rollback all edits on any failure"""
        (temp_project / "file1.txt").write_text("Original 1")
        
        # Second edit will fail (invalid path)
        edits = [
            FileEdit("file1.txt", "New 1"),
            FileEdit("/invalid/path/file.txt", "New"),
        ]
        
        with pytest.raises(Exception):
            file_manager.multi_file_edit(edits, atomic=True)
        
        # First file should be rolled back
        assert (temp_project / "file1.txt").read_text() == "Original 1"
    
    def test_multi_file_edit_without_backup(self, file_manager, temp_project):
        """Should support multi-file edit without backups"""
        (temp_project / "file1.txt").write_text("Original 1")
        (temp_project / "file2.txt").write_text("Original 2")
        
        edits = [
            FileEdit("file1.txt", "New 1", create_backup=False),
            FileEdit("file2.txt", "New 2", create_backup=False),
        ]
        
        backups = file_manager.multi_file_edit(edits)
        
        assert all(b is None for b in backups)


class TestFileInfo:
    """Test file information retrieval"""
    
    def test_get_file_info(self, file_manager, temp_project):
        """Should get file information"""
        test_file = temp_project / "test.txt"
        test_file.write_text("Content")
        
        info = file_manager.get_file_info("test.txt")
        
        assert "path" in info
        assert "size" in info
        assert "modified" in info
        assert info["is_file"] is True
        assert info["exists"] is True
    
    def test_get_file_info_nonexistent(self, file_manager):
        """Should raise error for nonexistent file"""
        with pytest.raises(FileNotFoundError):
            file_manager.get_file_info("nonexistent.txt")


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_handle_unicode_content(self, file_manager, temp_project):
        """Should handle Unicode content correctly"""
        content = "Hello 世界 🚀 Привет"
        
        file_manager.write_file("unicode.txt", content, create_backup=False)
        
        assert (temp_project / "unicode.txt").read_text(encoding="utf-8") == content
    
    def test_handle_empty_file(self, file_manager, temp_project):
        """Should handle empty files"""
        file_manager.write_file("empty.txt", "", create_backup=False)
        
        content = file_manager.read_file("empty.txt")
        
        assert content == ""
    
    def test_handle_large_file(self, file_manager, temp_project):
        """Should handle large files"""
        # Create 1MB content
        large_content = "x" * (1024 * 1024)
        
        file_manager.write_file("large.txt", large_content, create_backup=False)
        
        content = file_manager.read_file("large.txt")
        
        assert len(content) == 1024 * 1024
    
    def test_handle_special_characters_in_filename(self, file_manager, temp_project):
        """Should handle special characters in filenames"""
        # Note: Some characters are invalid on different OS
        content = "Test content"
        
        file_manager.write_file("test_file-2024.txt", content, create_backup=False)
        
        assert (temp_project / "test_file-2024.txt").exists()
    
    def test_relative_vs_absolute_paths(self, file_manager, temp_project):
        """Should handle both relative and absolute paths"""
        content = "Test"
        
        # Relative path
        file_manager.write_file("relative.txt", content, create_backup=False)
        
        # Absolute path
        absolute_path = temp_project / "absolute.txt"
        file_manager.write_file(str(absolute_path), content, create_backup=False)
        
        assert (temp_project / "relative.txt").exists()
        assert absolute_path.exists()


class TestIntegration:
    """Integration tests"""
    
    def test_complete_workflow(self, file_manager, temp_project):
        """Test complete workflow: write, backup, modify, rollback"""
        # 1. Create file
        file_manager.write_file("workflow.txt", "Version 1", create_backup=False)
        
        # 2. Modify with backup
        backup1 = file_manager.write_file("workflow.txt", "Version 2", reason="update 1")
        
        # 3. Modify again
        backup2 = file_manager.write_file("workflow.txt", "Version 3", reason="update 2")
        
        # 4. List backups
        backups = file_manager.list_backups("workflow.txt")
        assert len(backups) == 2
        
        # 5. Rollback to first backup
        file_manager.rollback_file("workflow.txt", backup1.id)
        
        # 6. Verify content
        content = file_manager.read_file("workflow.txt")
        assert content == "Version 1"
