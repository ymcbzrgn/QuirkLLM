"""
Safe File Operations Manager for QuirkLLM (Phase 3.7)

Provides atomic file operations with backup, diff generation, and rollback support.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
import hashlib
import json
from datetime import datetime
import difflib


@dataclass
class Backup:
    """Backup metadata."""
    id: str
    file_path: str
    original_content: str
    timestamp: datetime
    reason: str
    backup_path: str
    checksum: str


@dataclass
class FileEdit:
    """Single file edit operation."""
    file_path: str
    new_content: str
    reason: str = ""
    create_backup: bool = True


class FileManager:
    """
    Safe file manager with atomic operations, backups, and rollback.
    
    Features:
    - Atomic writes (temp file + rename)
    - Automatic backups before edits
    - Unified diff generation
    - Multi-file transactions
    - Rollback support
    """
    
    def __init__(
        self,
        project_root: str,
        backup_dir: Optional[str] = None,
        max_backups_per_file: int = 10
    ):
        """
        Initialize file manager.
        
        Args:
            project_root: Root directory of the project
            backup_dir: Backup directory (default: ~/.quirkllm/backups)
            max_backups_per_file: Maximum backups to keep per file
        """
        self.project_root = Path(project_root).resolve()
        self.backup_dir = Path(backup_dir or Path.home() / ".quirkllm" / "backups")
        self.max_backups_per_file = max_backups_per_file
        
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_checksum(self, content: str) -> str:
        """Compute SHA256 checksum of content."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_backup_id(self) -> str:
        """Generate unique backup ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    def _create_backup(
        self,
        file_path: Path,
        content: str,
        reason: str
    ) -> Backup:
        """
        Create backup of file.
        
        Args:
            file_path: Path to file
            content: File content
            reason: Reason for backup
            
        Returns:
            Backup metadata
        """
        backup_id = self._get_backup_id()
        
        # Create backup subdirectory for this file
        relative_path = file_path.relative_to(self.project_root)
        file_backup_dir = self.backup_dir / str(relative_path).replace(os.sep, "_")
        file_backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Save backup content
        backup_path = file_backup_dir / f"{backup_id}.txt"
        backup_path.write_text(content, encoding="utf-8")
        
        # Save metadata
        metadata = {
            "id": backup_id,
            "file_path": str(file_path),
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "checksum": self._compute_checksum(content)
        }
        
        metadata_path = file_backup_dir / f"{backup_id}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        
        # Cleanup old backups
        self._cleanup_old_backups(file_backup_dir)
        
        return Backup(
            id=backup_id,
            file_path=str(file_path),
            original_content=content,
            timestamp=datetime.now(),
            reason=reason,
            backup_path=str(backup_path),
            checksum=metadata["checksum"]
        )
    
    def _cleanup_old_backups(self, backup_dir: Path) -> None:
        """Remove old backups, keep only max_backups_per_file recent ones."""
        # Get all backup files
        backup_files = sorted(backup_dir.glob("*.txt"))
        
        # Remove oldest if exceeded limit
        if len(backup_files) > self.max_backups_per_file:
            for old_backup in backup_files[:-self.max_backups_per_file]:
                old_backup.unlink()
                # Also remove corresponding metadata
                metadata_file = old_backup.with_suffix(".json")
                if metadata_file.exists():
                    metadata_file.unlink()
    
    def read_file(self, file_path: str) -> str:
        """
        Read file content safely.
        
        Args:
            file_path: Path to file (absolute or relative to project_root)
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file can't be read
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_root / path
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        
        return path.read_text(encoding="utf-8")
    
    def write_file(
        self,
        file_path: str,
        content: str,
        create_backup: bool = True,
        reason: str = ""
    ) -> Optional[Backup]:
        """
        Write file atomically with optional backup.
        
        Args:
            file_path: Path to file
            content: New content
            create_backup: Whether to create backup
            reason: Reason for edit
            
        Returns:
            Backup metadata if backup was created, None otherwise
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_root / path
        
        backup = None
        
        # Create backup if file exists
        if create_backup and path.exists():
            original_content = path.read_text(encoding="utf-8")
            backup = self._create_backup(path, original_content, reason)
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first (atomic operation)
        temp_fd, temp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=".quirkllm_temp_",
            suffix=path.suffix
        )
        
        try:
            # Write content to temp file
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Atomic rename (replaces original file)
            os.replace(temp_path, path)
            
        except Exception as e:
            # Cleanup temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
        
        return backup
    
    def generate_diff(self, file_path: str, new_content: str) -> str:
        """
        Generate unified diff between current and new content.
        
        Args:
            file_path: Path to file
            new_content: New content
            
        Returns:
            Unified diff string
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_root / path
        
        # Get current content
        if path.exists():
            current_content = path.read_text(encoding="utf-8")
        else:
            current_content = ""
        
        # Generate unified diff
        current_lines = current_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            current_lines,
            new_lines,
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
            lineterm=""
        )
        
        return "".join(diff)
    
    def apply_diff(self, file_path: str, diff: str) -> None:
        """
        Apply unified diff to file.
        
        Args:
            file_path: Path to file
            diff: Unified diff string
            
        Raises:
            ValueError: If diff cannot be applied
        """
        # This is a simplified implementation
        # In production, use a proper patch library
        raise NotImplementedError("Diff application not yet implemented")
    
    def multi_file_edit(
        self,
        edits: List[FileEdit],
        atomic: bool = True
    ) -> List[Optional[Backup]]:
        """
        Apply multiple file edits.
        
        Args:
            edits: List of file edits
            atomic: If True, rollback all on any failure
            
        Returns:
            List of backup metadata for each edit
            
        Raises:
            Exception: If any edit fails (and atomic=True)
        """
        backups = []
        completed_edits = []
        
        try:
            for edit in edits:
                backup = self.write_file(
                    edit.file_path,
                    edit.new_content,
                    create_backup=edit.create_backup,
                    reason=edit.reason
                )
                backups.append(backup)
                completed_edits.append(edit)
            
            return backups
            
        except Exception as e:
            if atomic and completed_edits:
                # Rollback completed edits
                for edit, backup in zip(completed_edits, backups):
                    if backup:
                        try:
                            self.rollback_file(edit.file_path, backup.id)
                        except Exception as rollback_error:
                            print(f"Rollback failed for {edit.file_path}: {rollback_error}")
            
            raise e
    
    def rollback_file(self, file_path: str, backup_id: str) -> None:
        """
        Restore file from specific backup.
        
        Args:
            file_path: Path to file
            backup_id: Backup ID to restore
            
        Raises:
            FileNotFoundError: If backup doesn't exist
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_root / path
        
        # Find backup
        relative_path = path.relative_to(self.project_root)
        file_backup_dir = self.backup_dir / str(relative_path).replace(os.sep, "_")
        
        backup_path = file_backup_dir / f"{backup_id}.txt"
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_id}")
        
        # Restore content
        backup_content = backup_path.read_text(encoding="utf-8")
        self.write_file(str(path), backup_content, create_backup=False)
    
    def _get_backup_dirs(self, file_path: Optional[str]) -> List[Path]:
        """Get backup directories to search."""
        if not file_path:
            return [d for d in self.backup_dir.iterdir() if d.is_dir()]
        
        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_root / path
        
        relative_path = path.relative_to(self.project_root)
        file_backup_dir = self.backup_dir / str(relative_path).replace(os.sep, "_")
        
        return [file_backup_dir] if file_backup_dir.exists() else []
    
    def _load_backup_from_metadata(self, metadata_file: Path) -> Optional[Backup]:
        """Load backup from metadata file."""
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            
            backup_path = metadata_file.with_suffix(".txt")
            original_content = backup_path.read_text(encoding="utf-8") if backup_path.exists() else ""
            
            return Backup(
                id=metadata["id"],
                file_path=metadata["file_path"],
                original_content=original_content,
                timestamp=datetime.fromisoformat(metadata["timestamp"]),
                reason=metadata.get("reason", ""),
                backup_path=str(backup_path),
                checksum=metadata["checksum"]
            )
        except Exception as e:
            print(f"Error loading backup {metadata_file}: {e}")
            return None
    
    def list_backups(self, file_path: Optional[str] = None) -> List[Backup]:
        """
        List available backups.
        
        Args:
            file_path: Optional file path to filter backups
            
        Returns:
            List of Backup metadata
        """
        backup_dirs = self._get_backup_dirs(file_path)
        
        backups = []
        for backup_dir in backup_dirs:
            for metadata_file in backup_dir.glob("*.json"):
                backup = self._load_backup_from_metadata(metadata_file)
                if backup:
                    backups.append(backup)
        
        # Sort by timestamp descending
        backups.sort(key=lambda b: b.timestamp, reverse=True)
        
        return backups
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get file information.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file info (size, modified time, etc.)
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_root / path
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        stat = path.stat()
        
        return {
            "path": str(path),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_file": path.is_file(),
            "is_symlink": path.is_symlink(),
            "exists": path.exists()
        }
