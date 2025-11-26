"""
Ghost Mode - Background file watcher and impact analyzer.

Ghost Mode is a non-intrusive background watcher that monitors file changes
and provides real-time analysis without disrupting the developer's workflow.
Perfect for passive code review and catching potential issues early.

Behavior:
- Runs in background thread (non-blocking)
- Watches configured file patterns (*.py, *.js, etc.)
- Queues file change events with debouncing
- Triggers analysis on file save
- Provides impact analysis and breaking change detection
- Non-intrusive notifications

Use Cases:
- Real-time code review during development
- Breaking change detection across codebase
- Impact analysis for refactoring
- Passive assistance while coding in IDE
- Catching potential bugs on save

Technical Features:
- watchdog library for file system events
- Threading for background execution
- Debouncing to avoid duplicate events
- Pattern matching for selective watching
- Queue-based event processing
"""

import time
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from threading import Event, Lock
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from quirkllm.modes.base import (
    ModeBase,
    ModeType,
    ModeConfig,
    ActionRequest,
    ActionResult,
)


class CodeChangeHandler(FileSystemEventHandler):
    """
    File system event handler for Ghost Mode.
    
    Processes file create, modify, and delete events, filtering by
    configured patterns and forwarding to the callback.
    
    Attributes:
        callback: Function to call when relevant file changes
        patterns: List of glob patterns to watch (e.g., ['*.py', '*.js'])
        debounce_time: Minimum time between events for same file (seconds)
        last_event_times: Tracking for debouncing
        lock: Thread safety for last_event_times
    """
    
    def __init__(
        self,
        callback: Callable[[str, str], None],
        patterns: List[str],
        debounce_time: float = 0.5,
    ):
        """
        Initialize event handler.
        
        Args:
            callback: Function(filepath, event_type) to call on events
            patterns: Glob patterns to watch
            debounce_time: Seconds to wait before processing duplicate events
        """
        super().__init__()
        self.callback = callback
        self.patterns = patterns
        self.debounce_time = debounce_time
        self.last_event_times: Dict[str, float] = {}
        self.lock = Lock()
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """
        Handle file modification event.
        
        Args:
            event: File system event from watchdog
        """
        if event.is_directory:
            return
        
        filepath = event.src_path
        
        if self._should_watch(filepath) and self._should_process(filepath):
            self.callback(filepath, "modified")
    
    def on_created(self, event: FileSystemEvent) -> None:
        """
        Handle file creation event.
        
        Args:
            event: File system event from watchdog
        """
        if event.is_directory:
            return
        
        filepath = event.src_path
        
        if self._should_watch(filepath) and self._should_process(filepath):
            self.callback(filepath, "created")
    
    def on_deleted(self, event: FileSystemEvent) -> None:
        """
        Handle file deletion event.
        
        Args:
            event: File system event from watchdog
        """
        if event.is_directory:
            return
        
        filepath = event.src_path
        
        if self._should_watch(filepath) and self._should_process(filepath):
            self.callback(filepath, "deleted")
    
    def _should_watch(self, filepath: str) -> bool:
        """
        Check if file matches watch patterns.
        
        Args:
            filepath: Absolute path to file
            
        Returns:
            True if file should be watched
        """
        path = Path(filepath)
        
        # Check against all patterns
        for pattern in self.patterns:
            if path.match(pattern):
                return True
        
        return False
    
    def _should_process(self, filepath: str) -> bool:
        """
        Check if event should be processed (debouncing).
        
        Prevents processing multiple events for the same file
        within debounce_time window.
        
        Args:
            filepath: Absolute path to file
            
        Returns:
            True if event should be processed
        """
        current_time = time.time()
        
        with self.lock:
            last_time = self.last_event_times.get(filepath, 0)
            
            # Check if enough time has passed
            if current_time - last_time >= self.debounce_time:
                self.last_event_times[filepath] = current_time
                return True
        
        return False


class GhostMode(ModeBase):
    """
    Ghost Mode - Background file watcher and impact analyzer.
    
    Ghost Mode runs as a background thread watching for file changes.
    It provides non-intrusive analysis and notifications without blocking
    the main workflow.
    
    Read-Only Observation:
    - Watches file changes but doesn't modify anything
    - Analyzes impact and potential issues
    - Queues changes for review
    
    Background Operation:
    - Runs in separate thread
    - Non-blocking file system monitoring
    - Debounced event processing
    
    Watch Patterns:
    Supports glob patterns like:
    - *.py - All Python files
    - src/**/*.js - All JS files in src/ recursively
    - **/*.{ts,tsx} - All TypeScript files
    
    Attributes:
        console: Rich console for UI output
        watch_dir: Root directory to watch
        observer: Watchdog observer instance
        watching: Event flag for watcher thread
        analysis_queue: Queue of detected changes
        session_stats: Statistics tracking
    """
    
    DEFAULT_PATTERNS = [
        "**/*.py",
        "**/*.js",
        "**/*.ts",
        "**/*.tsx",
        "**/*.jsx",
    ]
    
    def __init__(
        self,
        watch_dir: str = ".",
        patterns: Optional[List[str]] = None,
        debounce_ms: int = 500,
        **kwargs: Any
    ) -> None:
        """
        Initialize Ghost mode.
        
        Args:
            watch_dir: Directory to watch for changes
            patterns: Glob patterns to watch (defaults to DEFAULT_PATTERNS)
            debounce_ms: Debounce delay in milliseconds
            **kwargs: Additional configuration options
        """
        # Ghost config: background watch, read-only, auto-confirm
        config = ModeConfig(
            auto_confirm=True,  # Background mode, no confirmations
            allow_file_edits=False,  # Read-only
            allow_destructive=False,
            background_watch=True,
            watch_patterns=patterns or self.DEFAULT_PATTERNS,
            watch_debounce_ms=debounce_ms,
        )
        super().__init__(ModeType.GHOST, config)
        self.console = Console()
        self.watch_dir = Path(watch_dir).resolve()
        self.observer: Optional[Observer] = None
        self.watching = Event()
        self.analysis_queue: List[Dict[str, Any]] = []
        self.queue_lock = Lock()
        
        # Session statistics
        self.session_stats: Dict[str, Any] = {
            "changes_detected": 0,
            "files_analyzed": 0,
            "watcher_active": False,
        }
    
    def activate(self) -> None:
        """
        Activate Ghost mode and start file watcher.
        
        Starts background thread monitoring file system for changes.
        Displays activation message with watch configuration.
        """
        self._active = True
        
        # Reset session stats
        self.session_stats = {
            "changes_detected": 0,
            "files_analyzed": 0,
            "watcher_active": True,
        }
        
        # Display activation panel
        welcome_panel = Panel(
            "[bold purple]GHOST MODE ACTIVATED[/]\n\n"
            "[purple]Background file watcher[/]\n\n"
            "👻 [green]Watching[/] for file changes\n"
            "📂 [green]Directory[/]: {}\n"
            "🔍 [green]Patterns[/]: {}\n"
            "⏱️  [green]Debounce[/]: {}ms\n\n"
            "[dim]Non-intrusive analysis running in background[/]".format(
                self.watch_dir,
                ", ".join(self.config.watch_patterns[:3])
                + (f" (+{len(self.config.watch_patterns) - 3} more)" if len(self.config.watch_patterns) > 3 else ""),
                self.config.watch_debounce_ms,
            ),
            title="👻 Ghost Mode",
            border_style="purple",
        )
        self.console.print(welcome_panel)
        
        # Start file watcher
        self._start_watching()
    
    def deactivate(self) -> None:
        """
        Deactivate Ghost mode and stop file watcher.
        
        Stops background thread and displays session summary with
        detected changes and analysis queue.
        """
        self._active = False
        self.watching.clear()
        self.session_stats["watcher_active"] = False
        
        # Stop observer
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=2.0)
        
        # Display session summary
        if self.session_stats["changes_detected"] > 0:
            summary_table = Table(title="Ghost Mode Session Summary", show_header=True)
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Count", style="green", justify="right")
            
            summary_table.add_row("Changes Detected", str(self.session_stats["changes_detected"]))
            summary_table.add_row("Files in Queue", str(len(self.analysis_queue)))
            
            self.console.print("\n")
            self.console.print(summary_table)
            
            # Show recent changes
            if self.analysis_queue:
                self.console.print("\n[purple]Recent Changes:[/]")
                for change in self.analysis_queue[-5:]:  # Last 5
                    filepath = Path(change["filepath"]).name
                    event_type = change["event_type"]
                    self.console.print(f"  [dim]• {filepath} ({event_type})[/]")
        else:
            self.console.print("\n[dim]No changes detected in this session[/]")
        
        self.console.print("\n[dim]Ghost mode deactivated[/]")
    
    def handle_action(self, request: ActionRequest) -> ActionResult:
        """
        Handle action with read-only enforcement.
        
        Ghost mode only supports analysis actions, all write operations
        are blocked.
        
        Args:
            request: The action to handle
            
        Returns:
            ActionResult with execution status
        """
        # Handle analysis requests
        if request.action_type == "analyze_change":
            return self._analyze_file_change(request)
        
        # Block all write operations
        if request.action_type in [
            "file_edit",
            "delete",
            "command",
            "write_file",
            "create_file",
        ]:
            return ActionResult(
                success=False,
                message="Ghost Mode is read-only observation mode. Use /mode chat to make changes.",
                warnings=["Background watcher is read-only"],
            )
        
        # Allow read operations
        if request.action_type in ["read_file", "list_files"]:
            return ActionResult(
                success=True,
                message=f"Read operation '{request.action_type}' allowed",
            )
        
        return ActionResult(
            success=False,
            message=f"Action '{request.action_type}' not supported in Ghost Mode",
        )
    
    def get_prompt_indicator(self) -> str:
        """
        Get the prompt indicator for Ghost mode.
        
        Returns:
            Ghost emoji indicating background watching
        """
        return "👻"
    
    def _start_watching(self) -> None:
        """
        Start file system watcher in background thread.
        
        Creates watchdog Observer and schedules event handler for
        the watch directory with recursive monitoring.
        """
        # Create event handler
        event_handler = CodeChangeHandler(
            callback=self._on_file_changed,
            patterns=self.config.watch_patterns,
            debounce_time=self.config.watch_debounce_ms / 1000.0,
        )
        
        # Create and start observer
        self.observer = Observer()
        self.observer.schedule(
            event_handler,
            str(self.watch_dir),
            recursive=True,
        )
        self.observer.start()
        self.watching.set()
        
        self.console.print("[dim]👻 Watcher started[/]")
    
    def _on_file_changed(self, filepath: str, event_type: str) -> None:
        """
        Callback when a watched file changes.
        
        Adds change to analysis queue and updates statistics.
        In full implementation, would trigger background analysis.
        
        Args:
            filepath: Absolute path to changed file
            event_type: Type of event (created, modified, deleted)
        """
        # Add to analysis queue
        with self.queue_lock:
            self.analysis_queue.append({
                "filepath": filepath,
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
            })
            self.session_stats["changes_detected"] += 1
        
        # Display notification
        path = Path(filepath)
        self.console.print(
            f"\n[dim]👻 Detected: {path.name} ({event_type})[/]"
        )
        
        # Trigger analysis (placeholder for full implementation)
        self._trigger_analysis(filepath, event_type)
    
    def _trigger_analysis(self, filepath: str, event_type: str) -> None:
        """
        Trigger background analysis of changed file.
        
        Placeholder for full implementation with RAG and project analysis.
        In real system, would:
        1. Parse the changed file
        2. Check for breaking changes
        3. Analyze impact on dependent files
        4. Check for potential bugs
        5. Show non-intrusive notification
        
        Args:
            filepath: Path to changed file
            event_type: Type of change event
        """
        # Placeholder - real implementation will use FileManager + RAG
        _ = Path(filepath)  # Will be used in full implementation
        
        # Simulate quick analysis
        self.session_stats["files_analyzed"] += 1
        
        # In full implementation:
        # - Use FileManager to read/parse file
        # - Use RAG to find dependent files
        # - Use project analyzer for impact assessment
        # - Generate analysis report
        # - Show notification panel
    
    def _analyze_file_change(self, request: ActionRequest) -> ActionResult:
        """
        Analyze a specific file change.
        
        Args:
            request: Analysis request with file path
            
        Returns:
            ActionResult with analysis details
        """
        filepath = request.target
        _ = request.details or {}  # Reserved for future use
        
        path = Path(filepath)
        
        # Check if file exists
        if not path.exists():
            return ActionResult(
                success=False,
                message=f"File not found: {filepath}",
            )
        
        # Perform analysis (placeholder)
        analysis = {
            "filepath": filepath,
            "filename": path.name,
            "size": path.stat().st_size if path.exists() else 0,
            "message": "Analysis complete (placeholder)",
        }
        
        return ActionResult(
            success=True,
            message=f"Analyzed {path.name}",
            details=analysis,
        )
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get current session statistics.
        
        Returns:
            Dictionary with changes_detected, files_analyzed, watcher_active
        """
        return self.session_stats.copy()
    
    def get_analysis_queue(self) -> List[Dict[str, Any]]:
        """
        Get current analysis queue.
        
        Returns:
            List of queued file changes
        """
        with self.queue_lock:
            return self.analysis_queue.copy()
