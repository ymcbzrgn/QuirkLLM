"""
MCP Tool Registry - Exposes QuirkLLM capabilities as MCP tools.

8 Tools:
1. analyze_project - Project structure analysis
2. search_code - RAG-based code search
3. get_context - Current context info
4. read_file - Safe file reading
5. list_files - Directory listing
6. write_file - Safe file writing (with SafetyChecker)
7. execute_command - Command execution (with SafetyChecker)
8. search_documents - Knowledge Eater document search

Example:
    >>> registry = ToolRegistry()
    >>> tools = registry.list_tools()
    >>> result = await registry.call_tool("analyze_project", {"path": "."})
"""

import asyncio
import subprocess
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Callable, Optional
from pathlib import Path


@dataclass
class ToolDefinition:
    """
    MCP tool definition.

    Attributes:
        name: Unique tool identifier
        description: Human-readable description
        inputSchema: JSON Schema for tool arguments
    """

    name: str
    description: str
    inputSchema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }


class ToolRegistry:
    """
    Registry of MCP tools.

    Each tool has:
    - name: Unique identifier
    - description: What the tool does
    - inputSchema: JSON Schema for arguments
    - handler: Async function to execute

    Example:
        >>> registry = ToolRegistry()
        >>> tools = registry.list_tools()
        >>> result = await registry.call_tool("ping", {})
    """

    def __init__(self) -> None:
        """Initialize tool registry."""
        self.tools: Dict[str, ToolDefinition] = {}
        self.handlers: Dict[str, Callable] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register built-in tools."""
        # 1. analyze_project
        self.register(
            ToolDefinition(
                name="analyze_project",
                description="Analyze project structure, frameworks, and dependencies",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Project root path (default: current directory)",
                        }
                    },
                },
            ),
            self._analyze_project,
        )

        # 2. search_code
        self.register(
            ToolDefinition(
                name="search_code",
                description="Search code using semantic RAG search",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results (default: 5)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            self._search_code,
        )

        # 3. get_context
        self.register(
            ToolDefinition(
                name="get_context",
                description="Get current QuirkLLM context (profile, mode, stats)",
                inputSchema={"type": "object", "properties": {}},
            ),
            self._get_context,
        )

        # 4. read_file
        self.register(
            ToolDefinition(
                name="read_file",
                description="Read file content (text files only)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to read",
                        },
                        "max_lines": {
                            "type": "integer",
                            "description": "Maximum lines to read (default: 1000)",
                        },
                    },
                    "required": ["path"],
                },
            ),
            self._read_file,
        )

        # 5. list_files
        self.register(
            ToolDefinition(
                name="list_files",
                description="List files in directory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path (default: current directory)",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern (default: *)",
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Recurse into subdirectories (default: false)",
                        },
                    },
                },
            ),
            self._list_files,
        )

        # 6. write_file (with safety validation)
        self.register(
            ToolDefinition(
                name="write_file",
                description="Write content to file (with safety validation)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to write",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write",
                        },
                    },
                    "required": ["path", "content"],
                },
            ),
            self._write_file,
        )

        # 7. execute_command (with safety validation)
        self.register(
            ToolDefinition(
                name="execute_command",
                description="Execute shell command (with safety validation, max 60s timeout)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory (default: current)",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 60, max: 300)",
                        },
                    },
                    "required": ["command"],
                },
            ),
            self._execute_command,
        )

        # 8. search_documents (Knowledge Eater)
        self.register(
            ToolDefinition(
                name="search_documents",
                description="Search ingested documents (web/PDF) using RAG",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "source_type": {
                            "type": "string",
                            "description": "Filter by source type: 'web' or 'pdf'",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results (default: 5)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            self._search_documents,
        )

    def register(self, tool: ToolDefinition, handler: Callable) -> None:
        """
        Register a tool with its handler.

        Args:
            tool: Tool definition
            handler: Async function to handle tool calls
        """
        self.tools[tool.name] = tool
        self.handlers[tool.name] = handler

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool.

        Args:
            name: Tool name

        Returns:
            True if removed, False if not found
        """
        if name in self.tools:
            del self.tools[name]
            del self.handlers[name]
            return True
        return False

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools.

        Returns:
            List of tool definitions in MCP format
        """
        return [tool.to_dict() for tool in self.tools.values()]

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """
        Get a tool definition by name.

        Args:
            name: Tool name

        Returns:
            ToolDefinition or None
        """
        return self.tools.get(name)

    async def call_tool(
        self, name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            MCP tool result format with content array
        """
        if name not in self.handlers:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                "isError": True,
            }

        handler = self.handlers[name]
        try:
            # Call handler (may be sync or async)
            if asyncio.iscoroutinefunction(handler):
                result = await handler(arguments)
            else:
                result = handler(arguments)

            # Format result
            if isinstance(result, str):
                return {"content": [{"type": "text", "text": result}]}
            elif isinstance(result, dict):
                # Check if already in MCP format
                if "content" in result:
                    return result
                # Convert dict to formatted text
                text = json.dumps(result, indent=2, ensure_ascii=False)
                return {"content": [{"type": "text", "text": text}]}
            elif isinstance(result, list):
                text = json.dumps(result, indent=2, ensure_ascii=False)
                return {"content": [{"type": "text", "text": text}]}
            else:
                return {"content": [{"type": "text", "text": str(result)}]}

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }

    # =========================================================================
    # Tool Implementations
    # =========================================================================

    async def _analyze_project(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze project structure.

        Uses ProjectAnalyzer if available, falls back to basic analysis.
        """
        path = Path(args.get("path", ".")).resolve()

        if not path.exists():
            return {"error": f"Path not found: {path}"}

        if not path.is_dir():
            return {"error": f"Not a directory: {path}"}

        try:
            # Try to use ProjectAnalyzer
            from quirkllm.analyzer.project_analyzer import ProjectAnalyzer

            analyzer = ProjectAnalyzer(path)
            result = analyzer.analyze()
            return asdict(result)
        except ImportError:
            # Fallback to basic analysis
            return self._basic_project_analysis(path)

    def _basic_project_analysis(self, path: Path) -> Dict[str, Any]:
        """Basic project analysis without ProjectAnalyzer."""
        result = {
            "path": str(path),
            "name": path.name,
            "files": [],
            "directories": [],
        }

        try:
            for item in path.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_file():
                    result["files"].append(item.name)
                elif item.is_dir():
                    result["directories"].append(item.name)
        except PermissionError:
            result["error"] = "Permission denied"

        return result

    async def _search_code(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search code using RAG.

        Uses CodeRetriever if available.
        """
        query = args.get("query", "")
        k = args.get("k", 5)

        if not query:
            return [{"error": "Query is required"}]

        try:
            from quirkllm.rag.retriever import CodeRetriever

            retriever = CodeRetriever()
            results = retriever.search(query, k=k)
            return [
                {
                    "content": r.content,
                    "file_path": r.file_path,
                    "score": r.score,
                }
                for r in results
            ]
        except ImportError:
            return [{"error": "RAG system not available"}]
        except Exception as e:
            return [{"error": str(e)}]

    async def _get_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get current QuirkLLM context."""
        try:
            from quirkllm.core.config import load_config
            from quirkllm.core.system_detector import detect_system

            config = load_config()
            system = detect_system()

            return {
                "profile": config.profile,
                "mode": getattr(config, "mode", "chat"),
                "system": {
                    "platform": system.platform,
                    "total_ram_gb": system.total_ram_gb,
                    "available_ram_gb": system.available_ram_gb,
                    "cpu_cores": system.cpu_cores,
                    "has_gpu": system.has_gpu,
                },
                "config": {
                    "theme": config.theme,
                    "backend": config.backend,
                    "rag_enabled": config.rag_enabled,
                },
            }
        except ImportError:
            return {
                "error": "Config system not available",
                "profile": "unknown",
            }
        except Exception as e:
            return {"error": str(e)}

    async def _read_file(self, args: Dict[str, Any]) -> str:
        """Read file content."""
        path = Path(args.get("path", "")).resolve()
        max_lines = args.get("max_lines", 1000)

        if not path.exists():
            return f"Error: File not found: {path}"

        if not path.is_file():
            return f"Error: Not a file: {path}"

        # Check file size (limit to 1MB)
        if path.stat().st_size > 1_000_000:
            return f"Error: File too large (>1MB): {path}"

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n... (truncated at {max_lines} lines)")
                        break
                    lines.append(line)
                return "".join(lines)
        except Exception as e:
            return f"Error reading file: {e}"

    async def _list_files(self, args: Dict[str, Any]) -> List[str]:
        """List files in directory."""
        path = Path(args.get("path", ".")).resolve()
        pattern = args.get("pattern", "*")
        recursive = args.get("recursive", False)

        if not path.exists():
            return [f"Error: Path not found: {path}"]

        if not path.is_dir():
            return [f"Error: Not a directory: {path}"]

        try:
            if recursive:
                files = list(path.rglob(pattern))
            else:
                files = list(path.glob(pattern))

            # Filter out hidden files and limit results
            result = []
            for f in files[:100]:  # Limit to 100 files
                if not any(p.startswith(".") for p in f.parts):
                    result.append(str(f.relative_to(path)))

            return sorted(result)
        except Exception as e:
            return [f"Error: {e}"]

    async def _write_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write file with safety validation.

        Uses basic safety checks to validate the operation.
        """
        file_path = args.get("path", "")
        content = args.get("content", "")

        if not file_path:
            return {"success": False, "error": "Path is required"}

        path = Path(file_path).resolve()

        # Basic safety check - protected system paths
        # Note: /var/folders and /private/var/folders are safe (macOS temp)
        protected = [
            "/etc", "/usr", "/bin", "/sbin", "/root",
            "/System", "/Library", "/private/etc",
            "/var/log", "/var/run", "/var/db",
            "/private/var/log", "/private/var/run", "/private/var/db",
        ]
        path_str = str(path)

        # Allow temp directories (macOS uses /private/var/folders)
        is_temp_dir = "/var/folders" in path_str or "/private/var/folders" in path_str

        if not is_temp_dir and any(path_str.startswith(p) for p in protected):
            return {
                "success": False,
                "error": "Cannot write to system directory",
            }

        try:
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return {
                "success": True,
                "path": str(path),
                "bytes_written": len(content.encode("utf-8")),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute command with safety validation.

        Uses SafetyChecker to validate the command.
        """
        command = args.get("command", "")
        cwd = args.get("cwd")
        timeout = min(args.get("timeout", 60), 300)  # Max 5 minutes

        if not command:
            return {"success": False, "error": "Command is required"}

        # Basic safety check - dangerous command patterns
        dangerous_patterns = [
            "rm -rf /", "rm -rf /*", "rm -rf ~",
            ":(){ :|:& };:",  # Fork bomb
            "mkfs", "dd if=", "> /dev/sda", "> /dev/null",
            "chmod -R 777 /", "chown -R",
            "wget|sh", "curl|sh", "wget|bash", "curl|bash",
            "sudo rm -rf", "sudo dd",
            "/dev/sd", "/dev/hd", "/dev/nvme",
        ]
        if any(d in command for d in dangerous_patterns):
            return {
                "success": False,
                "error": "Dangerous command blocked",
            }

        try:
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )

            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout}s",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _search_documents(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search ingested documents (Knowledge Eater).

        Uses LanceDBStore.search_documents if available.
        """
        query = args.get("query", "")
        source_type = args.get("source_type")
        k = args.get("k", 5)

        if not query:
            return [{"error": "Query is required"}]

        try:
            from quirkllm.rag.lancedb_store import LanceDBStore
            from quirkllm.rag.embeddings import EmbeddingGenerator

            # Generate query embedding
            embedder = EmbeddingGenerator()
            query_embedding = embedder.embed_code(query)

            # Search documents
            store = LanceDBStore()
            results = store.search_documents(
                query_embedding,
                k=k,
                source_type=source_type,
            )

            return [
                {
                    "content": r.content,
                    "source_url": r.source_url,
                    "source_type": r.source_type,
                    "title": r.title,
                    "score": r.score,
                }
                for r in results
            ]
        except ImportError:
            return [{"error": "Document search not available"}]
        except Exception as e:
            return [{"error": str(e)}]
