"""
MCP Server - Stdio transport for Claude Desktop.

Runs as a subprocess, communicates via stdin/stdout using JSON-RPC 2.0.

Example:
    # Start server programmatically
    >>> server = MCPServer()
    >>> asyncio.run(server.start())

    # Or via CLI
    $ quirkllm --mcp
"""

import sys
import asyncio
import logging
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field

from quirkllm.mcp.protocol import MCPProtocol, MCPRequest, MCPResponse


# Configure logging to stderr (stdout is for JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """
    MCP server configuration.

    Attributes:
        name: Server name for identification
        version: Server version string
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """

    name: str = "quirkllm"
    version: str = "0.1.0"
    log_level: str = "INFO"


class MCPServer:
    """
    MCP Server with stdio transport.

    Lifecycle:
    1. Start -> Initialize protocol handlers
    2. Read JSON-RPC message from stdin
    3. Parse and route to handler
    4. Write response to stdout
    5. Loop until EOF or shutdown

    Attributes:
        config: Server configuration
        protocol: MCP protocol handler
        tools: Tool registry (set externally)

    Example:
        >>> server = MCPServer()
        >>> asyncio.run(server.start())
    """

    def __init__(self, config: Optional[ServerConfig] = None) -> None:
        """
        Initialize MCP server.

        Args:
            config: Optional server configuration
        """
        self.config = config or ServerConfig()
        self.protocol = MCPProtocol()
        self._running = False
        self._initialized = False
        self._tools: Optional[Any] = None  # Set by _setup_handlers

        # Set logging level
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)

        # Register core handlers
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Register protocol handlers."""
        self.protocol.register_handler("initialize", self._handle_initialize)
        self.protocol.register_handler("notifications/initialized", self._handle_initialized)
        self.protocol.register_handler("shutdown", self._handle_shutdown)
        self.protocol.register_handler("ping", self._handle_ping)
        self.protocol.register_handler("tools/list", self._handle_list_tools)
        self.protocol.register_handler("tools/call", self._handle_call_tool)

    def set_tools(self, tools: Any) -> None:
        """
        Set the tool registry.

        Args:
            tools: ToolRegistry instance
        """
        self._tools = tools

    async def start(self) -> None:
        """
        Start the MCP server (stdio mode).

        Runs the main event loop reading from stdin and
        writing responses to stdout.
        """
        self._running = True
        logger.info(f"Starting MCP server: {self.config.name} v{self.config.version}")

        try:
            while self._running:
                # Read message from stdin
                message = await self._read_message()
                if message is None:
                    # EOF reached
                    logger.info("EOF received, shutting down")
                    break

                # Process message
                response = await self._process_message(message)

                # Write response if not a notification
                if response is not None:
                    await self._write_message(response)

        except asyncio.CancelledError:
            logger.info("Server cancelled")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            self._running = False
            logger.info("MCP server stopped")

    async def stop(self) -> None:
        """Stop the MCP server gracefully."""
        logger.info("Stopping MCP server...")
        self._running = False

    async def _read_message(self) -> Optional[str]:
        """
        Read a JSON-RPC message from stdin.

        MCP uses Content-Length header format:
        Content-Length: <length>\r\n
        \r\n
        <json-content>

        Returns:
            JSON string or None on EOF
        """
        loop = asyncio.get_event_loop()

        try:
            # Read header line
            header_line = await loop.run_in_executor(None, sys.stdin.readline)
            if not header_line:
                return None

            # Parse Content-Length header
            header_line = header_line.strip()
            if not header_line.lower().startswith("content-length:"):
                # Simple JSON line mode (fallback)
                return header_line

            content_length = int(header_line.split(":")[1].strip())

            # Read empty line after header
            await loop.run_in_executor(None, sys.stdin.readline)

            # Read content
            content = await loop.run_in_executor(
                None, lambda: sys.stdin.read(content_length)
            )
            return content

        except Exception as e:
            logger.error(f"Error reading message: {e}")
            return None

    async def _write_message(self, message: str) -> None:
        """
        Write a JSON-RPC message to stdout.

        Uses Content-Length header format for MCP compatibility.

        Args:
            message: JSON string to write
        """
        loop = asyncio.get_event_loop()

        try:
            # Encode message
            content = message.encode("utf-8")
            content_length = len(content)

            # Write with Content-Length header
            output = f"Content-Length: {content_length}\r\n\r\n{message}"

            await loop.run_in_executor(None, lambda: sys.stdout.write(output))
            await loop.run_in_executor(None, sys.stdout.flush)

        except Exception as e:
            logger.error(f"Error writing message: {e}")

    async def _process_message(self, message: str) -> Optional[str]:
        """
        Process incoming JSON-RPC message.

        Args:
            message: JSON string

        Returns:
            Response JSON string or None for notifications
        """
        try:
            # Parse request
            request = self.protocol.parse_request(message)
            logger.debug(f"Received: {request.method}")

            # Handle request
            response = await self._handle_request_async(request)

            # Don't respond to notifications
            if request.is_notification():
                return None

            # Serialize response
            return self.protocol.serialize_response(response)

        except ValueError as e:
            # Parse error
            error_response = self.protocol.create_error(
                MCPProtocol.PARSE_ERROR,
                str(e),
            )
            return self.protocol.serialize_response(error_response)

    async def _handle_request_async(self, request: MCPRequest) -> MCPResponse:
        """
        Handle request with async support.

        Wraps sync handlers to work with async handlers.

        Args:
            request: MCPRequest to handle

        Returns:
            MCPResponse
        """
        # Check if handler exists
        if request.method not in self.protocol.handlers:
            return self.protocol.create_error(
                MCPProtocol.METHOD_NOT_FOUND,
                f"Method not found: {request.method}",
                request.id,
            )

        handler = self.protocol.handlers[request.method]
        params = request.params or {}

        try:
            # Call handler (may be sync or async)
            if asyncio.iscoroutinefunction(handler):
                result = await handler(params)
            else:
                result = handler(params)

            return self.protocol.create_success(result, request.id)

        except TypeError as e:
            return self.protocol.create_error(
                MCPProtocol.INVALID_PARAMS,
                f"Invalid params: {e}",
                request.id,
            )
        except Exception as e:
            logger.exception(f"Handler error: {e}")
            return self.protocol.create_error(
                MCPProtocol.INTERNAL_ERROR,
                f"Internal error: {e}",
                request.id,
            )

    # =========================================================================
    # Protocol Handlers
    # =========================================================================

    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle initialize request.

        Args:
            params: Client capabilities and info

        Returns:
            Server capabilities and info
        """
        logger.info("Initializing MCP session")

        # Extract client info
        client_info = params.get("clientInfo", {})
        logger.info(f"Client: {client_info.get('name', 'unknown')}")

        # Build capabilities
        capabilities: Dict[str, Any] = {}

        # Add tools capability if we have tools
        if self._tools is not None:
            capabilities["tools"] = {}

        return {
            "protocolVersion": "2024-11-05",
            "capabilities": capabilities,
            "serverInfo": {
                "name": self.config.name,
                "version": self.config.version,
            },
        }

    def _handle_initialized(self, params: Dict[str, Any]) -> None:
        """
        Handle initialized notification.

        Called after client receives initialize response.

        Args:
            params: Empty dict
        """
        self._initialized = True
        logger.info("MCP session initialized")

    def _handle_shutdown(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle shutdown request.

        Args:
            params: Empty dict

        Returns:
            Empty result
        """
        logger.info("Shutdown requested")
        self._running = False
        return {}

    def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle ping request.

        Args:
            params: Empty dict

        Returns:
            Empty result (pong)
        """
        return {}

    def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle tools/list request.

        Args:
            params: Optional cursor for pagination

        Returns:
            List of available tools
        """
        if self._tools is None:
            return {"tools": []}

        tools_list = self._tools.list_tools()
        return {"tools": tools_list}

    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle tools/call request.

        Args:
            params: Tool name and arguments

        Returns:
            Tool execution result
        """
        if self._tools is None:
            return {
                "content": [{"type": "text", "text": "No tools available"}],
                "isError": True,
            }

        name = params.get("name")
        arguments = params.get("arguments", {})

        if not name:
            return {
                "content": [{"type": "text", "text": "Missing tool name"}],
                "isError": True,
            }

        logger.info(f"Calling tool: {name}")
        result = await self._tools.call_tool(name, arguments)
        return result
