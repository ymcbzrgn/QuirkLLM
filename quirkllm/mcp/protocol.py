"""
MCP Protocol Handler - JSON-RPC 2.0 implementation.

Implements Model Context Protocol for Claude Desktop integration.
Reference: https://modelcontextprotocol.io

Example:
    >>> protocol = MCPProtocol()
    >>> request = protocol.parse_request('{"jsonrpc": "2.0", "method": "ping", "id": 1}')
    >>> response = protocol.handle_request(request)
    >>> print(protocol.serialize_response(response))
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Callable, Union
from enum import Enum


class MCPMethod(Enum):
    """MCP protocol methods."""

    # Lifecycle
    INITIALIZE = "initialize"
    INITIALIZED = "notifications/initialized"
    SHUTDOWN = "shutdown"

    # Tools
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"

    # Resources
    LIST_RESOURCES = "resources/list"
    READ_RESOURCE = "resources/read"

    # Utility
    PING = "ping"


@dataclass
class MCPRequest:
    """
    JSON-RPC 2.0 request message.

    Attributes:
        jsonrpc: Protocol version (always "2.0")
        method: Method name to call
        id: Request identifier (None for notifications)
        params: Optional method parameters
    """

    jsonrpc: str
    method: str
    id: Optional[Union[int, str]] = None
    params: Optional[Dict[str, Any]] = None

    def is_notification(self) -> bool:
        """Check if this is a notification (no id)."""
        return self.id is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
        }
        if self.id is not None:
            result["id"] = self.id
        if self.params is not None:
            result["params"] = self.params
        return result


@dataclass
class MCPResponse:
    """
    JSON-RPC 2.0 response message.

    Attributes:
        jsonrpc: Protocol version (always "2.0")
        id: Request identifier (matches request)
        result: Success result (mutually exclusive with error)
        error: Error object (mutually exclusive with result)
    """

    jsonrpc: str
    id: Optional[Union[int, str]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

    def is_error(self) -> bool:
        """Check if this is an error response."""
        return self.error is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            result["id"] = self.id
        if self.error is not None:
            result["error"] = self.error
        else:
            result["result"] = self.result
        return result


@dataclass
class MCPError:
    """
    JSON-RPC 2.0 error object.

    Attributes:
        code: Integer error code
        message: Human-readable error message
        data: Optional additional error data
    """

    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.data is not None:
            result["data"] = self.data
        return result


class MCPProtocol:
    """
    MCP protocol handler.

    Handles JSON-RPC 2.0 message parsing, serialization, and routing.

    Attributes:
        handlers: Registered method handlers
        server_info: Server identification info
        capabilities: Server capabilities

    Example:
        >>> protocol = MCPProtocol()
        >>> protocol.register_handler("ping", lambda p: {})
        >>> req = protocol.parse_request('{"jsonrpc": "2.0", "method": "ping", "id": 1}')
        >>> resp = protocol.handle_request(req)
    """

    # Standard JSON-RPC 2.0 error codes
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # MCP-specific error codes
    TOOL_NOT_FOUND = -32000
    RESOURCE_NOT_FOUND = -32001
    PERMISSION_DENIED = -32002

    def __init__(self) -> None:
        """Initialize protocol handler."""
        self.handlers: Dict[str, Callable] = {}
        self.server_info = {
            "name": "quirkllm",
            "version": "0.1.0",
        }
        self.capabilities = {
            "tools": {},
            "resources": {},
        }

    def register_handler(self, method: str, handler: Callable) -> None:
        """
        Register a method handler.

        Args:
            method: Method name (e.g., "tools/list")
            handler: Callable that takes params dict and returns result
        """
        self.handlers[method] = handler

    def unregister_handler(self, method: str) -> bool:
        """
        Unregister a method handler.

        Args:
            method: Method name to unregister

        Returns:
            True if handler was removed, False if not found
        """
        if method in self.handlers:
            del self.handlers[method]
            return True
        return False

    def parse_request(self, data: str) -> MCPRequest:
        """
        Parse JSON-RPC request from string.

        Args:
            data: JSON string containing the request

        Returns:
            MCPRequest object

        Raises:
            ValueError: If JSON is invalid or missing required fields
        """
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        if not isinstance(parsed, dict):
            raise ValueError("Request must be a JSON object")

        # Validate required fields
        if "jsonrpc" not in parsed:
            raise ValueError("Missing required field: jsonrpc")
        if parsed["jsonrpc"] != "2.0":
            raise ValueError(f"Unsupported JSON-RPC version: {parsed['jsonrpc']}")
        if "method" not in parsed:
            raise ValueError("Missing required field: method")
        if not isinstance(parsed["method"], str):
            raise ValueError("Method must be a string")

        return MCPRequest(
            jsonrpc=parsed["jsonrpc"],
            method=parsed["method"],
            id=parsed.get("id"),
            params=parsed.get("params"),
        )

    def serialize_response(self, response: MCPResponse) -> str:
        """
        Serialize response to JSON string.

        Args:
            response: MCPResponse object

        Returns:
            JSON string
        """
        return json.dumps(response.to_dict(), ensure_ascii=False)

    def handle_request(self, request: MCPRequest) -> MCPResponse:
        """
        Handle incoming request and return response.

        Args:
            request: MCPRequest to handle

        Returns:
            MCPResponse with result or error
        """
        # Check if handler exists
        if request.method not in self.handlers:
            return self.create_error(
                self.METHOD_NOT_FOUND,
                f"Method not found: {request.method}",
                request.id,
            )

        # Call handler
        handler = self.handlers[request.method]
        try:
            params = request.params or {}
            result = handler(params)
            return MCPResponse(
                jsonrpc="2.0",
                id=request.id,
                result=result,
            )
        except TypeError as e:
            return self.create_error(
                self.INVALID_PARAMS,
                f"Invalid params: {e}",
                request.id,
            )
        except Exception as e:
            return self.create_error(
                self.INTERNAL_ERROR,
                f"Internal error: {e}",
                request.id,
            )

    def create_error(
        self,
        code: int,
        message: str,
        id: Optional[Union[int, str]] = None,
        data: Optional[Any] = None,
    ) -> MCPResponse:
        """
        Create an error response.

        Args:
            code: Error code
            message: Error message
            id: Request ID (if known)
            data: Additional error data

        Returns:
            MCPResponse with error
        """
        error = MCPError(code=code, message=message, data=data)
        return MCPResponse(
            jsonrpc="2.0",
            id=id,
            error=error.to_dict(),
        )

    def create_success(
        self,
        result: Any,
        id: Optional[Union[int, str]] = None,
    ) -> MCPResponse:
        """
        Create a success response.

        Args:
            result: Result data
            id: Request ID

        Returns:
            MCPResponse with result
        """
        return MCPResponse(
            jsonrpc="2.0",
            id=id,
            result=result,
        )
