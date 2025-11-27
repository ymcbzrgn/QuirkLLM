"""
MCP (Model Context Protocol) module for QuirkLLM.

Provides JSON-RPC 2.0 based protocol implementation for
Claude Desktop integration.

Components:
- protocol.py: JSON-RPC 2.0 message handling
- server.py: MCP server with stdio transport
- tools.py: Tool registry with 8 tools
- config.py: Claude Desktop configuration

Usage:
    # Start as MCP server
    quirkllm --mcp

    # Generate Claude Desktop config
    quirkllm --mcp-config
"""

from quirkllm.mcp.protocol import (
    MCPProtocol,
    MCPRequest,
    MCPResponse,
    MCPError,
    MCPMethod,
)
from quirkllm.mcp.server import MCPServer, ServerConfig
from quirkllm.mcp.tools import ToolRegistry, ToolDefinition
from quirkllm.mcp.config import (
    generate_mcp_config,
    install_config,
    check_installation,
    get_claude_config_path,
)

__all__ = [
    # Protocol
    "MCPProtocol",
    "MCPRequest",
    "MCPResponse",
    "MCPError",
    "MCPMethod",
    # Server
    "MCPServer",
    "ServerConfig",
    # Tools
    "ToolRegistry",
    "ToolDefinition",
    # Config
    "generate_mcp_config",
    "install_config",
    "check_installation",
    "get_claude_config_path",
]
