"""
Tests for MCP Server.

Test Categories:
1. Server Lifecycle Tests (2)
2. Protocol Handler Tests (5)
3. Error Handling Tests (3)

Total: 10 tests
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch

from quirkllm.mcp.server import MCPServer, ServerConfig
from quirkllm.mcp.protocol import MCPRequest, MCPResponse


def run_async(coro):
    """Helper to run async code in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def server():
    """Create a fresh MCP server."""
    return MCPServer()


@pytest.fixture
def server_with_config():
    """Create server with custom config."""
    config = ServerConfig(
        name="test-server",
        version="1.0.0",
        log_level="DEBUG",
    )
    return MCPServer(config)


@pytest.fixture
def mock_tools():
    """Create mock tool registry."""
    tools = Mock()
    tools.list_tools.return_value = [
        {"name": "test_tool", "description": "A test tool", "inputSchema": {}},
    ]
    tools.call_tool = AsyncMock(return_value={"content": [{"type": "text", "text": "result"}]})
    return tools


# =============================================================================
# 1. Server Lifecycle Tests (2)
# =============================================================================


class TestServerLifecycle:
    """Tests for server initialization and lifecycle."""

    def test_server_init(self, server):
        """Test server initializes with default config."""
        assert server.config.name == "quirkllm"
        assert server.config.version == "0.1.0"
        assert server._running is False
        assert server._initialized is False

    def test_server_init_custom_config(self, server_with_config):
        """Test server initializes with custom config."""
        assert server_with_config.config.name == "test-server"
        assert server_with_config.config.version == "1.0.0"
        assert server_with_config.config.log_level == "DEBUG"

    def test_server_set_tools(self, server, mock_tools):
        """Test setting tool registry."""
        server.set_tools(mock_tools)
        assert server._tools is mock_tools


# =============================================================================
# 2. Protocol Handler Tests (5)
# =============================================================================


class TestProtocolHandlers:
    """Tests for MCP protocol handlers."""

    def test_initialize_handler(self, server):
        """Test initialize request handler."""
        params = {
            "clientInfo": {"name": "test-client"},
            "protocolVersion": "2024-11-05",
        }

        result = server._handle_initialize(params)

        assert "protocolVersion" in result
        assert "capabilities" in result
        assert "serverInfo" in result
        assert result["serverInfo"]["name"] == "quirkllm"

    def test_initialized_handler(self, server):
        """Test initialized notification handler."""
        assert server._initialized is False

        server._handle_initialized({})

        assert server._initialized is True

    def test_ping_handler(self, server):
        """Test ping request handler."""
        result = server._handle_ping({})

        assert result == {}

    def test_shutdown_handler(self, server):
        """Test shutdown request handler."""
        server._running = True

        result = server._handle_shutdown({})

        assert result == {}
        assert server._running is False

    def test_list_tools_handler(self, server, mock_tools):
        """Test tools/list request handler."""
        server.set_tools(mock_tools)

        result = server._handle_list_tools({})

        assert "tools" in result
        assert len(result["tools"]) == 1
        mock_tools.list_tools.assert_called_once()

    def test_call_tool_handler(self, server, mock_tools):
        """Test tools/call request handler."""
        server.set_tools(mock_tools)

        result = run_async(server._handle_call_tool({
            "name": "test_tool",
            "arguments": {"arg1": "value1"},
        }))

        assert "content" in result
        mock_tools.call_tool.assert_called_once_with("test_tool", {"arg1": "value1"})


# =============================================================================
# 3. Error Handling Tests (3)
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_handle_request_async_unknown_method(self, server):
        """Test handling unknown method returns error."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="unknown/method",
            id=1,
        )

        response = run_async(server._handle_request_async(request))

        assert response.error is not None
        assert response.error["code"] == -32601  # METHOD_NOT_FOUND

    def test_call_tool_no_tools(self, server):
        """Test tools/call with no tools registered."""
        result = run_async(server._handle_call_tool({"name": "test"}))

        assert result["isError"] is True
        assert "No tools available" in result["content"][0]["text"]

    def test_call_tool_missing_name(self, server, mock_tools):
        """Test tools/call with missing tool name."""
        server.set_tools(mock_tools)

        result = run_async(server._handle_call_tool({}))

        assert result["isError"] is True
        assert "Missing tool name" in result["content"][0]["text"]


# =============================================================================
# Additional Tests
# =============================================================================


class TestServerConfig:
    """Tests for ServerConfig dataclass."""

    def test_default_config(self):
        """Test default ServerConfig values."""
        config = ServerConfig()

        assert config.name == "quirkllm"
        assert config.version == "0.1.0"
        assert config.log_level == "INFO"

    def test_custom_config(self):
        """Test custom ServerConfig values."""
        config = ServerConfig(
            name="custom",
            version="2.0.0",
            log_level="DEBUG",
        )

        assert config.name == "custom"
        assert config.version == "2.0.0"
        assert config.log_level == "DEBUG"


class TestMessageProcessing:
    """Tests for message processing."""

    def test_process_valid_message(self, server):
        """Test processing a valid JSON-RPC message."""
        message = '{"jsonrpc": "2.0", "method": "ping", "id": 1}'

        response = run_async(server._process_message(message))
        assert response is not None

        import json
        parsed = json.loads(response)
        assert parsed["jsonrpc"] == "2.0"
        assert parsed["id"] == 1
        assert parsed["result"] == {}

    def test_process_invalid_json(self, server):
        """Test processing invalid JSON returns error."""
        message = "not valid json"

        response = run_async(server._process_message(message))
        assert response is not None

        import json
        parsed = json.loads(response)
        assert "error" in parsed
        assert parsed["error"]["code"] == -32700  # PARSE_ERROR

    def test_process_notification_no_response(self, server):
        """Test processing notification returns None."""
        message = '{"jsonrpc": "2.0", "method": "notifications/initialized"}'

        response = run_async(server._process_message(message))

        # Notifications don't get responses
        assert response is None
