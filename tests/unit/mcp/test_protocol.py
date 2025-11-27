"""
Tests for MCP Protocol Handler.

Test Categories:
1. Request Parsing Tests (4)
2. Response Serialization Tests (3)
3. Handler Tests (5)

Total: 12 tests
"""

import json
import pytest

from quirkllm.mcp.protocol import (
    MCPProtocol,
    MCPRequest,
    MCPResponse,
    MCPError,
    MCPMethod,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def protocol():
    """Create a fresh protocol handler."""
    return MCPProtocol()


@pytest.fixture
def sample_handler():
    """Create a sample handler that returns params."""
    def handler(params):
        return {"received": params}
    return handler


# =============================================================================
# 1. Request Parsing Tests (4)
# =============================================================================


class TestRequestParsing:
    """Tests for JSON-RPC request parsing."""

    def test_parse_valid_request(self, protocol):
        """Test parsing a valid JSON-RPC request."""
        data = '{"jsonrpc": "2.0", "method": "ping", "id": 1}'
        request = protocol.parse_request(data)

        assert request.jsonrpc == "2.0"
        assert request.method == "ping"
        assert request.id == 1
        assert request.params is None

    def test_parse_request_with_params(self, protocol):
        """Test parsing request with parameters."""
        data = '{"jsonrpc": "2.0", "method": "tools/call", "id": 2, "params": {"name": "test"}}'
        request = protocol.parse_request(data)

        assert request.method == "tools/call"
        assert request.params == {"name": "test"}

    def test_parse_invalid_json(self, protocol):
        """Test parsing invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            protocol.parse_request("not valid json")

    def test_parse_missing_jsonrpc(self, protocol):
        """Test parsing request without jsonrpc field."""
        data = '{"method": "ping", "id": 1}'
        with pytest.raises(ValueError, match="Missing required field: jsonrpc"):
            protocol.parse_request(data)

    def test_parse_missing_method(self, protocol):
        """Test parsing request without method field."""
        data = '{"jsonrpc": "2.0", "id": 1}'
        with pytest.raises(ValueError, match="Missing required field: method"):
            protocol.parse_request(data)

    def test_parse_notification(self, protocol):
        """Test parsing a notification (no id)."""
        data = '{"jsonrpc": "2.0", "method": "notifications/initialized"}'
        request = protocol.parse_request(data)

        assert request.id is None
        assert request.is_notification() is True


# =============================================================================
# 2. Response Serialization Tests (3)
# =============================================================================


class TestResponseSerialization:
    """Tests for JSON-RPC response serialization."""

    def test_serialize_success_response(self, protocol):
        """Test serializing a success response."""
        response = MCPResponse(
            jsonrpc="2.0",
            id=1,
            result={"status": "ok"},
        )
        serialized = protocol.serialize_response(response)
        parsed = json.loads(serialized)

        assert parsed["jsonrpc"] == "2.0"
        assert parsed["id"] == 1
        assert parsed["result"] == {"status": "ok"}
        assert "error" not in parsed

    def test_serialize_error_response(self, protocol):
        """Test serializing an error response."""
        response = protocol.create_error(
            code=-32601,
            message="Method not found",
            id=2,
        )
        serialized = protocol.serialize_response(response)
        parsed = json.loads(serialized)

        assert parsed["jsonrpc"] == "2.0"
        assert parsed["id"] == 2
        assert parsed["error"]["code"] == -32601
        assert parsed["error"]["message"] == "Method not found"

    def test_serialize_notification_response(self, protocol):
        """Test serializing a response without id."""
        response = MCPResponse(
            jsonrpc="2.0",
            result={},
        )
        serialized = protocol.serialize_response(response)
        parsed = json.loads(serialized)

        assert "id" not in parsed or parsed["id"] is None


# =============================================================================
# 3. Handler Tests (5)
# =============================================================================


class TestHandlers:
    """Tests for protocol handler registration and execution."""

    def test_register_handler(self, protocol, sample_handler):
        """Test registering a method handler."""
        protocol.register_handler("test/method", sample_handler)

        assert "test/method" in protocol.handlers

    def test_handle_registered_method(self, protocol, sample_handler):
        """Test handling a request for registered method."""
        protocol.register_handler("echo", sample_handler)
        request = MCPRequest(
            jsonrpc="2.0",
            method="echo",
            id=1,
            params={"message": "hello"},
        )

        response = protocol.handle_request(request)

        assert response.result == {"received": {"message": "hello"}}
        assert response.error is None

    def test_handle_unknown_method(self, protocol):
        """Test handling request for unknown method."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="unknown/method",
            id=1,
        )

        response = protocol.handle_request(request)

        assert response.error is not None
        assert response.error["code"] == MCPProtocol.METHOD_NOT_FOUND

    def test_handle_with_params(self, protocol):
        """Test handling request with parameters."""
        def add_handler(params):
            return params.get("a", 0) + params.get("b", 0)

        protocol.register_handler("add", add_handler)
        request = MCPRequest(
            jsonrpc="2.0",
            method="add",
            id=1,
            params={"a": 2, "b": 3},
        )

        response = protocol.handle_request(request)

        assert response.result == 5

    def test_handle_without_params(self, protocol):
        """Test handling request without parameters."""
        def ping_handler(params):
            return "pong"

        protocol.register_handler("ping", ping_handler)
        request = MCPRequest(
            jsonrpc="2.0",
            method="ping",
            id=1,
        )

        response = protocol.handle_request(request)

        assert response.result == "pong"

    def test_handler_exception_returns_error(self, protocol):
        """Test that handler exceptions return error response."""
        def error_handler(params):
            raise RuntimeError("Something went wrong")

        protocol.register_handler("fail", error_handler)
        request = MCPRequest(
            jsonrpc="2.0",
            method="fail",
            id=1,
        )

        response = protocol.handle_request(request)

        assert response.error is not None
        assert response.error["code"] == MCPProtocol.INTERNAL_ERROR


# =============================================================================
# Additional Tests
# =============================================================================


class TestMCPDataclasses:
    """Tests for MCP dataclass functionality."""

    def test_request_to_dict(self):
        """Test MCPRequest.to_dict()."""
        request = MCPRequest(
            jsonrpc="2.0",
            method="test",
            id=1,
            params={"key": "value"},
        )
        result = request.to_dict()

        assert result["jsonrpc"] == "2.0"
        assert result["method"] == "test"
        assert result["id"] == 1
        assert result["params"] == {"key": "value"}

    def test_response_to_dict(self):
        """Test MCPResponse.to_dict()."""
        response = MCPResponse(
            jsonrpc="2.0",
            id=1,
            result={"status": "ok"},
        )
        result = response.to_dict()

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1
        assert result["result"] == {"status": "ok"}

    def test_error_to_dict(self):
        """Test MCPError.to_dict()."""
        error = MCPError(
            code=-32600,
            message="Invalid Request",
            data={"details": "missing field"},
        )
        result = error.to_dict()

        assert result["code"] == -32600
        assert result["message"] == "Invalid Request"
        assert result["data"] == {"details": "missing field"}

    def test_response_is_error(self):
        """Test MCPResponse.is_error()."""
        success = MCPResponse(jsonrpc="2.0", id=1, result={})
        error = MCPResponse(
            jsonrpc="2.0",
            id=1,
            error={"code": -1, "message": "error"},
        )

        assert success.is_error() is False
        assert error.is_error() is True


class TestMCPMethod:
    """Tests for MCPMethod enum."""

    def test_method_values(self):
        """Test MCPMethod enum values."""
        assert MCPMethod.INITIALIZE.value == "initialize"
        assert MCPMethod.LIST_TOOLS.value == "tools/list"
        assert MCPMethod.CALL_TOOL.value == "tools/call"
        assert MCPMethod.PING.value == "ping"
