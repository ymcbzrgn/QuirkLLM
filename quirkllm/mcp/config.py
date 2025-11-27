"""
MCP Configuration for Claude Desktop.

Generates and installs configuration for Claude Desktop's MCP integration.

Config locations:
- macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
- Windows: %APPDATA%/Claude/claude_desktop_config.json
- Linux: ~/.config/Claude/claude_desktop_config.json

Example:
    >>> from quirkllm.mcp.config import install_config
    >>> path = install_config()
    >>> print(f"Config installed to: {path}")
"""

import json
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, Optional


def get_claude_config_path() -> Path:
    """
    Get Claude Desktop config file path.

    Returns:
        Path to claude_desktop_config.json

    Example:
        >>> path = get_claude_config_path()
        >>> print(path)
        # macOS: /Users/.../Library/Application Support/Claude/claude_desktop_config.json
    """
    if sys.platform == "darwin":  # macOS
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    elif sys.platform == "win32":  # Windows
        return (
            Path.home()
            / "AppData"
            / "Roaming"
            / "Claude"
            / "claude_desktop_config.json"
        )
    else:  # Linux and others
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def get_quirkllm_path() -> str:
    """
    Get the path to quirkllm executable.

    Returns:
        Path string suitable for MCP config
    """
    # Try to find quirkllm in PATH
    quirkllm_path = shutil.which("quirkllm")
    if quirkllm_path:
        return quirkllm_path

    # Try common virtual environment locations
    venv_paths = [
        Path.home() / ".local" / "bin" / "quirkllm",
        Path.cwd() / ".venv" / "bin" / "quirkllm",
        Path.cwd() / "venv" / "bin" / "quirkllm",
    ]

    for venv_path in venv_paths:
        if venv_path.exists():
            return str(venv_path)

    # Fallback to assuming it's in PATH
    return "quirkllm"


def generate_mcp_config() -> Dict[str, Any]:
    """
    Generate MCP server configuration for Claude Desktop.

    Returns:
        Dict with mcpServers configuration

    Example:
        >>> config = generate_mcp_config()
        >>> print(json.dumps(config, indent=2))
    """
    quirkllm_cmd = get_quirkllm_path()

    return {
        "mcpServers": {
            "quirkllm": {
                "command": quirkllm_cmd,
                "args": ["--mcp"],
                "env": {},
            }
        }
    }


def load_existing_config(config_path: Path) -> Dict[str, Any]:
    """
    Load existing Claude Desktop config.

    Args:
        config_path: Path to config file

    Returns:
        Existing config or empty dict
    """
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def install_config(merge: bool = True) -> Path:
    """
    Install MCP configuration to Claude Desktop.

    Args:
        merge: If True, merge with existing config. If False, overwrite.

    Returns:
        Path to config file

    Example:
        >>> path = install_config()
        >>> print(f"Config installed to: {path}")
    """
    config_path = get_claude_config_path()

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate new config
    new_config = generate_mcp_config()

    if merge:
        # Load and merge with existing config
        existing_config = load_existing_config(config_path)

        # Merge mcpServers
        if "mcpServers" not in existing_config:
            existing_config["mcpServers"] = {}

        existing_config["mcpServers"]["quirkllm"] = new_config["mcpServers"]["quirkllm"]
        final_config = existing_config
    else:
        final_config = new_config

    # Write config
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=2, ensure_ascii=False)

    return config_path


def uninstall_config() -> bool:
    """
    Remove QuirkLLM from Claude Desktop config.

    Returns:
        True if removed, False if not found

    Example:
        >>> success = uninstall_config()
        >>> print("Removed" if success else "Not found")
    """
    config_path = get_claude_config_path()

    if not config_path.exists():
        return False

    try:
        config = load_existing_config(config_path)

        if "mcpServers" not in config:
            return False

        if "quirkllm" not in config["mcpServers"]:
            return False

        # Remove quirkllm entry
        del config["mcpServers"]["quirkllm"]

        # Write updated config
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return True

    except Exception:
        return False


def check_installation() -> Dict[str, Any]:
    """
    Check QuirkLLM MCP installation status.

    Returns:
        Dict with:
        - installed: bool
        - config_path: str
        - quirkllm_path: str
        - errors: list of error messages

    Example:
        >>> status = check_installation()
        >>> if status["installed"]:
        ...     print("QuirkLLM MCP is configured!")
    """
    result = {
        "installed": False,
        "config_path": str(get_claude_config_path()),
        "quirkllm_path": get_quirkllm_path(),
        "errors": [],
    }

    config_path = get_claude_config_path()

    # Check if config file exists
    if not config_path.exists():
        result["errors"].append("Claude Desktop config file not found")
        return result

    # Check if quirkllm is in config
    try:
        config = load_existing_config(config_path)

        if "mcpServers" not in config:
            result["errors"].append("No MCP servers configured")
            return result

        if "quirkllm" not in config["mcpServers"]:
            result["errors"].append("QuirkLLM not configured as MCP server")
            return result

        # Check if quirkllm command exists
        quirkllm_cmd = config["mcpServers"]["quirkllm"].get("command", "")
        if quirkllm_cmd != "quirkllm" and not Path(quirkllm_cmd).exists():
            result["errors"].append(f"QuirkLLM not found at: {quirkllm_cmd}")
            return result

        result["installed"] = True

    except Exception as e:
        result["errors"].append(f"Error reading config: {e}")

    return result


def get_config_info() -> Dict[str, Any]:
    """
    Get detailed configuration information.

    Returns:
        Dict with config details and diagnostics
    """
    config_path = get_claude_config_path()

    info = {
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "config_dir_exists": config_path.parent.exists(),
        "platform": sys.platform,
        "quirkllm_path": get_quirkllm_path(),
        "quirkllm_exists": shutil.which("quirkllm") is not None,
    }

    if config_path.exists():
        try:
            config = load_existing_config(config_path)
            info["mcp_servers"] = list(config.get("mcpServers", {}).keys())
            info["quirkllm_configured"] = "quirkllm" in config.get("mcpServers", {})
        except Exception as e:
            info["config_error"] = str(e)

    return info
