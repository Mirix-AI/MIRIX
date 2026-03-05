"""
MCP Client implementation for Mirix - adapted from Letta's structure
"""

from .base_client import BaseMCPClient
from .exceptions import MCPConnectionError, MCPNotInitializedError, MCPTimeoutError
from .gmail_client import GmailMCPClient
from .manager import MCPClientManager, get_mcp_client_manager
from .stdio_client import StdioMCPClient
from .types import (
    BaseServerConfig,
    GmailServerConfig,
    MCPServerType,
    MCPTool,
    SSEServerConfig,
    StdioServerConfig,
)

__all__ = [
    "MCPTimeoutError",
    "MCPConnectionError",
    "MCPNotInitializedError",
    "MCPTool",
    "BaseServerConfig",
    "StdioServerConfig",
    "SSEServerConfig",
    "GmailServerConfig",
    "MCPServerType",
    "BaseMCPClient",
    "StdioMCPClient",
    "GmailMCPClient",
    "MCPClientManager",
    "get_mcp_client_manager",
]
