"""
MCP Client implementation for Mirix - adapted from Letta's structure
"""

from .exceptions import MCPTimeoutError, MCPConnectionError, MCPNotInitializedError
from .types import MCPTool, BaseServerConfig, StdioServerConfig, SSEServerConfig, GmailServerConfig, MCPServerType
from .base_client import BaseMCPClient, BaseAsyncMCPClient
from .stdio_client import StdioMCPClient, AsyncStdioMCPClient
from .gmail_client import GmailMCPClient
from .manager import MCPClientManager, AsyncMCPClientManager, get_mcp_client_manager, get_async_mcp_client_manager

__all__ = [
    'MCPTimeoutError',
    'MCPConnectionError', 
    'MCPNotInitializedError',
    'MCPTool', 
    'BaseServerConfig',
    'StdioServerConfig',
    'SSEServerConfig',
    'GmailServerConfig',
    'MCPServerType',
    'BaseMCPClient',
    'BaseAsyncMCPClient',
    'StdioMCPClient',
    'AsyncStdioMCPClient',
    'GmailMCPClient',
    'MCPClientManager',
    'AsyncMCPClientManager',
    'get_mcp_client_manager',
    'get_async_mcp_client_manager'
]