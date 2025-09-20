"""
MCP Client Exceptions
"""


class MCPTimeoutError(RuntimeError):
    """Custom exception raised when an MCP operation times out."""

    def __init__(self, operation: str, server_name: str, timeout: float):
        message = f"Timed out while {operation} for MCP server {server_name} (timeout={timeout}s)."
        super().__init__(message)


class MCPConnectionError(RuntimeError):
    """Custom exception raised when MCP connection fails."""

    def __init__(self, server_name: str, details: str = ""):
        message = f"Failed to connect to MCP server {server_name}"
        if details:
            message += f": {details}"
        super().__init__(message)


class MCPNotInitializedError(RuntimeError):
    """Custom exception raised when MCP client is not initialized."""

    def __init__(self, server_name: str = ""):
        message = "MCP client has not been initialized"
        if server_name:
            message += f" for server {server_name}"
        super().__init__(message)