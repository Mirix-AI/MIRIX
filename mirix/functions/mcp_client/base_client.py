"""
Base MCP Client - adapted from Letta structure with working implementation
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any

from mcp import ClientSession
from mcp.types import TextContent

from .exceptions import MCPTimeoutError, MCPConnectionError, MCPNotInitializedError
from .types import BaseServerConfig, MCPTool

logger = logging.getLogger(__name__)

# Default timeouts (can be overridden)
DEFAULT_CONNECT_TIMEOUT = 30.0
DEFAULT_INITIALIZE_TIMEOUT = 30.0
DEFAULT_LIST_TOOLS_TIMEOUT = 10.0
DEFAULT_EXECUTE_TOOL_TIMEOUT = 60.0


class BaseMCPClient(ABC):
    """Base class for MCP clients with different transport methods"""

    def __init__(self, server_config: BaseServerConfig):
        self.server_config = server_config
        self.session: Optional[ClientSession] = None
        self.stdio = None
        self.write = None
        self.initialized = False
        self.loop = asyncio.new_event_loop()
        self.cleanup_funcs = []

    def connect_to_server(self, connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
                          initialize_timeout: float = DEFAULT_INITIALIZE_TIMEOUT):
        """Connect to the MCP server and initialize the session"""
        asyncio.set_event_loop(self.loop)
        
        try:
            success = self._initialize_connection(self.server_config, timeout=connect_timeout)
            
            if success:
                try:
                    self.loop.run_until_complete(
                        asyncio.wait_for(self.session.initialize(), timeout=initialize_timeout)
                    )
                    self.initialized = True
                    logger.info(f"Successfully connected to MCP server: {self.server_config.server_name}")
                except asyncio.TimeoutError:
                    raise MCPTimeoutError("initializing session", self.server_config.server_name, initialize_timeout)
            else:
                raise MCPConnectionError(
                    self.server_config.server_name,
                    "Failed to establish connection"
                )
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.server_config.server_name}: {str(e)}")
            raise

    @abstractmethod
    def _initialize_connection(self, server_config: BaseServerConfig, timeout: float) -> bool:
        """Initialize the connection - implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _initialize_connection")

    def list_tools(self, timeout: float = DEFAULT_LIST_TOOLS_TIMEOUT) -> List[MCPTool]:
        """List available tools from the MCP server"""
        self._check_initialized()
        
        try:
            response = self.loop.run_until_complete(
                asyncio.wait_for(self.session.list_tools(), timeout=timeout)
            )
            return response.tools
        except asyncio.TimeoutError:
            logger.error(f"Timed out while listing tools for MCP server {self.server_config.server_name}")
            raise MCPTimeoutError("listing tools", self.server_config.server_name, timeout)

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any], 
                     timeout: float = DEFAULT_EXECUTE_TOOL_TIMEOUT) -> Tuple[str, bool]:
        """Execute a tool on the MCP server"""
        self._check_initialized()
        
        try:
            result = self.loop.run_until_complete(
                asyncio.wait_for(
                    self.session.call_tool(tool_name, tool_args), 
                    timeout=timeout
                )
            )

            # Parse the content from the result
            parsed_content = []
            for content_piece in result.content:
                if isinstance(content_piece, TextContent):
                    parsed_content.append(content_piece.text)
                else:
                    parsed_content.append(str(content_piece))

            if len(parsed_content) > 0:
                final_content = " ".join(parsed_content)
            else:
                final_content = "Empty response from tool"

            # Return content and whether there was an error
            return final_content, getattr(result, 'isError', False)
            
        except asyncio.TimeoutError:
            logger.error(f"Timed out while executing tool '{tool_name}' for MCP server {self.server_config.server_name}")
            raise MCPTimeoutError(f"executing tool '{tool_name}'", self.server_config.server_name, timeout)

    def _check_initialized(self):
        """Check if the client has been initialized"""
        if not self.initialized:
            raise MCPNotInitializedError(self.server_config.server_name)

    def cleanup(self):
        """Clean up the client resources"""
        try:
            for cleanup_func in self.cleanup_funcs:
                cleanup_func()
            self.initialized = False
            if not self.loop.is_closed():
                self.loop.close()
            logger.info(f"Cleaned up MCP client for {self.server_config.server_name}")
        except Exception as e:
            logger.warning(f"Error during cleanup for {self.server_config.server_name}: {e}")


class BaseAsyncMCPClient(ABC):
    """Async version of the base MCP client"""

    def __init__(self, server_config: BaseServerConfig):
        self.server_config = server_config
        self.session: Optional[ClientSession] = None
        self.stdio = None
        self.write = None
        self.initialized = False
        self.cleanup_funcs = []

    async def connect_to_server(self, timeout: float = DEFAULT_CONNECT_TIMEOUT):
        """Asynchronously connect to the MCP server"""
        try:
            success = await self._initialize_connection(self.server_config, timeout=timeout)
            
            if success:
                await self.session.initialize()
                self.initialized = True
                logger.info(f"Successfully connected to MCP server: {self.server_config.server_name}")
            else:
                raise MCPConnectionError(
                    self.server_config.server_name,
                    "Failed to establish connection"
                )
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.server_config.server_name}: {str(e)}")
            raise

    @abstractmethod
    async def _initialize_connection(self, server_config: BaseServerConfig, timeout: float) -> bool:
        """Asynchronously initialize the connection"""
        raise NotImplementedError("Subclasses must implement _initialize_connection")

    async def list_tools(self) -> List[MCPTool]:
        """Asynchronously list available tools"""
        self._check_initialized()
        response = await self.session.list_tools()
        return response.tools

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[str, bool]:
        """Asynchronously execute a tool"""
        self._check_initialized()
        result = await self.session.call_tool(tool_name, tool_args)

        parsed_content = []
        for content_piece in result.content:
            if isinstance(content_piece, TextContent):
                parsed_content.append(content_piece.text)
            else:
                parsed_content.append(str(content_piece))

        if len(parsed_content) > 0:
            final_content = " ".join(parsed_content)
        else:
            final_content = "Empty response from tool"

        return final_content, getattr(result, 'isError', False)

    def _check_initialized(self):
        """Check if the async client has been initialized"""
        if not self.initialized:
            raise MCPNotInitializedError(self.server_config.server_name)

    async def cleanup(self):
        """Asynchronously clean up client resources"""
        try:
            for cleanup_func in self.cleanup_funcs:
                if asyncio.iscoroutinefunction(cleanup_func):
                    await cleanup_func()
                else:
                    cleanup_func()
            self.initialized = False
            logger.info(f"Cleaned up async MCP client for {self.server_config.server_name}")
        except Exception as e:
            logger.warning(f"Error during async cleanup for {self.server_config.server_name}: {e}")