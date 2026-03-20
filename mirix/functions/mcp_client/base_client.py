"""
Base MCP Client - adapted from Letta structure with working implementation
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession
from mcp.types import TextContent

from mirix.observability.context import get_trace_context
from mirix.observability.langfuse_client import get_langfuse_client

from .exceptions import MCPConnectionError, MCPNotInitializedError
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
                raise MCPConnectionError(self.server_config.server_name, "Failed to establish connection")
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

        # Get Langfuse client for tracing
        langfuse = get_langfuse_client()
        trace_context = get_trace_context() if langfuse else {}
        trace_id = trace_context.get("trace_id") if trace_context else None
        parent_span_id = trace_context.get("observation_id") if trace_context else None

        async def _do_execute() -> Tuple[str, bool]:
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

            return final_content, getattr(result, "isError", False)

        # Execute with Langfuse tracing if available
        if langfuse and trace_id:
            from typing import cast

            from langfuse.types import TraceContext

            # Build trace context
            trace_context_dict: dict = {"trace_id": trace_id}
            if parent_span_id:
                trace_context_dict["parent_span_id"] = parent_span_id

            # Sanitize args for tracing
            args_for_trace = {key: str(value) for key, value in tool_args.items()}

            try:
                with langfuse.start_as_current_observation(
                    name=f"mcp_tool: {tool_name}",
                    as_type="tool",
                    trace_context=cast(TraceContext, trace_context_dict),
                    input={
                        "tool_name": tool_name,
                        "server": self.server_config.server_name,
                        "args": args_for_trace,
                    },
                    metadata={
                        "mcp_server": self.server_config.server_name,
                        "tool_name": tool_name,
                    },
                ) as span:
                    final_content, is_error = await _do_execute()

                    span.update(
                        output={"response": final_content, "is_error": is_error},
                        level="ERROR" if is_error else "DEFAULT",
                    )
                    return final_content, is_error
            except Exception as e:
                logger.debug(f"Langfuse MCP tool trace failed: {e}")
                return await _do_execute()
        else:
            return await _do_execute()

    def _check_initialized(self):
        """Check if the async client has been initialized"""
        if not self.initialized:
            raise MCPNotInitializedError(self.server_config.server_name)

    async def cleanup(self):
        """Asynchronously clean up client resources"""
        try:
            for cleanup_func in self.cleanup_funcs:
                await cleanup_func()
            self.initialized = False
            logger.info(f"Cleaned up async MCP client for {self.server_config.server_name}")
        except Exception as e:
            logger.warning(f"Error during async cleanup for {self.server_config.server_name}: {e}")
