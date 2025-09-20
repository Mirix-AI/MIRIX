"""
MCP Client Manager - High-level interface for managing multiple MCP servers
"""

import asyncio
import json
import os
import logging
from typing import Dict, List, Optional, Any, Tuple

from .base_client import BaseMCPClient, BaseAsyncMCPClient
from .stdio_client import StdioMCPClient, AsyncStdioMCPClient
from .gmail_client import GmailMCPClient
from .types import BaseServerConfig, StdioServerConfig, SSEServerConfig, GmailServerConfig, MCPTool, MCPServerType
from .exceptions import MCPConnectionError, MCPNotInitializedError

logger = logging.getLogger(__name__)


class MCPClientManager:
    """Manager for multiple MCP clients with different transport types"""

    def __init__(self):
        self.clients: Dict[str, BaseMCPClient] = {}
        self.server_configs: Dict[str, BaseServerConfig] = {}
        self.config_file = os.path.expanduser("~/.mirix/mcp_connections.json")
        
        # Load existing connections on startup
        self._load_persistent_connections()
        
    def add_server(self, server_config: BaseServerConfig, 
                   environment_variables: Optional[Dict[str, str]] = None) -> bool:
        """Add a server configuration and connect to it"""
        try:
            # Resolve any templated environment variables
            server_config.resolve_environment_variables(environment_variables)
            
            # Create appropriate client based on server type
            if server_config.type == MCPServerType.STDIO:
                client = StdioMCPClient(server_config)
            elif server_config.type == MCPServerType.SSE:
                # Import SSE client when needed (not implemented yet)
                raise NotImplementedError("SSE transport not implemented yet")
            elif server_config.type == MCPServerType.GMAIL:
                client = GmailMCPClient(server_config)
            else:
                raise ValueError(f"Unsupported server type: {server_config.type}")
            
            # Connect to the server
            client.connect_to_server()
            
            # Store the client and config
            self.clients[server_config.server_name] = client
            self.server_configs[server_config.server_name] = server_config
            
            # Save configuration to disk for persistence
            self._save_persistent_connections()
            
            logger.info(f"Successfully added MCP server: {server_config.server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add MCP server {server_config.server_name}: {str(e)}")
            return False

    def _add_server_without_persistence(self, server_config: BaseServerConfig, 
                                       environment_variables: Optional[Dict[str, str]] = None) -> bool:
        """Add a server configuration and connect to it without saving to disk (used for restoration)"""
        try:
            # Resolve any templated environment variables
            server_config.resolve_environment_variables(environment_variables)
            
            # Create appropriate client based on server type
            if server_config.type == MCPServerType.STDIO:
                client = StdioMCPClient(server_config)
            elif server_config.type == MCPServerType.SSE:
                # Import SSE client when needed (not implemented yet)
                raise NotImplementedError("SSE transport not implemented yet")
            elif server_config.type == MCPServerType.GMAIL:
                client = GmailMCPClient(server_config)
            else:
                raise ValueError(f"Unsupported server type: {server_config.type}")
            
            # Connect to the server
            client.connect_to_server()
            
            # Store the client and config (without saving to disk)
            self.clients[server_config.server_name] = client
            self.server_configs[server_config.server_name] = server_config
            
            logger.info(f"Successfully restored MCP server: {server_config.server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore MCP server {server_config.server_name}: {str(e)}")
            return False

    def remove_server(self, server_name: str) -> bool:
        """Remove a server and clean up its client"""
        if server_name in self.clients:
            try:
                self.clients[server_name].cleanup()
                del self.clients[server_name]
                del self.server_configs[server_name]
                
                # Save configuration to disk for persistence
                self._save_persistent_connections()
                
                logger.info(f"Removed MCP server: {server_name}")
                return True
            except Exception as e:
                logger.error(f"Error removing MCP server {server_name}: {str(e)}")
                return False
        else:
            logger.warning(f"Server {server_name} not found")
            return False

    def list_servers(self) -> List[str]:
        """List all connected server names"""
        return list(self.clients.keys())

    def get_server_info(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific server"""
        if server_name in self.server_configs:
            config = self.server_configs[server_name]
            client = self.clients[server_name]
            return {
                "name": server_name,
                "type": config.type.value,
                "config": config.to_dict(),
                "initialized": client.initialized,
                "connected": server_name in self.clients
            }
        return None

    def list_tools(self, server_name: Optional[str] = None) -> Dict[str, List[MCPTool]]:
        """List tools from one or all servers"""
        if server_name:
            if server_name not in self.clients:
                raise MCPNotInitializedError(server_name)
            return {server_name: self.clients[server_name].list_tools()}
        else:
            # List tools from all servers
            all_tools = {}
            for name, client in self.clients.items():
                try:
                    all_tools[name] = client.list_tools()
                except Exception as e:
                    logger.error(f"Failed to list tools for server {name}: {str(e)}")
                    all_tools[name] = []
            return all_tools

    def execute_tool(self, server_name: str, tool_name: str, 
                     tool_args: Dict[str, Any]) -> Tuple[str, bool]:
        """Execute a tool on a specific server"""
        if server_name not in self.clients:
            raise MCPNotInitializedError(server_name)
        
        return self.clients[server_name].execute_tool(tool_name, tool_args)

    def find_tool(self, tool_name: str) -> Optional[Tuple[str, MCPTool]]:
        """Find a tool by name across all servers"""
        for server_name, client in self.clients.items():
            try:
                tools = client.list_tools()
                for tool in tools:
                    if tool.name == tool_name:
                        return server_name, tool
            except Exception as e:
                logger.error(f"Failed to search tools in server {server_name}: {str(e)}")
        return None

    def execute_tool_by_name(self, tool_name: str, 
                             tool_args: Dict[str, Any]) -> Tuple[str, bool]:
        """Execute a tool by name (searches all servers)"""
        result = self.find_tool(tool_name)
        if result:
            server_name, tool = result
            return self.execute_tool(server_name, tool_name, tool_args)
        else:
            raise ValueError(f"Tool '{tool_name}' not found in any connected server")

    def cleanup_all(self):
        """Clean up all clients"""
        for server_name in list(self.clients.keys()):
            self.remove_server(server_name)
        logger.info("Cleaned up all MCP clients")

    def _save_persistent_connections(self):
        """Save current server configurations to disk for persistence"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # Convert server configs to serializable format
            configs_data = {}
            for server_name, config in self.server_configs.items():
                configs_data[server_name] = {
                    'type': config.type.value,
                    'server_name': config.server_name,
                }
                
                # Add type-specific data
                if isinstance(config, StdioServerConfig):
                    configs_data[server_name].update({
                        'command': config.command,
                        'args': config.args,
                        'env': config.env
                    })
                elif isinstance(config, GmailServerConfig):
                    configs_data[server_name].update({
                        'client_id': config.client_id,
                        'client_secret': config.client_secret,
                        'token_file': config.token_file
                    })
            
            # Save to JSON file
            with open(self.config_file, 'w') as f:
                json.dump(configs_data, f, indent=2)
            
            logger.debug(f"Saved {len(configs_data)} MCP server configurations to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save MCP server configurations: {str(e)}")

    def _load_persistent_connections(self):
        """Load and restore server configurations from disk"""
        if not os.path.exists(self.config_file):
            logger.debug("No persistent MCP server configurations found")
            print("📝 No persistent MCP configurations found")
            return
        
        try:
            with open(self.config_file, 'r') as f:
                configs_data = json.load(f)
            
            if not configs_data:
                logger.info("No MCP server configurations to restore")
                print("📝 No MCP server configurations to restore")
                return
            
            logger.info(f"Loading {len(configs_data)} persistent MCP server configurations")
            print(f"🔄 Loading {len(configs_data)} persistent MCP configurations...")
            
            restored_count = 0
            failed_count = 0
            
            for server_name, config_data in configs_data.items():
                try:
                    # Recreate server config based on type
                    server_type = MCPServerType(config_data['type'])
                    
                    if server_type == MCPServerType.STDIO:
                        config = StdioServerConfig(
                            server_name=config_data['server_name'],
                            command=config_data['command'],
                            args=config_data.get('args', []),
                            env=config_data.get('env', {})
                        )
                    elif server_type == MCPServerType.GMAIL:
                        config = GmailServerConfig(
                            server_name=config_data['server_name'],
                            client_id=config_data['client_id'],
                            client_secret=config_data['client_secret'],
                            token_file=config_data.get('token_file')
                        )
                    else:
                        logger.warning(f"Unsupported server type {server_type} for {server_name}")
                        print(f"⚠️  Unsupported server type {server_type} for {server_name}")
                        failed_count += 1
                        continue
                    
                    # Try to reconnect (but don't save to disk to avoid recursion)
                    print(f"🔗 Attempting to restore {server_name} ({server_type.value})...")
                    if self._add_server_without_persistence(config):
                        logger.info(f"✅ Successfully restored connection to {server_name}")
                        print(f"✅ Restored MCP connection: {server_name}")
                        restored_count += 1
                    else:
                        logger.warning(f"❌ Failed to restore connection to {server_name}")
                        print(f"❌ Failed to restore MCP connection: {server_name}")
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to restore server {server_name}: {str(e)}")
                    print(f"❌ Error restoring {server_name}: {str(e)}")
                    failed_count += 1
                    continue
            
            print(f"🏁 MCP Restoration Complete: {restored_count} successful, {failed_count} failed")
            logger.info(f"MCP restoration complete: {restored_count} successful, {failed_count} failed")
                    
        except Exception as e:
            logger.error(f"Failed to load persistent MCP server configurations: {str(e)}")
            print(f"❌ Failed to load MCP configurations: {str(e)}")


class AsyncMCPClientManager:
    """Async version of MCP client manager"""

    def __init__(self):
        self.clients: Dict[str, BaseAsyncMCPClient] = {}
        self.server_configs: Dict[str, BaseServerConfig] = {}

    async def add_server(self, server_config: BaseServerConfig,
                         environment_variables: Optional[Dict[str, str]] = None) -> bool:
        """Asynchronously add a server configuration and connect to it"""
        try:
            # Resolve any templated environment variables
            server_config.resolve_environment_variables(environment_variables)
            
            # Create appropriate client based on server type
            if server_config.type == MCPServerType.STDIO:
                client = AsyncStdioMCPClient(server_config)
            elif server_config.type == MCPServerType.SSE:
                # Import async SSE client when needed (not implemented yet)
                raise NotImplementedError("Async SSE transport not implemented yet")
            else:
                raise ValueError(f"Unsupported server type: {server_config.type}")
            
            # Connect to the server
            await client.connect_to_server()
            
            # Store the client and config
            self.clients[server_config.server_name] = client
            self.server_configs[server_config.server_name] = server_config
            
            logger.info(f"Successfully added async MCP server: {server_config.server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add async MCP server {server_config.server_name}: {str(e)}")
            return False

    async def remove_server(self, server_name: str) -> bool:
        """Asynchronously remove a server and clean up its client"""
        if server_name in self.clients:
            try:
                await self.clients[server_name].cleanup()
                del self.clients[server_name]
                del self.server_configs[server_name]
                logger.info(f"Removed async MCP server: {server_name}")
                return True
            except Exception as e:
                logger.error(f"Error removing async MCP server {server_name}: {str(e)}")
                return False
        else:
            logger.warning(f"Async server {server_name} not found")
            return False

    async def list_tools(self, server_name: Optional[str] = None) -> Dict[str, List[MCPTool]]:
        """Asynchronously list tools from one or all servers"""
        if server_name:
            if server_name not in self.clients:
                raise MCPNotInitializedError(server_name)
            tools = await self.clients[server_name].list_tools()
            return {server_name: tools}
        else:
            # List tools from all servers
            all_tools = {}
            for name, client in self.clients.items():
                try:
                    all_tools[name] = await client.list_tools()
                except Exception as e:
                    logger.error(f"Failed to list tools for async server {name}: {str(e)}")
                    all_tools[name] = []
            return all_tools

    async def execute_tool(self, server_name: str, tool_name: str,
                           tool_args: Dict[str, Any]) -> Tuple[str, bool]:
        """Asynchronously execute a tool on a specific server"""
        if server_name not in self.clients:
            raise MCPNotInitializedError(server_name)
        
        return await self.clients[server_name].execute_tool(tool_name, tool_args)

    async def cleanup_all(self):
        """Asynchronously clean up all clients"""
        for server_name in list(self.clients.keys()):
            await self.remove_server(server_name)
        logger.info("Cleaned up all async MCP clients")


# Singleton instances for convenience
_mcp_manager = None
_async_mcp_manager = None


def get_mcp_client_manager() -> MCPClientManager:
    """Get the singleton MCP client manager instance"""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPClientManager()
    return _mcp_manager


def get_async_mcp_client_manager() -> AsyncMCPClientManager:
    """Get the singleton async MCP client manager instance"""
    global _async_mcp_manager
    if _async_mcp_manager is None:
        _async_mcp_manager = AsyncMCPClientManager()
    return _async_mcp_manager