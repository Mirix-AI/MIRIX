"""
Dynamic MCP Tool Registry - Handles discovery and registration of MCP tools
"""

import logging
from typing import Dict, List, Optional, Any
from mirix.functions.mcp_client import get_mcp_client_manager
from mirix.schemas.tool import Tool as PydanticTool, ToolCreate
from mirix.services.tool_manager import ToolManager
from mirix.schemas.user import User as PydanticUser
from mirix.orm.enums import ToolType
from mirix.utils import printd

logger = logging.getLogger(__name__)


class MCPToolRegistry:
    """Registry for dynamically discovered MCP tools"""
    
    def __init__(self):
        self.tool_manager = ToolManager()
        self._mcp_tool_cache: Dict[str, List[Dict[str, Any]]] = {}
    
    def discover_mcp_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Discover all tools from connected MCP servers"""
        try:
            mcp_manager = get_mcp_client_manager()
            if not mcp_manager:
                printd("No MCP manager available for tool discovery")
                return {}
            
            # Get tools from all servers
            tools_by_server = mcp_manager.list_tools()
            
            # Convert MCPTool objects to dictionaries for easier handling
            discovered_tools = {}
            for server_name, mcp_tools in tools_by_server.items():
                server_tools = []
                for mcp_tool in mcp_tools:
                    tool_dict = {
                        'name': mcp_tool.name,
                        'description': mcp_tool.description,
                        'server_name': server_name,
                        'full_name': f"{server_name}.{mcp_tool.name}",
                        'schema': mcp_tool.inputSchema if hasattr(mcp_tool, 'inputSchema') else {},
                        'mcp_tool': mcp_tool  # Keep reference to original object
                    }
                    server_tools.append(tool_dict)
                discovered_tools[server_name] = server_tools
            
            self._mcp_tool_cache = discovered_tools
            return discovered_tools
            
        except Exception as e:
            logger.error(f"Error discovering MCP tools: {str(e)}")
            return {}
    
    def register_mcp_tools(self, actor: PydanticUser, 
                          server_filter: Optional[List[str]] = None) -> List[PydanticTool]:
        """
        Register discovered MCP tools in the database
        
        Args:
            actor: User performing the registration
            server_filter: Optional list of server names to register tools from
            
        Returns:
            List of registered Tool objects
        """

        discovered_tools = self.discover_mcp_tools()
        registered_tools = []
        
        for server_name, tools in discovered_tools.items():
            # Skip if server not in filter
            if server_filter and server_name not in server_filter:
                continue
            
            for tool_info in tools:
                try:
                    # Generate source code that calls the generic MCP tool
                    source_code = self._generate_mcp_tool_wrapper(tool_info)
                    
                    # Create JSON schema for the tool
                    json_schema = self._create_mcp_tool_schema(tool_info)
                    
                    # Create or update tool in database
                    # Use the same name format as JSON schema (with underscores)
                    normalized_name = tool_info['full_name'].replace('.', '_').replace('-', '_')
                    pydantic_tool = PydanticTool(
                        name=normalized_name,
                        description=tool_info['description'],
                        source_code=source_code,
                        json_schema=json_schema,
                        tool_type=ToolType.MIRIX_MCP,
                        tags=[ToolType.MIRIX_MCP.value, f"mcp:{server_name}"],
                        source_type="python"
                    )
                    
                    registered_tool = self.tool_manager.create_or_update_tool(
                        pydantic_tool, actor
                    )
                    registered_tools.append(registered_tool)
                    
                    logger.info(f"Registered MCP tool: {tool_info['full_name']}")
                    
                except Exception as e:
                    logger.error(f"Failed to register MCP tool {tool_info['full_name']}: {str(e)}")
                    continue
                
        return registered_tools
    
    def _generate_mcp_tool_wrapper(self, tool_info: Dict[str, Any]) -> str:
        """Generate Python source code wrapper for an MCP tool"""
        server_name = tool_info['server_name']
        tool_name = tool_info['name']
        full_name = tool_info['full_name']
        
        # Generate parameter list from schema
        params = []
        if 'schema' in tool_info and 'properties' in tool_info['schema']:
            for param_name, param_info in tool_info['schema']['properties'].items():
                param_type = param_info.get('type', 'str')
                python_type = self._json_type_to_python_type(param_type)
                
                # Check if parameter is required
                required_params = tool_info['schema'].get('required', [])
                if param_name in required_params:
                    params.append(f"{param_name}: {python_type}")
                else:
                    params.append(f"{param_name}: Optional[{python_type}] = None")
        
        # Add self and agent_state parameters (required by Mirix)
        all_params = ["self: \"Agent\"", "agent_state: \"AgentState\""] + params
        param_str = ", ".join(all_params)
        
        # Generate function arguments dictionary
        args_dict_entries = []
        for param in params:
            param_name = param.split(':')[0].strip()
            if 'Optional' in param:
                args_dict_entries.append(f'        if {param_name} is not None:\n            args["{param_name}"] = {param_name}')
            else:
                args_dict_entries.append(f'        args["{param_name}"] = {param_name}')
        
        args_dict_str = "\n".join(args_dict_entries) if args_dict_entries else "        pass"
        
        source_code = f'''def {full_name.replace('.', '_').replace('-', '_')}(
    {param_str}
) -> str:
    """
    {tool_info['description']}
    
    This is an auto-generated wrapper for MCP tool: {server_name}.{tool_name}
    """
    from typing import Optional
    from mirix.functions.mcp_client import get_mcp_client_manager
    
    try:
        mcp_manager = get_mcp_client_manager()
        if not mcp_manager:
            return "Error: MCP manager not available"
        
        args = {{}}
{args_dict_str}
        
        result_text, is_error = mcp_manager.execute_tool("{server_name}", "{tool_name}", args)
        if is_error:
            return f"Error executing {full_name}: {{result_text}}"
        
        return result_text
        
    except Exception as e:
        return f"Error executing {full_name}: {{str(e)}}"
'''
        return source_code
    
    def _create_mcp_tool_schema(self, tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create OpenAI-compatible JSON schema for MCP tool"""
        base_schema = {
            "name": tool_info['full_name'].replace('.', '_').replace('-', '_'),
            "description": tool_info['description'],
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        
        # Convert MCP schema to OpenAI format
        if 'schema' in tool_info and 'properties' in tool_info['schema']:
            base_schema["parameters"]["properties"] = tool_info['schema']['properties']
            base_schema["parameters"]["required"] = tool_info['schema'].get('required', [])
        
        return base_schema
    
    def _json_type_to_python_type(self, json_type: str) -> str:
        """Convert JSON schema type to Python type annotation"""
        type_map = {
            'string': 'str',
            'integer': 'int',
            'number': 'float',
            'boolean': 'bool',
            'array': 'list',
            'object': 'dict'
        }
        return type_map.get(json_type, 'str')
    
    def unregister_mcp_tools(self, actor: PydanticUser, 
                           server_name: Optional[str] = None) -> int:
        """
        Unregister MCP tools from database
        
        Args:
            actor: User performing the unregistration
            server_name: Optional server name to filter tools by
            
        Returns:
            Number of tools unregistered
        """
        try:
            # Get all MCP tools
            tools = self.tool_manager.list_tools(actor)
            mcp_tools = [t for t in tools if t.tool_type == ToolType.MIRIX_MCP]
            
            unregistered_tools = []
            for tool in mcp_tools:
                # Filter by server name if provided
                if server_name:
                    server_tag = f"mcp:{server_name}"
                    if server_tag not in tool.tags:
                        continue
                
                unregistered_tools.append(tool.id)
                self.tool_manager.delete_tool_by_id(tool.id, actor)
                printd(f"Unregistered MCP tool: {tool.name}")
            
            return unregistered_tools
            
        except Exception as e:
            logger.error(f"Error unregistering MCP tools: {str(e)}")
            return []
    
    def sync_mcp_tools(self, actor: PydanticUser) -> Dict[str, int]:
        """
        Synchronize database with currently available MCP tools
        
        Returns:
            Dictionary with 'registered', 'unregistered', 'updated' counts
        """
        # Discover current tools
        discovered_tools = self.discover_mcp_tools()
        current_tool_names = set()
        for tools in discovered_tools.values():
            current_tool_names.update(t['full_name'] for t in tools)
        
        # Get existing MCP tools from database
        existing_tools = self.tool_manager.list_tools(actor)
        existing_mcp_tools = [t for t in existing_tools if t.tool_type == ToolType.MIRIX_MCP]
        existing_tool_names = {t.name for t in existing_mcp_tools}
        
        # Register new tools
        new_tools = current_tool_names - existing_tool_names
        registered_count = 0
        if new_tools:
            # Filter to only register new tools
            filtered_discovered = {}
            for server_name, tools in discovered_tools.items():
                filtered_tools = [t for t in tools if t['full_name'] in new_tools]
                if filtered_tools:
                    filtered_discovered[server_name] = filtered_tools
            
            # Temporarily override cache and register
            old_cache = self._mcp_tool_cache
            self._mcp_tool_cache = filtered_discovered
            registered_tools = self.register_mcp_tools(actor)
            registered_count = len(registered_tools)
            self._mcp_tool_cache = old_cache
        
        # Unregister obsolete tools
        obsolete_tools = existing_tool_names - current_tool_names
        unregistered_count = 0
        for tool in existing_mcp_tools:
            if tool.name in obsolete_tools:
                self.tool_manager.delete_tool_by_id(tool.id, actor)
                unregistered_count += 1
        
        return {
            'registered': registered_count,
            'unregistered': unregistered_count,
            'total_current': len(current_tool_names)
        }


# Global registry instance
_mcp_tool_registry = None

def get_mcp_tool_registry() -> MCPToolRegistry:
    """Get the global MCP tool registry instance"""
    global _mcp_tool_registry
    if _mcp_tool_registry is None:
        _mcp_tool_registry = MCPToolRegistry()
    return _mcp_tool_registry