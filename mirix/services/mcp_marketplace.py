"""
MCP Marketplace - Registry of available MCP servers that users can connect to
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class MCPServerListing:
    """Represents an MCP server available in the marketplace"""
    id: str
    name: str
    description: str
    category: str
    command: str  # Command to run the server
    args: List[str]  # Arguments for the command
    env: Optional[Dict[str, str]] = None  # Environment variables
    install_command: Optional[str] = None  # Command to install the server
    documentation_url: Optional[str] = None
    github_url: Optional[str] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    requirements: Optional[List[str]] = None  # System requirements
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'command': self.command,
            'args': self.args,
            'env': self.env or {},
            'install_command': self.install_command,
            'documentation_url': self.documentation_url,
            'github_url': self.github_url,
            'author': self.author,
            'tags': self.tags or [],
            'requirements': self.requirements or []
        }


class MCPMarketplace:
    """Registry of available MCP servers"""
    
    def __init__(self):
        self.servers = self._initialize_marketplace()
    
    def _initialize_marketplace(self) -> Dict[str, MCPServerListing]:
        """Initialize with Gmail native MCP server only"""
        servers = [
            # MIRIX NATIVE GMAIL IMPLEMENTATION
            MCPServerListing(
                id='gmail-native',
                name='Gmail (Native)',
                description='Native Gmail integration with advanced email management, sending, reading, drafting, and organization capabilities using your existing OAuth2 credentials',
                category='Communication',
                command='python',
                args=['-m', 'mirix.functions.mcp_client.gmail_server'],
                env={'PYTHONPATH': ''},  # Will be set to current working directory
                install_command='# Already integrated with Mirix - uses existing OAuth2 credentials',
                documentation_url='https://docs.mirix.io/',
                author='Mirix',
                tags=['gmail', 'email', 'oauth2', 'native', 'google-workspace'],
                requirements=['google-auth', 'google-api-python-client', 'oauth2-credentials']
            )
        ]
        
        return {server.id: server for server in servers}
    
    def get_all_servers(self) -> List[MCPServerListing]:
        """Get all available servers"""
        return list(self.servers.values())
    
    def get_server(self, server_id: str) -> Optional[MCPServerListing]:
        """Get a specific server by ID"""
        return self.servers.get(server_id)
    
    def get_by_category(self, category: str) -> List[MCPServerListing]:
        """Get servers by category"""
        return [s for s in self.servers.values() if s.category == category]
    
    def get_categories(self) -> List[str]:
        """Get all unique categories"""
        categories = set(s.category for s in self.servers.values())
        return sorted(list(categories))
    
    def search(self, query: str) -> List[MCPServerListing]:
        """Search servers by name, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for server in self.servers.values():
            # Search in name
            if query_lower in server.name.lower():
                results.append(server)
                continue
            
            # Search in description
            if query_lower in server.description.lower():
                results.append(server)
                continue
            
            # Search in tags
            if server.tags:
                for tag in server.tags:
                    if query_lower in tag.lower():
                        results.append(server)
                        break
        
        return results


# Global marketplace instance
_marketplace = None

def get_mcp_marketplace() -> MCPMarketplace:
    """Get the global MCP marketplace instance"""
    global _marketplace
    if _marketplace is None:
        _marketplace = MCPMarketplace()
    return _marketplace