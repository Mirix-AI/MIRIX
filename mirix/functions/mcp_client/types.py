"""
MCP Client Types and Configuration - adapted from Letta
"""

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

from mcp.types import Tool
from pydantic import BaseModel, Field

# Template variable regex for environment variable substitution
TEMPLATED_VARIABLE_REGEX = (
    r"\{\{\s*([A-Z_][A-Z0-9_]*)\s*(?:\|\s*([^}]+?)\s*)?\}\}"
)


class MCPTool(Tool):
    """A wrapper around MCP's tool definition for consistency"""
    pass


class MCPServerType(str, Enum):
    SSE = "sse"
    STDIO = "stdio"
    GMAIL = "gmail"


class BaseServerConfig(BaseModel, ABC):
    """Base configuration for MCP servers"""
    
    server_name: str = Field(..., description="The name of the server")
    type: MCPServerType

    def is_templated_tool_variable(self, value: str) -> bool:
        """Check if string contains templated variables like {{ VARIABLE_NAME }}"""
        return bool(re.search(TEMPLATED_VARIABLE_REGEX, value))

    def get_tool_variable(self, value: str, environment_variables: Dict[str, str]) -> Optional[str]:
        """Replace templated variables with environment variable values"""
        if not self.is_templated_tool_variable(value):
            return value

        def replace_template(match):
            variable_name = match.group(1)
            default_value = match.group(2) if match.group(2) else None

            # Try to get the value from environment variables
            env_value = environment_variables.get(variable_name) if environment_variables else None

            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                return match.group(0)  # Return original if no replacement found

        result = re.sub(TEMPLATED_VARIABLE_REGEX, replace_template, value)
        
        # If still contains unreplaced templates, return original
        if re.search(TEMPLATED_VARIABLE_REGEX, result):
            return value

        return result

    @abstractmethod
    def resolve_environment_variables(self, environment_variables: Optional[Dict[str, str]] = None) -> None:
        """Resolve templated environment variables in configuration"""
        raise NotImplementedError


class StdioServerConfig(BaseServerConfig):
    """Configuration for MCP server using stdio transport"""
    
    type: MCPServerType = MCPServerType.STDIO
    command: str = Field(..., description="The command to run")
    args: List[str] = Field(default_factory=list, description="Arguments to pass to the command")
    env: Optional[Dict[str, str]] = Field(None, description="Environment variables to set")

    def resolve_environment_variables(self, environment_variables: Optional[Dict[str, str]] = None) -> None:
        """Resolve any templated variables in env dict"""
        if self.env and environment_variables:
            for key, value in self.env.items():
                if self.is_templated_tool_variable(value):
                    resolved = self.get_tool_variable(value, environment_variables)
                    if resolved:
                        self.env[key] = resolved

    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "transport": "stdio",
            "command": self.command,
            "args": self.args,
            "env": self.env
        }


class SSEServerConfig(BaseServerConfig):
    """Configuration for MCP server using SSE transport"""
    
    type: MCPServerType = MCPServerType.SSE
    server_url: str = Field(..., description="The URL of the SSE server")
    auth_header: Optional[str] = Field(None, description="Authentication header name")
    auth_token: Optional[str] = Field(None, description="Authentication token")
    custom_headers: Optional[Dict[str, str]] = Field(None, description="Custom HTTP headers")

    def resolve_environment_variables(self, environment_variables: Optional[Dict[str, str]] = None) -> None:
        """Resolve templated variables in auth token and headers"""
        if self.auth_token and self.is_templated_tool_variable(self.auth_token):
            resolved = self.get_tool_variable(self.auth_token, environment_variables)
            if resolved:
                self.auth_token = resolved

        if self.custom_headers and environment_variables:
            for key, value in self.custom_headers.items():
                if self.is_templated_tool_variable(value):
                    resolved = self.get_tool_variable(value, environment_variables)
                    if resolved:
                        self.custom_headers[key] = resolved

    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        result = {
            "transport": "sse",
            "url": self.server_url
        }
        
        if self.custom_headers or (self.auth_header and self.auth_token):
            headers = self.custom_headers.copy() if self.custom_headers else {}
            if self.auth_header and self.auth_token:
                headers[self.auth_header] = self.auth_token
            result["headers"] = headers
            
        return result


class GmailServerConfig(BaseServerConfig):
    """Configuration for Gmail MCP server using OAuth2"""
    
    type: MCPServerType = MCPServerType.GMAIL
    client_id: str = Field(..., description="Gmail OAuth2 Client ID")
    client_secret: str = Field(..., description="Gmail OAuth2 Client Secret")
    token_file: Optional[str] = Field(None, description="Path to store OAuth2 token")

    def resolve_environment_variables(self, environment_variables: Optional[Dict[str, str]] = None) -> None:
        """Resolve templated variables in client credentials"""
        if environment_variables:
            if self.is_templated_tool_variable(self.client_id):
                resolved = self.get_tool_variable(self.client_id, environment_variables)
                if resolved:
                    self.client_id = resolved
            
            if self.is_templated_tool_variable(self.client_secret):
                resolved = self.get_tool_variable(self.client_secret, environment_variables)
                if resolved:
                    self.client_secret = resolved

    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "transport": "gmail",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "token_file": self.token_file
        }