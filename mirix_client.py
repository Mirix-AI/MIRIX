"""
MIRIX客户端

用于与MIRIX后端服务通信，获取工具、资源和提示信息。
"""

import asyncio
from typing import Dict, List, Any, Optional
import httpx
from datetime import datetime, timedelta

from .config import Settings
from .logging_config import LoggerMixin

class MIRIXClient(LoggerMixin):
    """MIRIX后端客户端"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None
        self._capabilities_cache = None
        self._cache_timestamp = None
        self._cache_ttl = timedelta(minutes=5)  # 缓存5分钟
    
    async def initialize(self):
        """初始化客户端"""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "User-Agent": "MIRIX-MCP-SSE/0.1.0",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        )
        
        # 测试连接
        try:
            response = await self.client.get("/health")
            if response.status_code == 200:
                self.logger.info("Successfully connected to MIRIX backend", 
                               base_url=self.base_url)
            else:
                self.logger.warning("MIRIX backend health check failed", 
                                  status_code=response.status_code)
        except Exception as e:
            self.logger.error("Failed to connect to MIRIX backend", 
                            base_url=self.base_url, error=str(e))
            raise
    
    async def close(self):
        """关闭客户端"""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def _make_request(self, method: str, endpoint: str, 
                           data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """发起HTTP请求"""
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        try:
            if method.upper() == "GET":
                response = await self.client.get(endpoint, params=data)
            elif method.upper() == "POST":
                response = await self.client.post(endpoint, json=data)
            elif method.upper() == "PUT":
                response = await self.client.put(endpoint, json=data)
            elif method.upper() == "DELETE":
                response = await self.client.delete(endpoint)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            self.logger.error("HTTP request failed", 
                            method=method, 
                            endpoint=endpoint, 
                            status_code=e.response.status_code,
                            error=str(e))
            raise
        except Exception as e:
            self.logger.error("Request error", 
                            method=method, 
                            endpoint=endpoint, 
                            error=str(e))
            raise
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """获取MIRIX能力信息"""
        # 检查缓存
        if (self._capabilities_cache and self._cache_timestamp and 
            datetime.now() - self._cache_timestamp < self._cache_ttl):
            return self._capabilities_cache
        
        try:
            # 获取工具列表
            tools_response = await self._make_request("GET", "/mcp/tools")
            tools = tools_response.get("tools", [])
            
            # 获取资源列表
            resources_response = await self._make_request("GET", "/mcp/resources")
            resources = resources_response.get("resources", [])
            
            # 获取提示列表
            prompts_response = await self._make_request("GET", "/mcp/prompts")
            prompts = prompts_response.get("prompts", [])
            
            # 构建能力信息
            capabilities = {
                "tools": {
                    "listChanged": True  # 支持工具列表变更通知
                } if tools else {},
                "resources": {
                    "subscribe": True,  # 支持资源订阅
                    "listChanged": True  # 支持资源列表变更通知
                } if resources else {},
                "prompts": {
                    "listChanged": True  # 支持提示列表变更通知
                } if prompts else {},
                "logging": {}  # 支持日志
            }
            
            # 更新缓存
            self._capabilities_cache = capabilities
            self._cache_timestamp = datetime.now()
            
            self.logger.debug("Retrieved MIRIX capabilities", 
                            tools_count=len(tools),
                            resources_count=len(resources),
                            prompts_count=len(prompts))
            
            return capabilities
            
        except Exception as e:
            self.logger.error("Failed to get MIRIX capabilities", error=str(e))
            # 返回默认能力
            return {
                "tools": {},
                "resources": {},
                "prompts": {},
                "logging": {}
            }
    
    async def add_memory(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """添加记忆到 MIRIX 系统"""
        try:
            # 调用 MIRIX 后端的记忆添加 API
            result = await self._make_request("POST", "/api/memory/add", memory_data)
            return result
        except Exception as e:
            self.logger.error("Failed to add memory", error=str(e))
            raise
    
    async def search_memory(self, search_data: Dict[str, Any]) -> Dict[str, Any]:
        """搜索用户记忆"""
        try:
            # 调用 MIRIX 后端的记忆搜索 API
            result = await self._make_request("POST", "/api/memory/search", search_data)
            return result
        except Exception as e:
            self.logger.error("Failed to search memory", error=str(e))
            raise
    
    async def send_chat_message(self, chat_data: Dict[str, Any]) -> Dict[str, Any]:
        """发送聊天消息并处理记忆"""
        try:
            # 调用 MIRIX 后端的聊天 API
            result = await self._make_request("POST", "/api/chat/message", chat_data)
            return result
        except Exception as e:
            self.logger.error("Failed to send chat message", error=str(e))
            raise
    
    async def get_user_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取用户档案"""
        try:
            # 调用 MIRIX 后端的用户档案 API
            result = await self._make_request("POST", "/api/user/profile", profile_data)
            return result
        except Exception as e:
            self.logger.error("Failed to get user profile", error=str(e))
            raise

    async def list_tools(self) -> List[Dict[str, Any]]:
        """获取工具列表"""
        try:
            # 首先尝试从MIRIX后端获取MCP状态
            status_response = await self._make_request("GET", "/mcp/status")
            connected_servers = status_response.get("connected_servers", [])

            # 如果没有连接的MCP服务器，返回空工具列表
            if not connected_servers:
                self.logger.debug("No connected MCP servers, returning empty tools list")
                return []

            # 如果有连接的服务器，尝试获取工具（但MIRIX后端可能还没有实现这个端点）
            try:
                response = await self._make_request("GET", "/mcp/tools")
                tools = response.get("tools", [])
            except Exception as e:
                # 如果/mcp/tools端点不存在，返回空列表而不是报错
                self.logger.warning("MIRIX backend does not support /mcp/tools endpoint yet", error=str(e))
                tools = []

            # 如果没有工具，返回一个空数组而不是None
            if tools is None:
                tools = []

            self.logger.debug("Retrieved tools list", count=len(tools))
            return tools

        except Exception as e:
            self.logger.error("Failed to list tools", error=str(e))
            # 返回空数组而不是None，确保总是返回有效的tools字段
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        try:
            data = {
                "name": tool_name,
                "arguments": arguments
            }
            
            response = await self._make_request("POST", "/mcp/tools/call", data)
            
            self.logger.debug("Tool called successfully", 
                            tool_name=tool_name, 
                            arguments=arguments)
            
            return response
            
        except Exception as e:
            self.logger.error("Failed to call tool", 
                            tool_name=tool_name, 
                            arguments=arguments, 
                            error=str(e))
            
            # 返回错误结果
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error calling tool {tool_name}: {str(e)}"
                }],
                "isError": True
            }
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """获取资源列表"""
        try:
            # 首先检查是否有连接的MCP服务器
            status_response = await self._make_request("GET", "/mcp/status")
            connected_servers = status_response.get("connected_servers", [])

            if not connected_servers:
                self.logger.debug("No connected MCP servers, returning empty resources list")
                return []

            # 尝试获取资源列表
            try:
                response = await self._make_request("GET", "/mcp/resources")
                resources = response.get("resources", [])
            except Exception as e:
                self.logger.warning("MIRIX backend does not support /mcp/resources endpoint yet", error=str(e))
                resources = []

            self.logger.debug("Retrieved resources list", count=len(resources))
            return resources

        except Exception as e:
            self.logger.error("Failed to list resources", error=str(e))
            return []
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """读取资源"""
        try:
            data = {"uri": uri}
            response = await self._make_request("POST", "/mcp/resources/read", data)
            
            self.logger.debug("Resource read successfully", uri=uri)
            return response
            
        except Exception as e:
            self.logger.error("Failed to read resource", uri=uri, error=str(e))
            
            # 返回错误内容
            return {
                "uri": uri,
                "mimeType": "text/plain",
                "text": f"Error reading resource {uri}: {str(e)}"
            }
    
    async def list_prompts(self) -> List[Dict[str, Any]]:
        """获取提示列表"""
        try:
            # 首先检查是否有连接的MCP服务器
            status_response = await self._make_request("GET", "/mcp/status")
            connected_servers = status_response.get("connected_servers", [])

            if not connected_servers:
                self.logger.debug("No connected MCP servers, returning empty prompts list")
                return []

            # 尝试获取提示列表
            try:
                response = await self._make_request("GET", "/mcp/prompts")
                prompts = response.get("prompts", [])
            except Exception as e:
                self.logger.warning("MIRIX backend does not support /mcp/prompts endpoint yet", error=str(e))
                prompts = []

            self.logger.debug("Retrieved prompts list", count=len(prompts))
            return prompts

        except Exception as e:
            self.logger.error("Failed to list prompts", error=str(e))
            return []
    
    async def get_prompt(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """获取提示"""
        try:
            data = {
                "name": name,
                "arguments": arguments
            }
            
            response = await self._make_request("POST", "/mcp/prompts/get", data)
            
            self.logger.debug("Prompt retrieved successfully", 
                            name=name, 
                            arguments=arguments)
            
            return response
            
        except Exception as e:
            self.logger.error("Failed to get prompt", 
                            name=name, 
                            arguments=arguments, 
                            error=str(e))
            
            # 返回错误提示
            return {
                "description": f"Error getting prompt {name}",
                "messages": [{
                    "role": "assistant",
                    "content": f"Error getting prompt {name}: {str(e)}"
                }]
            }
    
    async def send_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """发送消息到MIRIX"""
        try:
            data = {
                "message": message,
                "context": context or {}
            }
            
            response = await self._make_request("POST", "/send_message", data)
            
            self.logger.debug("Message sent successfully", message_length=len(message))
            return response
            
        except Exception as e:
            self.logger.error("Failed to send message", error=str(e))
            raise
    
    async def send_streaming_message(self, message: str, 
                                   context: Optional[Dict[str, Any]] = None) -> httpx.Response:
        """发送流式消息到MIRIX"""
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        try:
            data = {
                "message": message,
                "context": context or {}
            }
            
            response = await self.client.post("/send_streaming_message", 
                                            json=data, 
                                            headers={"Accept": "text/event-stream"})
            response.raise_for_status()
            
            self.logger.debug("Streaming message sent successfully", message_length=len(message))
            return response
            
        except Exception as e:
            self.logger.error("Failed to send streaming message", error=str(e))
            raise
    
    async def get_mcp_status(self) -> Dict[str, Any]:
        """获取MCP状态"""
        try:
            response = await self._make_request("GET", "/mcp/status")
            
            self.logger.debug("Retrieved MCP status")
            return response
            
        except Exception as e:
            self.logger.error("Failed to get MCP status", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            response = await self._make_request("GET", "/health")
            return response.get("status") == "healthy"
            
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False