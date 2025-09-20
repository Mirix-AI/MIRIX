"""
MCP SSE Server 核心实现

使用官方 MCP Python SDK 实现标准的 MCP (Model Context Protocol) 协议的 SSE (Server-Sent Events) 传输层。
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Sequence
from datetime import datetime, timedelta

import httpx
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 导入官方 MCP SDK
from mcp.server.fastmcp import FastMCP
from mcp.types import Tool, Resource, Prompt, TextContent, ImageContent
from mcp import types

from .config import Settings, get_settings
from .logging_config import LoggerMixin
from .mirix_client import MIRIXClient
from .session_manager import SessionManager
from .rate_limiter import RateLimiter

class MCPSSEServer(LoggerMixin):
    """基于官方 MCP SDK 的 SSE 服务器实现"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.router = APIRouter()
        self.mirix_client = MIRIXClient(settings.mirix_backend_url, settings.mirix_backend_timeout)
        self.session_manager = SessionManager(settings)
        self.rate_limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window)
        
        # 创建 FastMCP 实例
        self.mcp = FastMCP(
            name="MIRIX MCP SSE Service"
        )
        
        # 设置 MCP 服务器的工具、资源和提示
        self._setup_mcp_handlers()
        
        # 设置路由
        self._setup_routes()
        
        # 服务器状态
        self.server_info = {
            "name": "mirix-mcp-sse",
            "version": "0.1.0",
            "protocolVersion": settings.mcp_version,
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {},
                "logging": {}
            }
        }
    
    def _setup_mcp_handlers(self):
        """设置 MCP 处理器 - 专注记忆管理功能"""
        
        # 记忆管理工具
        @self.mcp.tool()
        async def memory_add(
            content: str,
            memory_type: str,
            context: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            添加记忆到 MIRIX 记忆系统
            
            使用场景：
            - 用户分享个人信息、偏好或重要事实时使用
            - 学习新知识或技能时，将关键信息存储起来
            - 记录重要的对话内容或决定
            - 保存工作流程、步骤说明等程序性知识
            
            执行顺序：通常是对话中的第一步，在获取到有价值信息后立即使用
            预期效果：信息被永久存储，可通过 memory_search 检索，增强 AI 对用户的了解
            """
            try:
                # 验证记忆类型
                valid_types = ["core", "episodic", "semantic", "procedural", "resource", "knowledge_vault"]
                if memory_type not in valid_types:
                    return {
                        "success": False,
                        "error": f"Invalid memory_type. Must be one of: {', '.join(valid_types)}"
                    }
                
                # 构建请求数据
                request_data = {
                    "content": content,
                    "memory_type": memory_type,
                    "user_id": self.settings.default_user_id
                }
                
                if context:
                    request_data["context"] = context
                
                # 调用 MIRIX 后端添加记忆
                result = await self.mirix_client.add_memory(request_data)
                
                self.logger.info("Memory added successfully", 
                               memory_type=memory_type, 
                               content_length=len(content))
                
                return {
                    "success": True,
                    "message": "Memory added successfully",
                    "memory_id": result.get("memory_id"),
                    "memory_type": memory_type
                }
                
            except Exception as e:
                self.logger.error("Failed to add memory", 
                                memory_type=memory_type, 
                                error=str(e))
                return {
                    "success": False,
                    "error": f"Failed to add memory: {str(e)}"
                }
        
        @self.mcp.tool()
        async def memory_search(
            query: str,
            memory_types: Optional[List[str]] = None,
            limit: int = 10
        ) -> Dict[str, Any]:
            """
            在用户记忆系统中搜索相关信息
            
            使用场景：
            - 用户询问之前讨论过的话题时使用
            - 需要回忆用户的偏好、习惯或个人信息时
            - 查找相关的知识、经验或程序步骤
            - 在回答问题前，先检索相关的背景信息
            
            执行顺序：通常在 memory_add 之前使用，避免重复存储；在回答用户问题前使用
            预期效果：获取相关的历史信息，提供更个性化、连贯的回应
            """
            try:
                # 验证记忆类型（如果提供）
                if memory_types:
                    valid_types = ["core", "episodic", "semantic", "procedural", "resource", "knowledge_vault"]
                    invalid_types = [t for t in memory_types if t not in valid_types]
                    if invalid_types:
                        return {
                            "success": False,
                            "error": f"Invalid memory_types: {', '.join(invalid_types)}. Must be one of: {', '.join(valid_types)}"
                        }
                
                # 构建搜索请求
                search_data = {
                    "query": query,
                    "user_id": self.settings.default_user_id,
                    "limit": min(limit, 50)  # 限制最大返回数量
                }
                
                if memory_types:
                    search_data["memory_types"] = memory_types
                
                # 调用 MIRIX 后端搜索记忆
                results = await self.mirix_client.search_memory(search_data)
                
                self.logger.info("Memory search completed", 
                               query=query, 
                               results_count=len(results.get("memories", [])))
                
                return {
                    "success": True,
                    "query": query,
                    "memories": results.get("memories", []),
                    "total_count": results.get("total_count", 0)
                }
                
            except Exception as e:
                self.logger.error("Failed to search memory", 
                                query=query, 
                                error=str(e))
                return {
                    "success": False,
                    "error": f"Failed to search memory: {str(e)}"
                }
        
        @self.mcp.tool()
        async def memory_chat(
            message: str,
            memorizing: bool = True,
            image_uris: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """
            发送消息给 MIRIX Agent 并自动管理记忆
            
            使用场景：
            - 进行需要记忆上下文的深度对话
            - 讨论重要话题，希望 AI 记住关键信息
            - 获取基于个人记忆的个性化回应
            - 当需要 AI 学习和适应用户偏好时
            
            执行顺序：可以独立使用，或在 memory_search 后使用以提供更好的上下文
            预期效果：获得个性化回应，重要信息自动存储到记忆中
            """
            try:
                # 构建消息请求
                chat_data = {
                    "message": message,
                    "user_id": self.settings.default_user_id,
                    "memorizing": memorizing,
                    "model": self.settings.ai_model
                }
                
                if image_uris:
                    chat_data["image_uris"] = image_uris
                
                # 调用 MIRIX 后端进行对话
                response = await self.mirix_client.send_chat_message(chat_data)
                
                self.logger.info("Memory chat completed", 
                               message_length=len(message), 
                               memorizing=memorizing,
                               has_images=bool(image_uris))
                
                return {
                    "success": True,
                    "response": response.get("response", ""),
                    "memorized": response.get("memorized", False),
                    "memory_updates": response.get("memory_updates", [])
                }
                
            except Exception as e:
                self.logger.error("Failed to process memory chat", 
                                message_length=len(message), 
                                error=str(e))
                return {
                    "success": False,
                    "error": f"Failed to process chat: {str(e)}"
                }
        
        @self.mcp.tool()
        async def memory_get_profile(
            memory_types: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """
            获取用户的完整记忆档案概览
            
            使用场景：
            - 初次与用户交互时，了解用户背景
            - 需要全面了解用户偏好和特点时
            - 为用户提供个性化建议前的信息收集
            - 定期回顾和更新对用户的了解
            
            执行顺序：通常在对话开始时使用，为后续交互提供基础
            预期效果：获得用户的全面画像，包括个人信息、偏好、历史记录等
            """
            try:
                # 验证记忆类型（如果提供）
                if memory_types:
                    valid_types = ["core", "episodic", "semantic", "procedural", "resource", "knowledge_vault"]
                    invalid_types = [t for t in memory_types if t not in valid_types]
                    if invalid_types:
                        return {
                            "success": False,
                            "error": f"Invalid memory_types: {', '.join(invalid_types)}. Must be one of: {', '.join(valid_types)}"
                        }
                
                # 构建档案请求
                profile_data = {
                    "user_id": self.settings.default_user_id
                }
                
                if memory_types:
                    profile_data["memory_types"] = memory_types
                
                # 调用 MIRIX 后端获取用户档案
                profile = await self.mirix_client.get_user_profile(profile_data)
                
                self.logger.info("User profile retrieved", 
                               user_id=self.settings.default_user_id,
                               memory_types_count=len(profile.get("memory_summary", {})))
                
                return {
                    "success": True,
                    "user_id": self.settings.default_user_id,
                    "profile": profile.get("profile", {}),
                    "memory_summary": profile.get("memory_summary", {}),
                    "total_memories": profile.get("total_memories", 0),
                    "last_updated": profile.get("last_updated")
                }
                
            except Exception as e:
                self.logger.error("Failed to get user profile", 
                                user_id=self.settings.default_user_id, 
                                error=str(e))
                return {
                    "success": False,
                    "error": f"Failed to get user profile: {str(e)}"
                }
        
        # 动态注册资源
        @self.mcp.resource("mirix://status")
        async def get_mirix_status() -> str:
            """获取 MIRIX 后端状态"""
            try:
                status = await self.mirix_client.get_mcp_status()
                return json.dumps(status, indent=2)
            except Exception as e:
                self.logger.error("Failed to get MIRIX status", error=str(e))
                return json.dumps({"error": str(e)}, indent=2)
        
        @self.mcp.resource("mirix://tools")
        async def get_mirix_tools_resource() -> str:
            """获取 MIRIX 工具列表作为资源"""
            try:
                tools = await self.mirix_client.list_tools()
                return json.dumps(tools, indent=2)
            except Exception as e:
                self.logger.error("Failed to get MIRIX tools resource", error=str(e))
                return json.dumps({"error": str(e)}, indent=2)
        
        # 动态注册提示
        @self.mcp.prompt()
        async def mirix_system_prompt(context: str = "general") -> str:
            """生成 MIRIX 系统提示"""
            return f"""You are MIRIX, an AI assistant with access to various tools and resources.
            
Context: {context}

Available capabilities:
- Tool execution through MIRIX backend
- Resource access and management
- Dynamic prompt generation

Please use the available tools and resources to help the user effectively."""
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.router.get("/")
        async def mcp_root():
            """MCP服务根路径"""
            return {
                "service": "MIRIX MCP SSE Service",
                "protocol": "MCP",
                "transport": "SSE",
                "version": self.server_info["version"],
                "protocolVersion": self.server_info["protocolVersion"]
            }
        
        # 健康检查端点
        @self.router.get("/health")
        async def health_check():
            """健康检查端点"""
            return {
                "status": "healthy",
                "service": "mirix-mcp-sse",
                "version": "0.1.0"
            }
        
        # 服务信息端点
        @self.router.get("/info")
        async def server_info():
            """服务器信息端点"""
            return self.server_info
    
    def get_sse_app(self):
        """获取 SSE ASGI 应用"""
        # 使用官方 MCP SDK 的 SSE 应用，不指定 mount_path 参数
        # 因为我们在 main.py 中已经将其挂载到 /sse 路径
        return self.mcp.sse_app()
    
    async def startup(self):
        """启动服务"""
        self.logger.info("Starting MCP SSE Server")
        
        # 启动 MIRIX 客户端
        await self.mirix_client.initialize()
        
        # 启动会话管理器
        await self.session_manager.startup()
        
        self.logger.info("MCP SSE Server started successfully")
    
    async def shutdown(self):
        """关闭服务"""
        self.logger.info("Shutting down MCP SSE Server")
        
        # 关闭会话管理器
        await self.session_manager.shutdown()
        
        # 关闭 MIRIX 客户端
        await self.mirix_client.close()
        
        self.logger.info("MCP SSE Server shut down successfully")