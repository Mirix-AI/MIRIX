"""
API路由定义

定义MCP SSE服务的所有API端点。
"""

from fastapi import APIRouter, HTTPException, Request, Depends, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from pydantic import BaseModel

from .config import Settings, get_settings
from .mcp_protocol import MCPMessage, validate_mcp_message
from .logging_config import LoggerMixin

if TYPE_CHECKING:
    from .mcp_server import MCPSSEServer

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float

class StatsResponse(BaseModel):
    """统计信息响应"""
    sessions: Dict[str, Any]
    rate_limits: Dict[str, Any]
    server_info: Dict[str, Any]

class MessageRequest(BaseModel):
    """消息请求"""
    message: str
    context: Optional[Dict[str, Any]] = None

def create_routes(mcp_server: "MCPSSEServer", settings: Settings) -> APIRouter:
    """创建API路由"""
    
    router = APIRouter()
    logger_mixin = LoggerMixin()
    
    # 健康检查
    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        """健康检查端点"""
        import time
        from datetime import datetime
        
        # 检查MIRIX后端连接
        mirix_healthy = await mcp_server.mirix_client.health_check()
        
        status = "healthy" if mirix_healthy else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.utcnow().isoformat(),
            version="0.1.0",
            uptime_seconds=time.time() - getattr(mcp_server, '_start_time', time.time())
        )
    
    # 服务信息
    @router.get("/info")
    async def service_info():
        """服务信息端点"""
        return {
            "service": "MIRIX MCP SSE Service",
            "version": "0.1.0",
            "protocol": "MCP",
            "transport": "SSE",
            "capabilities": mcp_server.server_info["capabilities"],
            "endpoints": {
                "connect": "/connect",
                "sse": "/sse/{session_id}",
                "message": "/message/{session_id}",
                "disconnect": "/disconnect/{session_id}"
            }
        }
    
    # 统计信息
    @router.get("/stats", response_model=StatsResponse)
    async def get_stats():
        """获取服务统计信息"""
        try:
            # 获取会话统计
            session_stats = await mcp_server.session_manager.get_session_stats()
            
            # 获取速率限制统计
            rate_limit_stats = await mcp_server.rate_limiter.get_all_stats()
            
            # 服务器信息
            server_info = {
                "name": mcp_server.server_info["name"],
                "version": mcp_server.server_info["version"],
                "protocol_version": mcp_server.server_info["protocolVersion"],
                "max_sessions": settings.max_sessions,
                "session_timeout": settings.session_timeout
            }
            
            return StatsResponse(
                sessions=session_stats,
                rate_limits=rate_limit_stats,
                server_info=server_info
            )
            
        except Exception as e:
            logger_mixin.logger.error("Failed to get stats", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to get statistics")
    
    # 列出活跃会话
    @router.get("/sessions")
    async def list_sessions():
        """列出所有活跃会话"""
        try:
            sessions = await mcp_server.session_manager.list_sessions()
            return {"sessions": sessions}
        except Exception as e:
            logger_mixin.logger.error("Failed to list sessions", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to list sessions")
    
    # 获取特定会话信息
    @router.get("/sessions/{session_id}")
    async def get_session_info(session_id: str):
        """获取特定会话信息"""
        session = await mcp_server.session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session.session_id,
            "client_ip": session.client_ip,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "is_initialized": session.is_initialized,
            "queue_size": session.message_queue.qsize(),
            "metadata": session.metadata
        }
    
    # 强制断开会话
    @router.delete("/sessions/{session_id}")
    async def force_disconnect_session(session_id: str):
        """强制断开会话"""
        success = await mcp_server.session_manager.remove_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": "Session disconnected successfully"}
    
    # 广播消息到所有会话
    @router.post("/broadcast")
    async def broadcast_message(request: MessageRequest):
        """向所有会话广播消息"""
        try:
            message = {
                "type": "broadcast",
                "content": request.message,
                "context": request.context or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await mcp_server.session_manager.broadcast_message(message)
            
            session_count = await mcp_server.session_manager.get_session_count()
            return {
                "message": "Message broadcasted successfully",
                "recipients": session_count
            }
            
        except Exception as e:
            logger_mixin.logger.error("Failed to broadcast message", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to broadcast message")
    
    # 重置客户端速率限制
    @router.post("/rate-limit/reset/{client_id}")
    async def reset_rate_limit(client_id: str):
        """重置客户端速率限制"""
        try:
            await mcp_server.rate_limiter.reset_client_limit(client_id)
            return {"message": f"Rate limit reset for client {client_id}"}
        except Exception as e:
            logger_mixin.logger.error("Failed to reset rate limit", 
                                    client_id=client_id, error=str(e))
            raise HTTPException(status_code=500, detail="Failed to reset rate limit")
    
    # 获取客户端速率限制状态
    @router.get("/rate-limit/{client_id}")
    async def get_rate_limit_status(client_id: str):
        """获取客户端速率限制状态"""
        try:
            stats = await mcp_server.rate_limiter.get_client_stats(client_id)
            remaining_tokens = await mcp_server.rate_limiter.get_remaining_tokens(client_id)
            time_until_next = await mcp_server.rate_limiter.get_time_until_next_token(client_id)
            
            return {
                "client_id": client_id,
                "remaining_tokens": remaining_tokens,
                "time_until_next_token": time_until_next,
                "stats": stats
            }
        except Exception as e:
            logger_mixin.logger.error("Failed to get rate limit status", 
                                    client_id=client_id, error=str(e))
            raise HTTPException(status_code=500, detail="Failed to get rate limit status")
    
    # MCP工具相关端点
    @router.get("/tools")
    async def list_mcp_tools(request: Request):
        """列出MCP工具"""
        # For GET requests, we might not have a standard JSON-RPC ID,
        # but we can try to get it from query params if provided.
        request_id = request.query_params.get("id")
        try:
            tools = await mcp_server.mirix_client.list_tools()
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": tools}
            })
        except Exception as e:
            logger_mixin.logger.error("Failed to list MCP tools", error=str(e))
            return JSONResponse(status_code=500, content={
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": "Internal error", "data": str(e)}
            })

    @router.get("/mcp/tools")
    async def list_mcp_tools_v2(request: Request):
        """列出MCP工具 - MCP协议端点"""
        request_id = request.query_params.get("id")
        try:
            tools = await mcp_server.mirix_client.list_tools()
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": tools}
            })
        except Exception as e:
            logger_mixin.logger.error("Failed to list MCP tools", error=str(e))
            return JSONResponse(status_code=500, content={
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": "Internal error", "data": str(e)}
            })
    
    @router.post("/tools/call")
    async def call_mcp_tool(request: Dict[str, Any]):
        """调用MCP工具"""
        try:
            tool_name = request.get("name")
            arguments = request.get("arguments", {})
            
            if not tool_name:
                raise HTTPException(status_code=400, detail="Tool name is required")
            
            result = await mcp_server.mirix_client.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            logger_mixin.logger.error("Failed to call MCP tool", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to call tool")
    
    # MCP资源相关端点
    @router.get("/mcp/resources")
    async def list_mcp_resources():
        """列出MCP资源"""
        try:
            resources = await mcp_server.mirix_client.list_resources()
            return {"resources": resources}
        except Exception as e:
            logger_mixin.logger.error("Failed to list MCP resources", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to list resources")
    
    @router.post("/mcp/resources/read")
    async def read_mcp_resource(request: Dict[str, Any]):
        """读取MCP资源"""
        try:
            uri = request.get("uri")
            if not uri:
                raise HTTPException(status_code=400, detail="Resource URI is required")
            
            content = await mcp_server.mirix_client.read_resource(uri)
            return {"content": content}
        except Exception as e:
            logger_mixin.logger.error("Failed to read MCP resource", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to read resource")
    
    # MCP提示相关端点
    @router.get("/mcp/prompts")
    async def list_mcp_prompts():
        """列出MCP提示"""
        try:
            prompts = await mcp_server.mirix_client.list_prompts()
            return {"prompts": prompts}
        except Exception as e:
            logger_mixin.logger.error("Failed to list MCP prompts", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to list prompts")
    
    @router.post("/mcp/prompts/get")
    async def get_mcp_prompt(request: Dict[str, Any]):
        """获取MCP提示"""
        try:
            name = request.get("name")
            arguments = request.get("arguments", {})
            
            if not name:
                raise HTTPException(status_code=400, detail="Prompt name is required")
            
            prompt = await mcp_server.mirix_client.get_prompt(name, arguments)
            return prompt
        except Exception as e:
            logger_mixin.logger.error("Failed to get MCP prompt", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to get prompt")
    
    # MIRIX消息端点
    @router.post("/mirix/message")
    async def send_mirix_message(request: MessageRequest):
        """发送消息到MIRIX"""
        try:
            result = await mcp_server.mirix_client.send_message(
                request.message, 
                request.context
            )
            return result
        except Exception as e:
            logger_mixin.logger.error("Failed to send message to MIRIX", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to send message")
    
    # 配置端点
    @router.get("/config")
    async def get_config():
        """获取服务配置（敏感信息已脱敏）"""
        config_dict = settings.dict()
        
        # 脱敏处理
        sensitive_keys = ['mirix_backend_url', 'cors_origins', 'log_file']
        for key in sensitive_keys:
            if key in config_dict and config_dict[key]:
                if isinstance(config_dict[key], str):
                    config_dict[key] = "***"
                elif isinstance(config_dict[key], list):
                    config_dict[key] = ["***"] * len(config_dict[key])
        
        return config_dict
    
    # 错误处理 - 移除exception_handler，因为APIRouter不支持
    # 错误处理应该在FastAPI应用级别设置
    
    return router