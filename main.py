#!/usr/bin/env python3
"""
MIRIX MCP SSE Service - 主入口文件

启动MCP SSE服务，提供对外的MCP协议接口。
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .mcp_server import MCPSSEServer
from .logging_config import setup_logging

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """创建FastAPI应用实例"""
    settings = get_settings()
    
    app = FastAPI(
        title="MIRIX MCP SSE Service",
        description="提供对外的MCP (Model Context Protocol) 服务接口，支持SSE传输",
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 创建MCP服务器实例
    mcp_server = MCPSSEServer(settings)
    
    # 注册路由
    app.include_router(mcp_server.router, prefix="/mcp")
    
    # 挂载 SSE 应用到 /sse 端点（符合 MCP 客户端配置）
    sse_app = mcp_server.get_sse_app()
    app.mount("/", sse_app)
    
    # 健康检查端点
    @app.get("/health")
    async def health_check():
        """健康检查端点"""
        return {
            "status": "healthy",
            "service": "mirix-mcp-sse",
            "version": "0.1.0"
        }
    
    # 根路径信息
    @app.get("/")
    async def root():
        """根路径信息"""
        return {
            "service": "MIRIX MCP SSE Service",
            "version": "0.1.0",
            "description": "提供对外的MCP协议接口，支持SSE传输",
            "endpoints": {
                "health": "/health",
                "mcp": "/mcp",
                "docs": "/docs" if settings.debug else "disabled"
            }
        }
    
    # 启动事件
    @app.on_event("startup")
    async def startup_event():
        """应用启动事件"""
        logger.info("MIRIX MCP SSE Service starting up...")
        await mcp_server.startup()
        logger.info("MIRIX MCP SSE Service started successfully")
    
    # 关闭事件
    @app.on_event("shutdown")
    async def shutdown_event():
        """应用关闭事件"""
        logger.info("MIRIX MCP SSE Service shutting down...")
        await mcp_server.shutdown()
        logger.info("MIRIX MCP SSE Service shut down successfully")
    
    return app

def main():
    """主函数"""
    settings = get_settings()
    
    logger.info(f"Starting MIRIX MCP SSE Service on {settings.host}:{settings.port}")
    logger.info(f"MIRIX Backend URL: {settings.mirix_backend_url}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # 创建应用
    app = create_app()
    
    # 启动服务器
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=settings.debug,
        reload=settings.debug and settings.reload,
    )

if __name__ == "__main__":
    main()