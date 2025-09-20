"""
MIRIX MCP SSE Service

提供对外的MCP (Model Context Protocol) 服务接口，支持SSE (Server-Sent Events) 传输。
该服务作为MIRIX系统的对外接口，允许其他应用通过MCP协议与MIRIX进行交互。

主要功能：
1. MCP协议实现
2. SSE流式传输
3. 工具调用代理
4. 会话管理
5. 错误处理和日志记录
"""

__version__ = "0.1.0"
__author__ = "MIRIX Team"