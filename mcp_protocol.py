"""
MCP (Model Context Protocol) 协议数据模型

定义MCP协议中使用的所有数据结构和消息类型。
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

class MCPMessageType(str, Enum):
    """MCP消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"

class MCPErrorCode(int, Enum):
    """MCP错误代码"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

class MCPError(BaseModel):
    """MCP错误"""
    code: int = Field(..., description="错误代码")
    message: str = Field(..., description="错误消息")
    data: Optional[Any] = Field(None, description="错误数据")

class MCPMessage(BaseModel):
    """MCP消息基类"""
    jsonrpc: str = Field("2.0", description="JSON-RPC版本")

class MCPRequest(MCPMessage):
    """MCP请求"""
    id: Union[str, int] = Field(..., description="请求ID")
    method: str = Field(..., description="方法名")
    params: Optional[Dict[str, Any]] = Field(None, description="参数")

class MCPResponse(MCPMessage):
    """MCP响应"""
    id: Union[str, int] = Field(..., description="请求ID")
    result: Optional[Any] = Field(None, description="结果")
    error: Optional[MCPError] = Field(None, description="错误")

class MCPNotification(MCPMessage):
    """MCP通知"""
    method: str = Field(..., description="方法名")
    params: Optional[Dict[str, Any]] = Field(None, description="参数")

# 工具相关模型
class MCPTool(BaseModel):
    """MCP工具定义"""
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    inputSchema: Dict[str, Any] = Field(..., description="输入模式")

class MCPToolCall(BaseModel):
    """MCP工具调用"""
    name: str = Field(..., description="工具名称")
    arguments: Dict[str, Any] = Field(..., description="工具参数")

class MCPToolResult(BaseModel):
    """MCP工具结果"""
    content: List[Dict[str, Any]] = Field(..., description="结果内容")
    isError: Optional[bool] = Field(False, description="是否为错误")

# 资源相关模型
class MCPResource(BaseModel):
    """MCP资源定义"""
    uri: str = Field(..., description="资源URI")
    name: str = Field(..., description="资源名称")
    description: Optional[str] = Field(None, description="资源描述")
    mimeType: Optional[str] = Field(None, description="MIME类型")

class MCPResourceContent(BaseModel):
    """MCP资源内容"""
    uri: str = Field(..., description="资源URI")
    mimeType: Optional[str] = Field(None, description="MIME类型")
    text: Optional[str] = Field(None, description="文本内容")
    blob: Optional[str] = Field(None, description="二进制内容(base64)")

# 提示相关模型
class MCPPrompt(BaseModel):
    """MCP提示定义"""
    name: str = Field(..., description="提示名称")
    description: Optional[str] = Field(None, description="提示描述")
    arguments: Optional[List[Dict[str, Any]]] = Field(None, description="参数定义")

class MCPPromptMessage(BaseModel):
    """MCP提示消息"""
    role: str = Field(..., description="角色")
    content: Union[str, Dict[str, Any]] = Field(..., description="内容")

class MCPPromptResult(BaseModel):
    """MCP提示结果"""
    description: Optional[str] = Field(None, description="描述")
    messages: List[MCPPromptMessage] = Field(..., description="消息列表")

# 能力相关模型
class MCPCapabilities(BaseModel):
    """MCP能力"""
    tools: Optional[Dict[str, Any]] = Field(None, description="工具能力")
    resources: Optional[Dict[str, Any]] = Field(None, description="资源能力")
    prompts: Optional[Dict[str, Any]] = Field(None, description="提示能力")
    logging: Optional[Dict[str, Any]] = Field(None, description="日志能力")

class MCPServerInfo(BaseModel):
    """MCP服务器信息"""
    name: str = Field(..., description="服务器名称")
    version: str = Field(..., description="服务器版本")

class MCPClientInfo(BaseModel):
    """MCP客户端信息"""
    name: str = Field(..., description="客户端名称")
    version: str = Field(..., description="客户端版本")

# 初始化相关模型
class MCPInitializeParams(BaseModel):
    """MCP初始化参数"""
    protocolVersion: str = Field(..., description="协议版本")
    capabilities: MCPCapabilities = Field(..., description="客户端能力")
    clientInfo: MCPClientInfo = Field(..., description="客户端信息")

class MCPInitializeResult(BaseModel):
    """MCP初始化结果"""
    protocolVersion: str = Field(..., description="协议版本")
    capabilities: MCPCapabilities = Field(..., description="服务器能力")
    serverInfo: MCPServerInfo = Field(..., description="服务器信息")

# 日志相关模型
class MCPLogLevel(str, Enum):
    """日志级别"""
    DEBUG = "debug"
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    ALERT = "alert"
    EMERGENCY = "emergency"

class MCPLogEntry(BaseModel):
    """日志条目"""
    level: MCPLogLevel = Field(..., description="日志级别")
    data: Any = Field(..., description="日志数据")
    logger: Optional[str] = Field(None, description="日志器名称")

# 分页相关模型
class MCPPaginationParams(BaseModel):
    """分页参数"""
    cursor: Optional[str] = Field(None, description="游标")

class MCPPaginationResult(BaseModel):
    """分页结果"""
    nextCursor: Optional[str] = Field(None, description="下一页游标")

# 进度相关模型
class MCPProgress(BaseModel):
    """进度信息"""
    progress: int = Field(..., description="进度值", ge=0)
    total: Optional[int] = Field(None, description="总数", ge=0)

# 取消相关模型
class MCPCancelParams(BaseModel):
    """取消参数"""
    requestId: Union[str, int] = Field(..., description="请求ID")

# 常用方法名常量
class MCPMethods:
    """MCP方法名常量"""
    
    # 初始化
    INITIALIZE = "initialize"
    INITIALIZED = "notifications/initialized"
    
    # 工具
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    
    # 资源
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    RESOURCES_SUBSCRIBE = "resources/subscribe"
    RESOURCES_UNSUBSCRIBE = "resources/unsubscribe"
    RESOURCES_UPDATED = "notifications/resources/updated"
    
    # 提示
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"
    
    # 日志
    LOGGING_SET_LEVEL = "logging/setLevel"
    
    # 取消
    CANCEL_REQUEST = "notifications/cancelled"
    
    # 进度
    PROGRESS = "notifications/progress"

# 工厂函数
def create_request(method: str, params: Optional[Dict[str, Any]] = None, 
                  request_id: Optional[Union[str, int]] = None) -> MCPRequest:
    """创建MCP请求"""
    if request_id is None:
        import uuid
        request_id = str(uuid.uuid4())
    
    return MCPRequest(
        id=request_id,
        method=method,
        params=params
    )

def create_response(request_id: Union[str, int], result: Optional[Any] = None, 
                   error: Optional[MCPError] = None) -> MCPResponse:
    """创建MCP响应"""
    return MCPResponse(
        id=request_id,
        result=result,
        error=error
    )

def create_notification(method: str, params: Optional[Dict[str, Any]] = None) -> MCPNotification:
    """创建MCP通知"""
    return MCPNotification(
        method=method,
        params=params
    )

def create_error_response(request_id: Union[str, int], code: int, 
                         message: str, data: Optional[Any] = None) -> MCPResponse:
    """创建错误响应"""
    error = MCPError(code=code, message=message, data=data)
    return create_response(request_id, error=error)

# 验证函数
def validate_mcp_message(data: Dict[str, Any]) -> Union[MCPRequest, MCPResponse, MCPNotification]:
    """验证并解析MCP消息"""
    if "id" in data:
        if "method" in data:
            # 请求
            return MCPRequest(**data)
        else:
            # 响应
            return MCPResponse(**data)
    elif "method" in data:
        # 通知
        return MCPNotification(**data)
    else:
        raise ValueError("Invalid MCP message format")

# 序列化函数
def serialize_mcp_message(message: Union[MCPRequest, MCPResponse, MCPNotification]) -> Dict[str, Any]:
    """序列化MCP消息"""
    return message.dict(exclude_none=True)