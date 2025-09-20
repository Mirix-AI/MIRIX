"""
MCP SSE Service 配置管理

使用Pydantic Settings管理配置，支持环境变量和.env文件。
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator

class Settings(BaseSettings):
    """MCP SSE服务配置"""
    
    # 服务配置
    host: str = Field(default="0.0.0.0", env="MCP_SSE_HOST")
    port: int = Field(default=8080, env="MCP_SSE_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    reload: bool = Field(default=False, env="RELOAD")
    
    # 日志配置
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    
    # MIRIX后端配置
    mirix_backend_url: str = Field(
        default="http://localhost:47283",
        env="MIRIX_BACKEND_URL",
        description="MIRIX 后端服务 URL"
    )
    mirix_backend_timeout: int = Field(default=30, env="MIRIX_BACKEND_TIMEOUT")
    
    # CORS配置
    allowed_origins: List[str] = Field(
        default=["*"],
        env="ALLOWED_ORIGINS"
    )
    
    # MCP协议配置
    mcp_version: str = Field(default="2024-11-05", env="MCP_VERSION")
    max_connections: int = Field(default=100, env="MAX_CONNECTIONS")
    connection_timeout: int = Field(default=300, env="CONNECTION_TIMEOUT")  # 5分钟
    
    # SSE配置
    sse_heartbeat_interval: int = Field(default=30, env="SSE_HEARTBEAT_INTERVAL")  # 心跳间隔（秒）
    sse_retry_interval: int = Field(default=5000, env="SSE_RETRY_INTERVAL")  # 重试间隔（毫秒）
    sse_max_message_size: int = Field(default=1024*1024, env="SSE_MAX_MESSAGE_SIZE")  # 1MB
    
    # 安全配置
    api_key: Optional[str] = Field(default=None, env="MCP_API_KEY")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")  # 每分钟请求数
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # 时间窗口（秒）
    
    # 缓存配置
    enable_cache: bool = Field(default=True, env="ENABLE_CACHE")
    cache_ttl: int = Field(default=300, env="CACHE_TTL")  # 缓存TTL（秒）
    
    # 会话管理配置
    max_sessions: int = Field(default=100, env="MAX_SESSIONS")  # 最大会话数
    session_timeout: int = Field(default=3600, env="SESSION_TIMEOUT")  # 会话超时时间（秒）
    session_cleanup_interval: int = Field(default=60, env="SESSION_CLEANUP_INTERVAL")  # 会话清理间隔（秒）
    
    # 记忆管理配置
    default_user_id: str = Field(default="default_user", env="DEFAULT_USER_ID")
    ai_model: str = Field(default="gpt-4", env="AI_MODEL")
    auto_categorize_memory: bool = Field(default=True, env="AUTO_CATEGORIZE_MEMORY")
    default_memory_type: str = Field(default="semantic", env="DEFAULT_MEMORY_TYPE")
    memory_search_limit: int = Field(default=10, env="MEMORY_SEARCH_LIMIT")
    
    @validator("allowed_origins", pre=True)
    def parse_allowed_origins(cls, v):
        """解析CORS允许的源"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """验证日志级别"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
    
    @validator("log_format")
    def validate_log_format(cls, v):
        """验证日志格式"""
        valid_formats = ["json", "text"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Invalid log format: {v}. Must be one of {valid_formats}")
        return v.lower()
    
    @validator("mirix_backend_url")
    def validate_backend_url(cls, v):
        """验证后端URL"""
        if not v.startswith(("http://", "https://")):
            raise ValueError("MIRIX backend URL must start with http:// or https://")
        return v.rstrip("/")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # 忽略额外的字段

# 全局设置实例
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """获取配置实例（单例模式）"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def reload_settings() -> Settings:
    """重新加载配置"""
    global _settings
    _settings = Settings()
    return _settings