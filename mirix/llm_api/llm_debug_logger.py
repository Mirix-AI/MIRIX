"""
LLM Debug Logger - 用于详细记录 LLM 调用和响应的调试工具

这个模块提供了详细的日志记录功能，用于跟踪：
1. 发送给 LLM 的请求内容
2. LLM 返回的响应内容
3. 请求和响应的元数据
4. 错误和异常信息
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mirix.log import get_logger

class LLMDebugLogger:
    """LLM 调试日志记录器"""
    
    def __init__(self, log_dir: Optional[str] = None, enable_file_logging: bool = True):
        """
        初始化 LLM 调试日志记录器
        
        Args:
            log_dir: 日志文件保存目录，默认为 ./logs/llm_debug
            enable_file_logging: 是否启用文件日志记录
        """
        # 创建独立的日志记录器，避免与主日志系统冲突
        self.logger = logging.getLogger("Mirix.LLMDebug")
        self.logger.setLevel(logging.INFO)
        
        # 清除所有现有的处理器，确保只输出到文件
        self.logger.handlers.clear()
        
        # 防止日志传播到父日志记录器（避免控制台输出）
        self.logger.propagate = False
        
        self.enable_file_logging = enable_file_logging
        
        if enable_file_logging:
            if log_dir is None:
                log_dir = "./logs/llm_debug"
            
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建文件处理器
            self._setup_file_handlers()
    
    def _setup_file_handlers(self):
        """设置文件处理器"""
        # 请求日志文件
        request_log_file = self.log_dir / f"llm_requests_{datetime.now().strftime('%Y%m%d')}.log"
        self.request_handler = logging.FileHandler(request_log_file, encoding='utf-8')
        self.request_handler.setLevel(logging.INFO)
        request_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.request_handler.setFormatter(request_formatter)
        
        # 响应日志文件
        response_log_file = self.log_dir / f"llm_responses_{datetime.now().strftime('%Y%m%d')}.log"
        self.response_handler = logging.FileHandler(response_log_file, encoding='utf-8')
        self.response_handler.setLevel(logging.INFO)
        response_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.response_handler.setFormatter(response_formatter)
        
        # 错误日志文件
        error_log_file = self.log_dir / f"llm_errors_{datetime.now().strftime('%Y%m%d')}.log"
        self.error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        self.error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.error_handler.setFormatter(error_formatter)
        
        # 将处理器添加到日志记录器
        self.logger.addHandler(self.request_handler)
        self.logger.addHandler(self.response_handler)
        self.logger.addHandler(self.error_handler)
    
    def log_request(
        self,
        model_name: str,
        endpoint: str,
        request_data: Dict[str, Any],
        request_id: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        记录 LLM 请求
        
        Args:
            model_name: 模型名称
            endpoint: API 端点
            request_data: 请求数据
            request_id: 请求 ID（用于关联请求和响应）
            additional_info: 额外信息
        """
        if not request_id:
            request_id = f"req_{int(time.time() * 1000)}"
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "model_name": model_name,
            "endpoint": endpoint,
            "request_data": request_data,
            "additional_info": additional_info or {}
        }
        
        # 只输出到文件，不在控制台显示
        # 注释掉控制台输出，避免混淆终端输出
        
        # 文件输出
        if self.enable_file_logging:
            request_logger = logging.getLogger(f"Mirix.LLMDebug.Requests.{request_id}")
            request_logger.setLevel(logging.INFO)
            request_logger.propagate = False
            
            # 清除现有处理器，只添加请求处理器
            request_logger.handlers.clear()
            request_logger.addHandler(self.request_handler)
            
            # 记录请求信息
            request_logger.info(f"🚀 LLM Request [{request_id}]")
            request_logger.info(f"   Model: {model_name}")
            request_logger.info(f"   Endpoint: {endpoint}")
            request_logger.info(f"   Messages Count: {len(request_data.get('messages', []))}")
            
            if 'messages' in request_data:
                for i, msg in enumerate(request_data['messages']):
                    request_logger.info(f"   Message {i+1}: {msg.get('role', 'unknown')} - {str(msg.get('content', ''))[:100]}...")
            
            if 'tools' in request_data and request_data['tools']:
                request_logger.info(f"   Tools: {len(request_data['tools'])} tools available")
                for tool in request_data['tools']:
                    if isinstance(tool, dict) and 'function' in tool:
                        request_logger.info(f"     - {tool['function'].get('name', 'unknown')}")
            
            # 记录完整的请求数据
            request_logger.info("=== FULL REQUEST DATA ===")
            request_logger.info(json.dumps(log_data, ensure_ascii=False, indent=2))
        
        return request_id
    
    def log_response(
        self,
        request_id: str,
        response_data: Dict[str, Any],
        response_time_ms: Optional[float] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        记录 LLM 响应
        
        Args:
            request_id: 关联的请求 ID
            response_data: 响应数据
            response_time_ms: 响应时间（毫秒）
            additional_info: 额外信息
        """
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "response_data": response_data,
            "response_time_ms": response_time_ms,
            "additional_info": additional_info or {}
        }
        
        # 只输出到文件，不在控制台显示
        # 注释掉控制台输出，避免混淆终端输出
        
        # 文件输出
        if self.enable_file_logging:
            response_logger = logging.getLogger(f"Mirix.LLMDebug.Responses.{request_id}")
            response_logger.setLevel(logging.INFO)
            response_logger.propagate = False
            
            # 清除现有处理器，只添加响应处理器
            response_logger.handlers.clear()
            response_logger.addHandler(self.response_handler)
            
            # 记录响应信息
            response_logger.info(f"📥 LLM Response [{request_id}]")
            if response_time_ms:
                response_logger.info(f"   Response Time: {response_time_ms:.2f}ms")
            
            if 'choices' in response_data:
                choices = response_data['choices']
                response_logger.info(f"   Choices Count: {len(choices)}")
                
                for i, choice in enumerate(choices):
                    if 'message' in choice:
                        msg = choice['message']
                        response_logger.info(f"   Choice {i+1}: {msg.get('role', 'unknown')}")
                        
                        if 'content' in msg and msg['content']:
                            content = str(msg['content'])
                            response_logger.info(f"     Content: {content[:200]}{'...' if len(content) > 200 else ''}")
                        
                        if 'tool_calls' in msg and msg['tool_calls']:
                            response_logger.info(f"     Tool Calls: {len(msg['tool_calls'])}")
                            for j, tool_call in enumerate(msg['tool_calls']):
                                if isinstance(tool_call, dict):
                                    func_name = tool_call.get('function', {}).get('name', 'unknown')
                                    func_args = tool_call.get('function', {}).get('arguments', '')
                                    response_logger.info(f"       Tool {j+1}: {func_name}")
                                    response_logger.info(f"         Args: {func_args[:100]}{'...' if len(func_args) > 100 else ''}")
            
            if 'usage' in response_data:
                usage = response_data['usage']
                response_logger.info(f"   Usage: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion = {usage.get('total_tokens', 0)} total")
            
            # 记录完整的响应数据
            response_logger.info("=== FULL RESPONSE DATA ===")
            response_logger.info(json.dumps(log_data, ensure_ascii=False, indent=2))
    
    def log_error(
        self,
        request_id: str,
        error: Exception,
        error_context: Optional[Dict[str, Any]] = None
    ):
        """
        记录 LLM 错误
        
        Args:
            request_id: 关联的请求 ID
            error: 错误对象
            error_context: 错误上下文
        """
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_context": error_context or {}
        }
        
        # 只输出到文件，不在控制台显示
        # 注释掉控制台输出，避免混淆终端输出
        
        # 文件输出
        if self.enable_file_logging:
            error_logger = logging.getLogger(f"Mirix.LLMDebug.Errors.{request_id}")
            error_logger.setLevel(logging.ERROR)
            error_logger.propagate = False
            
            # 清除现有处理器，只添加错误处理器
            error_logger.handlers.clear()
            error_logger.addHandler(self.error_handler)
            
            # 记录错误信息
            error_logger.error(f"❌ LLM Error [{request_id}]")
            error_logger.error(f"   Error Type: {type(error).__name__}")
            error_logger.error(f"   Error Message: {str(error)}")
            
            if error_context:
                error_logger.error(f"   Error Context: {error_context}")
            
            # 记录完整的错误数据
            error_logger.error("=== FULL ERROR DATA ===")
            error_logger.error(json.dumps(log_data, ensure_ascii=False, indent=2))
    
    def log_json_parse_error(
        self,
        request_id: str,
        json_string: str,
        error: Exception,
        context: str = "unknown"
    ):
        """
        记录 JSON 解析错误
        
        Args:
            request_id: 关联的请求 ID
            json_string: 解析失败的 JSON 字符串
            error: JSON 解析错误
            context: 错误上下文
        """
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "json_string": json_string,
            "json_length": len(json_string)
        }
        
        # 控制台输出
        self.logger.error(f"🔧 JSON Parse Error [{request_id}] - {context}")
        self.logger.error(f"   Error: {str(error)}")
        self.logger.error(f"   JSON Length: {len(json_string)}")
        self.logger.error(f"   JSON Preview: {json_string[:200]}{'...' if len(json_string) > 200 else ''}")
        
        # 文件输出
        if self.enable_file_logging:
            error_logger = logging.getLogger(f"Mirix.LLMDebug.Errors")
            error_logger.addHandler(self.error_handler)
            error_logger.setLevel(logging.ERROR)
            error_logger.error(json.dumps(log_data, ensure_ascii=False, indent=2))
            error_logger.removeHandler(self.error_handler)
    
    def save_debug_session(
        self,
        session_id: str,
        requests: List[Dict[str, Any]],
        responses: List[Dict[str, Any]],
        errors: List[Dict[str, Any]]
    ):
        """
        保存完整的调试会话
        
        Args:
            session_id: 会话 ID
            requests: 请求列表
            responses: 响应列表
            errors: 错误列表
        """
        if not self.enable_file_logging:
            return
        
        session_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "requests": requests,
            "responses": responses,
            "errors": errors,
            "summary": {
                "total_requests": len(requests),
                "total_responses": len(responses),
                "total_errors": len(errors)
            }
        }
        
        session_file = self.log_dir / f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 Debug session saved: {session_file}")


# 全局调试日志记录器实例
_debug_logger = None

def get_llm_debug_logger() -> LLMDebugLogger:
    """获取全局 LLM 调试日志记录器实例"""
    global _debug_logger
    if _debug_logger is None:
        # 从环境变量读取配置
        log_dir = os.environ.get('LLM_DEBUG_LOG_DIR', './logs/llm_debug')
        enable_file_logging = os.environ.get('LLM_DEBUG_ENABLE_FILE', 'true').lower() == 'true'
        _debug_logger = LLMDebugLogger(log_dir=log_dir, enable_file_logging=enable_file_logging)
    return _debug_logger

def enable_llm_debug_logging(log_dir: Optional[str] = None, enable_file_logging: bool = True):
    """
    启用 LLM 调试日志记录
    
    Args:
        log_dir: 日志文件保存目录
        enable_file_logging: 是否启用文件日志记录
    """
    global _debug_logger
    _debug_logger = LLMDebugLogger(log_dir=log_dir, enable_file_logging=enable_file_logging)
    _debug_logger.logger.info("🔍 LLM Debug Logging Enabled")

def disable_llm_debug_logging():
    """禁用 LLM 调试日志记录"""
    global _debug_logger
    _debug_logger = None
