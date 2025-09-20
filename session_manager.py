"""
会话管理器

管理MCP SSE连接的会话，包括会话创建、销毁、消息队列等。
"""

import asyncio
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .config import Settings
from .logging_config import LoggerMixin

@dataclass
class Session:
    """会话信息"""
    session_id: str
    client_ip: str
    created_at: datetime
    last_activity: datetime
    is_initialized: bool = False
    message_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue())
    metadata: Dict[str, Any] = field(default_factory=dict)

class SessionManager(LoggerMixin):
    """会话管理器"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sessions: Dict[str, Session] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def startup(self):
        """启动会话管理器"""
        self.logger.info("Starting session manager")
        
        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("Session manager started")
    
    async def shutdown(self):
        """关闭会话管理器"""
        self.logger.info("Shutting down session manager")
        
        # 停止清理任务
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 清理所有会话
        async with self._lock:
            session_ids = list(self.sessions.keys())
            for session_id in session_ids:
                await self._remove_session_internal(session_id)
        
        self.logger.info("Session manager shut down")
    
    async def create_session(self, session_id: str, client_ip: str) -> Session:
        """创建会话"""
        async with self._lock:
            # 检查会话是否已存在
            if session_id in self.sessions:
                self.logger.warning("Session already exists", session_id=session_id)
                return self.sessions[session_id]
            
            # 检查会话数量限制
            if len(self.sessions) >= self.settings.max_sessions:
                # 清理最旧的会话
                await self._cleanup_oldest_session()
            
            # 创建新会话
            now = datetime.utcnow()
            session = Session(
                session_id=session_id,
                client_ip=client_ip,
                created_at=now,
                last_activity=now
            )
            
            self.sessions[session_id] = session
            
            self.logger.info("Session created", 
                           session_id=session_id, 
                           client_ip=client_ip,
                           total_sessions=len(self.sessions))
            
            return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        async with self._lock:
            return self.sessions.get(session_id)
    
    async def remove_session(self, session_id: str) -> bool:
        """移除会话"""
        async with self._lock:
            return await self._remove_session_internal(session_id)
    
    async def _remove_session_internal(self, session_id: str) -> bool:
        """内部移除会话方法（不加锁）"""
        session = self.sessions.pop(session_id, None)
        if session:
            # 清空消息队列
            while not session.message_queue.empty():
                try:
                    session.message_queue.get_nowait()
                    session.message_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            
            self.logger.info("Session removed", 
                           session_id=session_id,
                           total_sessions=len(self.sessions))
            return True
        
        return False
    
    async def update_session_activity(self, session_id: str) -> bool:
        """更新会话活动时间"""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.last_activity = datetime.utcnow()
                return True
            return False
    
    async def mark_session_initialized(self, session_id: str) -> bool:
        """标记会话已初始化"""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.is_initialized = True
                self.logger.debug("Session marked as initialized", session_id=session_id)
                return True
            return False
    
    async def get_message_queue(self, session_id: str) -> Optional[asyncio.Queue]:
        """获取会话消息队列"""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session:
                return session.message_queue
            return None
    
    async def set_session_metadata(self, session_id: str, key: str, value: Any) -> bool:
        """设置会话元数据"""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.metadata[key] = value
                return True
            return False
    
    async def get_session_metadata(self, session_id: str, key: str) -> Optional[Any]:
        """获取会话元数据"""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session:
                return session.metadata.get(key)
            return None
    
    async def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """列出所有会话"""
        async with self._lock:
            result = {}
            for session_id, session in self.sessions.items():
                result[session_id] = {
                    "session_id": session.session_id,
                    "client_ip": session.client_ip,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "is_initialized": session.is_initialized,
                    "queue_size": session.message_queue.qsize(),
                    "metadata": session.metadata
                }
            return result
    
    async def get_session_count(self) -> int:
        """获取会话数量"""
        async with self._lock:
            return len(self.sessions)
    
    async def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                await asyncio.sleep(self.settings.session_cleanup_interval)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in cleanup loop", error=str(e))
    
    async def _cleanup_expired_sessions(self):
        """清理过期会话"""
        now = datetime.utcnow()
        expired_threshold = now - timedelta(seconds=self.settings.session_timeout)
        
        async with self._lock:
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                if session.last_activity < expired_threshold:
                    expired_sessions.append(session_id)
            
            # 移除过期会话
            for session_id in expired_sessions:
                await self._remove_session_internal(session_id)
                self.logger.info("Expired session removed", session_id=session_id)
            
            if expired_sessions:
                self.logger.info("Cleanup completed", 
                               removed_sessions=len(expired_sessions),
                               remaining_sessions=len(self.sessions))
    
    async def _cleanup_oldest_session(self):
        """清理最旧的会话"""
        if not self.sessions:
            return
        
        # 找到最旧的会话
        oldest_session_id = min(self.sessions.keys(), 
                               key=lambda sid: self.sessions[sid].created_at)
        
        await self._remove_session_internal(oldest_session_id)
        self.logger.info("Oldest session removed due to limit", 
                        session_id=oldest_session_id)
    
    async def broadcast_message(self, message: Dict[str, Any], 
                              exclude_session: Optional[str] = None):
        """向所有会话广播消息"""
        async with self._lock:
            broadcast_count = 0
            
            for session_id, session in self.sessions.items():
                if exclude_session and session_id == exclude_session:
                    continue
                
                try:
                    await session.message_queue.put(message)
                    broadcast_count += 1
                except Exception as e:
                    self.logger.error("Failed to broadcast message to session", 
                                    session_id=session_id, error=str(e))
            
            self.logger.debug("Message broadcasted", 
                            recipients=broadcast_count,
                            excluded=exclude_session)
    
    async def send_message_to_session(self, session_id: str, message: Dict[str, Any]) -> bool:
        """向指定会话发送消息"""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session:
                try:
                    await session.message_queue.put(message)
                    return True
                except Exception as e:
                    self.logger.error("Failed to send message to session", 
                                    session_id=session_id, error=str(e))
            return False
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """获取会话统计信息"""
        async with self._lock:
            now = datetime.utcnow()
            
            total_sessions = len(self.sessions)
            initialized_sessions = sum(1 for s in self.sessions.values() if s.is_initialized)
            
            # 计算平均会话时长
            if self.sessions:
                session_durations = [
                    (now - session.created_at).total_seconds()
                    for session in self.sessions.values()
                ]
                avg_duration = sum(session_durations) / len(session_durations)
            else:
                avg_duration = 0
            
            # 计算消息队列统计
            total_queued_messages = sum(s.message_queue.qsize() for s in self.sessions.values())
            
            return {
                "total_sessions": total_sessions,
                "initialized_sessions": initialized_sessions,
                "uninitialized_sessions": total_sessions - initialized_sessions,
                "average_session_duration_seconds": avg_duration,
                "total_queued_messages": total_queued_messages,
                "max_sessions": self.settings.max_sessions,
                "session_timeout_seconds": self.settings.session_timeout
            }