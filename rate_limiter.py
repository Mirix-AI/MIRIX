"""
速率限制器

实现基于令牌桶算法的速率限制，防止API滥用。
"""

import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass
from collections import defaultdict

from .logging_config import LoggerMixin

@dataclass
class TokenBucket:
    """令牌桶"""
    capacity: int  # 桶容量
    tokens: float  # 当前令牌数
    refill_rate: float  # 令牌补充速率（每秒）
    last_refill: float  # 上次补充时间

class RateLimiter(LoggerMixin):
    """速率限制器"""
    
    def __init__(self, requests_per_window: int, window_seconds: int):
        """
        初始化速率限制器
        
        Args:
            requests_per_window: 时间窗口内允许的请求数
            window_seconds: 时间窗口大小（秒）
        """
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.refill_rate = requests_per_window / window_seconds  # 每秒补充的令牌数
        
        # 每个客户端的令牌桶
        self.buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
        
        # 清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = max(60, window_seconds)  # 清理间隔
        
        self.logger.info("Rate limiter initialized", 
                        requests_per_window=requests_per_window,
                        window_seconds=window_seconds,
                        refill_rate=self.refill_rate)
    
    async def start(self):
        """启动速率限制器"""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.logger.info("Rate limiter started")
    
    async def stop(self):
        """停止速率限制器"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            self.logger.info("Rate limiter stopped")
    
    async def allow_request(self, client_id: str, tokens_required: int = 1) -> bool:
        """
        检查是否允许请求
        
        Args:
            client_id: 客户端标识符
            tokens_required: 需要的令牌数
            
        Returns:
            是否允许请求
        """
        async with self._lock:
            bucket = await self._get_or_create_bucket(client_id)
            
            # 补充令牌
            await self._refill_bucket(bucket)
            
            # 检查是否有足够的令牌
            if bucket.tokens >= tokens_required:
                bucket.tokens -= tokens_required
                
                self.logger.debug("Request allowed", 
                                client_id=client_id,
                                tokens_required=tokens_required,
                                remaining_tokens=bucket.tokens)
                return True
            else:
                self.logger.debug("Request denied - insufficient tokens", 
                                client_id=client_id,
                                tokens_required=tokens_required,
                                available_tokens=bucket.tokens)
                return False
    
    async def get_remaining_tokens(self, client_id: str) -> float:
        """获取剩余令牌数"""
        async with self._lock:
            bucket = await self._get_or_create_bucket(client_id)
            await self._refill_bucket(bucket)
            return bucket.tokens
    
    async def get_time_until_next_token(self, client_id: str) -> float:
        """获取下一个令牌的等待时间（秒）"""
        async with self._lock:
            bucket = await self._get_or_create_bucket(client_id)
            await self._refill_bucket(bucket)
            
            if bucket.tokens >= 1:
                return 0.0
            else:
                # 计算需要等待的时间
                tokens_needed = 1 - bucket.tokens
                return tokens_needed / self.refill_rate
    
    async def reset_client_limit(self, client_id: str):
        """重置客户端限制"""
        async with self._lock:
            if client_id in self.buckets:
                bucket = self.buckets[client_id]
                bucket.tokens = bucket.capacity
                bucket.last_refill = time.time()
                
                self.logger.info("Client rate limit reset", client_id=client_id)
    
    async def get_client_stats(self, client_id: str) -> Dict[str, float]:
        """获取客户端统计信息"""
        async with self._lock:
            bucket = await self._get_or_create_bucket(client_id)
            await self._refill_bucket(bucket)
            
            return {
                "capacity": bucket.capacity,
                "current_tokens": bucket.tokens,
                "refill_rate": self.refill_rate,
                "utilization": (bucket.capacity - bucket.tokens) / bucket.capacity
            }
    
    async def get_all_stats(self) -> Dict[str, any]:
        """获取所有统计信息"""
        async with self._lock:
            stats = {
                "total_clients": len(self.buckets),
                "requests_per_window": self.requests_per_window,
                "window_seconds": self.window_seconds,
                "refill_rate": self.refill_rate,
                "clients": {}
            }
            
            for client_id, bucket in self.buckets.items():
                await self._refill_bucket(bucket)
                stats["clients"][client_id] = {
                    "current_tokens": bucket.tokens,
                    "utilization": (bucket.capacity - bucket.tokens) / bucket.capacity,
                    "last_refill": bucket.last_refill
                }
            
            return stats
    
    async def _get_or_create_bucket(self, client_id: str) -> TokenBucket:
        """获取或创建令牌桶"""
        if client_id not in self.buckets:
            now = time.time()
            self.buckets[client_id] = TokenBucket(
                capacity=self.requests_per_window,
                tokens=self.requests_per_window,  # 初始满桶
                refill_rate=self.refill_rate,
                last_refill=now
            )
            
            self.logger.debug("Created new token bucket", 
                            client_id=client_id,
                            capacity=self.requests_per_window)
        
        return self.buckets[client_id]
    
    async def _refill_bucket(self, bucket: TokenBucket):
        """补充令牌桶"""
        now = time.time()
        time_passed = now - bucket.last_refill
        
        if time_passed > 0:
            # 计算应该补充的令牌数
            tokens_to_add = time_passed * bucket.refill_rate
            
            # 更新令牌数，不超过容量
            bucket.tokens = min(bucket.capacity, bucket.tokens + tokens_to_add)
            bucket.last_refill = now
    
    async def _cleanup_loop(self):
        """清理循环，移除长时间未使用的令牌桶"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_inactive_buckets()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in rate limiter cleanup loop", error=str(e))
    
    async def _cleanup_inactive_buckets(self):
        """清理不活跃的令牌桶"""
        now = time.time()
        inactive_threshold = now - (self._cleanup_interval * 2)  # 2个清理周期
        
        async with self._lock:
            inactive_clients = []
            
            for client_id, bucket in self.buckets.items():
                # 如果令牌桶满了且长时间未使用，则认为不活跃
                if (bucket.tokens >= bucket.capacity * 0.9 and 
                    bucket.last_refill < inactive_threshold):
                    inactive_clients.append(client_id)
            
            # 移除不活跃的客户端
            for client_id in inactive_clients:
                del self.buckets[client_id]
            
            if inactive_clients:
                self.logger.info("Cleaned up inactive rate limit buckets", 
                               count=len(inactive_clients),
                               remaining_clients=len(self.buckets))

class AdaptiveRateLimiter(RateLimiter):
    """自适应速率限制器"""
    
    def __init__(self, base_requests_per_window: int, window_seconds: int, 
                 max_multiplier: float = 2.0, min_multiplier: float = 0.5):
        """
        初始化自适应速率限制器
        
        Args:
            base_requests_per_window: 基础请求数
            window_seconds: 时间窗口
            max_multiplier: 最大倍数
            min_multiplier: 最小倍数
        """
        super().__init__(base_requests_per_window, window_seconds)
        self.base_requests = base_requests_per_window
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        
        # 客户端行为跟踪
        self.client_behavior: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "success_rate": 1.0,
            "avg_interval": 1.0,
            "burst_count": 0,
            "last_request": 0.0
        })
    
    async def allow_request(self, client_id: str, tokens_required: int = 1) -> bool:
        """允许请求（带自适应逻辑）"""
        # 更新客户端行为
        await self._update_client_behavior(client_id)
        
        # 计算自适应倍数
        multiplier = await self._calculate_adaptive_multiplier(client_id)
        
        # 临时调整令牌桶容量
        async with self._lock:
            bucket = await self._get_or_create_bucket(client_id)
            original_capacity = bucket.capacity
            
            # 调整容量
            new_capacity = int(self.base_requests * multiplier)
            bucket.capacity = max(1, min(int(self.base_requests * self.max_multiplier), new_capacity))
            
            # 如果容量增加，补充令牌
            if bucket.capacity > original_capacity:
                bucket.tokens = min(bucket.capacity, bucket.tokens + (bucket.capacity - original_capacity))
            
            result = await super().allow_request(client_id, tokens_required)
            
            self.logger.debug("Adaptive rate limit applied", 
                            client_id=client_id,
                            multiplier=multiplier,
                            original_capacity=original_capacity,
                            new_capacity=bucket.capacity,
                            allowed=result)
            
            return result
    
    async def _update_client_behavior(self, client_id: str):
        """更新客户端行为统计"""
        now = time.time()
        behavior = self.client_behavior[client_id]
        
        # 更新请求间隔
        if behavior["last_request"] > 0:
            interval = now - behavior["last_request"]
            behavior["avg_interval"] = (behavior["avg_interval"] * 0.9 + interval * 0.1)
        
        behavior["last_request"] = now
    
    async def _calculate_adaptive_multiplier(self, client_id: str) -> float:
        """计算自适应倍数"""
        behavior = self.client_behavior[client_id]
        
        # 基于成功率调整
        success_factor = behavior["success_rate"]
        
        # 基于请求间隔调整（间隔越大，限制越松）
        interval_factor = min(2.0, behavior["avg_interval"] / (self.window_seconds / self.base_requests))
        
        # 综合计算倍数
        multiplier = success_factor * interval_factor
        
        # 限制在范围内
        return max(self.min_multiplier, min(self.max_multiplier, multiplier))
    
    async def record_request_result(self, client_id: str, success: bool):
        """记录请求结果"""
        behavior = self.client_behavior[client_id]
        
        # 更新成功率（指数移动平均）
        success_value = 1.0 if success else 0.0
        behavior["success_rate"] = behavior["success_rate"] * 0.9 + success_value * 0.1