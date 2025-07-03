"""
Cache service for FM-LLM Solver.

Provides caching functionality for improved performance.
"""

from typing import Any, Optional
import json
import time

from fm_llm_solver.core.interfaces import Cache
from fm_llm_solver.core.logging import get_logger


class MemoryCache(Cache):
    """Simple in-memory cache implementation."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache = {}
        self.expiry = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            # Check if expired
            if key in self.expiry and time.time() > self.expiry[key]:
                del self.cache[key]
                del self.expiry[key]
                return None
            return self.cache[key]
        return None
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        self.cache[key] = value
        if ttl:
            self.expiry[key] = time.time() + ttl
            
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
            if key in self.expiry:
                del self.expiry[key]
            return True
        return False
        
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.expiry.clear()


class RedisCache(Cache):
    """Redis-based cache implementation."""
    
    def __init__(self, redis_client):
        self.logger = get_logger(__name__)
        self.redis = redis_client
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            value = self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            self.logger.error(f"Redis get error: {e}")
            return None
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis cache."""
        try:
            serialized = json.dumps(value)
            if ttl:
                self.redis.setex(key, ttl, serialized)
            else:
                self.redis.set(key, serialized)
        except Exception as e:
            self.logger.error(f"Redis set error: {e}")
            
    def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        try:
            result = self.redis.delete(key)
            return result > 0
        except Exception as e:
            self.logger.error(f"Redis delete error: {e}")
            return False
            
    def clear(self) -> None:
        """Clear all Redis cache entries."""
        try:
            self.redis.flushdb()
        except Exception as e:
            self.logger.error(f"Redis clear error: {e}")


class CacheService:
    """Main cache service that can use different backends."""
    
    def __init__(self, cache_impl: Cache):
        self.cache = cache_impl
        self.logger = get_logger(__name__)
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.cache.get(key)
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        self.cache.set(key, value, ttl)
        
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        return self.cache.delete(key)
        
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear() 