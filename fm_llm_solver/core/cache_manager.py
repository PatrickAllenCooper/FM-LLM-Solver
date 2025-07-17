"""
Comprehensive caching system for FM-LLM-Solver.

Provides Redis integration, cache invalidation, performance optimization,
and multiple cache backends with fallback mechanisms.
"""

import pickle
import time
import hashlib
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

try:
    import redis
    from redis import Redis

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

from .config_manager import ConfigurationManager
from .logging_manager import get_logger
from .exceptions import CacheError


class CacheBackend(Enum):
    """Cache backend types."""

    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


class CacheStrategy(Enum):
    """Cache strategies."""

    LRU = "lru"  # Least Recently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


@dataclass
class CacheConfig:
    """Configuration for cache settings."""

    backend: CacheBackend = CacheBackend.MEMORY
    max_size: int = 1000
    default_ttl: int = 3600  # 1 hour
    redis_url: Optional[str] = None
    redis_db: int = 0
    redis_max_connections: int = 20
    strategy: CacheStrategy = CacheStrategy.LRU
    compression: bool = True
    serialization: str = "pickle"  # pickle, json, msgpack
    key_prefix: str = "fm_llm:"
    namespace: str = "default"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size: int = 0
    tags: List[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def is_valid(self) -> bool:
        """Check if entry is valid."""
        return not self.is_expired()

    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class CacheBackendInterface(ABC):
    """Interface for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""

    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries."""

    @abstractmethod
    def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern."""

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""


class MemoryCache(CacheBackendInterface):
    """In-memory cache implementation."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, CacheEntry] = {}
        self.logger = get_logger(__name__)
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            self.miss_count += 1
            return None

        entry = self.cache[key]

        if entry.is_expired():
            del self.cache[key]
            self.miss_count += 1
            return None

        entry.update_access()
        self.hit_count += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            # Calculate expiration time
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            elif self.config.default_ttl > 0:
                expires_at = time.time() + self.config.default_ttl

            # Calculate size
            size = len(pickle.dumps(value)) if self.config.compression else 0

            # Create cache entry
            entry = CacheEntry(
                key=key, value=value, created_at=time.time(), expires_at=expires_at, size=size
            )

            # Check if we need to evict entries
            if len(self.cache) >= self.config.max_size:
                self._evict_entries()

            self.cache[key] = entry
            return True

        except Exception as e:
            self.logger.error(f"Failed to set cache entry {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self.cache and not self.cache[key].is_expired()

    def clear(self) -> bool:
        """Clear all cache entries."""
        self.cache.clear()
        return True

    def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern."""
        import fnmatch

        return [key for key in self.cache.keys() if fnmatch.fnmatch(key, pattern)]

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0

        return {
            "backend": "memory",
            "entries": len(self.cache),
            "max_size": self.config.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "eviction_count": self.eviction_count,
            "total_size": sum(entry.size for entry in self.cache.values()),
        }

    def _evict_entries(self):
        """Evict entries based on strategy."""
        if self.config.strategy == CacheStrategy.LRU:
            # Remove least recently used
            lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)
            del self.cache[lru_key]
        elif self.config.strategy == CacheStrategy.FIFO:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
            del self.cache[oldest_key]
        elif self.config.strategy == CacheStrategy.TTL:
            # Remove expired entries first
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            if expired_keys:
                for key in expired_keys:
                    del self.cache[key]
            else:
                # If no expired entries, use LRU
                self._evict_entries()

        self.eviction_count += 1


class RedisCache(CacheBackendInterface):
    """Redis cache implementation."""

    def __init__(self, config: CacheConfig):
        if not HAS_REDIS:
            raise CacheError("Redis not available - install redis package")

        self.config = config
        self.logger = get_logger(__name__)

        # Initialize Redis connection
        self.redis = self._create_redis_client()
        self.hit_count = 0
        self.miss_count = 0

    def _create_redis_client(self) -> "Redis":
        """Create Redis client."""
        try:
            if self.config.redis_url:
                return redis.from_url(
                    self.config.redis_url,
                    db=self.config.redis_db,
                    max_connections=self.config.redis_max_connections,
                    decode_responses=False,
                )
            else:
                return redis.Redis(
                    host="localhost",
                    port=6379,
                    db=self.config.redis_db,
                    max_connections=self.config.redis_max_connections,
                    decode_responses=False,
                )
        except Exception as e:
            raise CacheError(f"Failed to create Redis client: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            full_key = f"{self.config.key_prefix}{self.config.namespace}:{key}"
            data = self.redis.get(full_key)

            if data is None:
                self.miss_count += 1
                return None

            self.hit_count += 1
            return pickle.loads(data)

        except Exception as e:
            self.logger.error(f"Failed to get cache entry {key}: {e}")
            self.miss_count += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            full_key = f"{self.config.key_prefix}{self.config.namespace}:{key}"
            data = pickle.dumps(value)

            if ttl is None:
                ttl = self.config.default_ttl

            if ttl > 0:
                return self.redis.setex(full_key, ttl, data)
            else:
                return self.redis.set(full_key, data)

        except Exception as e:
            self.logger.error(f"Failed to set cache entry {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            full_key = f"{self.config.key_prefix}{self.config.namespace}:{key}"
            return bool(self.redis.delete(full_key))
        except Exception as e:
            self.logger.error(f"Failed to delete cache entry {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            full_key = f"{self.config.key_prefix}{self.config.namespace}:{key}"
            return bool(self.redis.exists(full_key))
        except Exception as e:
            self.logger.error(f"Failed to check cache entry {key}: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            pattern = f"{self.config.key_prefix}{self.config.namespace}:*"
            keys = self.redis.keys(pattern)
            if keys:
                return bool(self.redis.delete(*keys))
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False

    def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern."""
        try:
            full_pattern = f"{self.config.key_prefix}{self.config.namespace}:{pattern}"
            keys = self.redis.keys(full_pattern)
            prefix_len = len(f"{self.config.key_prefix}{self.config.namespace}:")
            return [key.decode("utf-8")[prefix_len:] for key in keys]
        except Exception as e:
            self.logger.error(f"Failed to get cache keys: {e}")
            return []

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = self.redis.info()
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0

            return {
                "backend": "redis",
                "connected": True,
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            self.logger.error(f"Failed to get Redis stats: {e}")
            return {"backend": "redis", "connected": False, "error": str(e)}


class HybridCache(CacheBackendInterface):
    """Hybrid cache combining memory and Redis."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize L1 (memory) cache
        memory_config = CacheConfig(
            backend=CacheBackend.MEMORY,
            max_size=min(config.max_size, 100),  # Smaller L1 cache
            default_ttl=config.default_ttl,
            strategy=config.strategy,
        )
        self.l1_cache = MemoryCache(memory_config)

        # Initialize L2 (Redis) cache
        try:
            self.l2_cache = RedisCache(config)
            self.has_l2 = True
        except CacheError:
            self.logger.warning("Redis not available, using memory-only cache")
            self.l2_cache = None
            self.has_l2 = False

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 first, then L2)."""
        # Try L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            return value

        # Try L2 cache
        if self.has_l2:
            value = self.l2_cache.get(key)
            if value is not None:
                # Store in L1 for faster access
                self.l1_cache.set(key, value)
                return value

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in both caches."""
        # Set in L1 cache
        l1_success = self.l1_cache.set(key, value, ttl)

        # Set in L2 cache
        l2_success = True
        if self.has_l2:
            l2_success = self.l2_cache.set(key, value, ttl)

        return l1_success and l2_success

    def delete(self, key: str) -> bool:
        """Delete from both caches."""
        l1_success = self.l1_cache.delete(key)
        l2_success = True
        if self.has_l2:
            l2_success = self.l2_cache.delete(key)

        return l1_success and l2_success

    def exists(self, key: str) -> bool:
        """Check if key exists in either cache."""
        return self.l1_cache.exists(key) or (self.has_l2 and self.l2_cache.exists(key))

    def clear(self) -> bool:
        """Clear both caches."""
        l1_success = self.l1_cache.clear()
        l2_success = True
        if self.has_l2:
            l2_success = self.l2_cache.clear()

        return l1_success and l2_success

    def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys from both caches."""
        l1_keys = set(self.l1_cache.keys(pattern))
        l2_keys = set(self.l2_cache.keys(pattern)) if self.has_l2 else set()
        return list(l1_keys | l2_keys)

    def stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        l1_stats = self.l1_cache.stats()
        l2_stats = self.l2_cache.stats() if self.has_l2 else {}

        return {
            "backend": "hybrid",
            "l1_cache": l1_stats,
            "l2_cache": l2_stats,
            "has_l2": self.has_l2,
        }


class CacheManager:
    """
    Main cache manager with advanced features.

    Features:
    - Multiple cache backends (memory, Redis, hybrid)
    - Cache invalidation strategies
    - Performance monitoring
    - Automatic cache warming
    - Tagged cache entries
    - Cache compression
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or ConfigurationManager()
        self.logger = get_logger(__name__)

        # Load cache configuration
        self.cache_config = self._load_cache_config()

        # Initialize cache backend
        self.cache_backend = self._create_cache_backend()

        # Cache invalidation tracking
        self.invalidation_tags: Dict[str, List[str]] = {}

        self.logger.info(
            f"Cache manager initialized with {self.cache_config.backend.value} backend"
        )

    def _load_cache_config(self) -> CacheConfig:
        """Load cache configuration."""
        try:
            config = self.config_manager.load_config()
            cache_config = config.get("cache", {})

            return CacheConfig(
                backend=CacheBackend(cache_config.get("backend", "memory")),
                max_size=cache_config.get("max_size", 1000),
                default_ttl=cache_config.get("default_ttl", 3600),
                redis_url=cache_config.get("redis_url"),
                redis_db=cache_config.get("redis_db", 0),
                redis_max_connections=cache_config.get("redis_max_connections", 20),
                strategy=CacheStrategy(cache_config.get("strategy", "lru")),
                compression=cache_config.get("compression", True),
                key_prefix=cache_config.get("key_prefix", "fm_llm:"),
                namespace=cache_config.get("namespace", "default"),
            )
        except Exception as e:
            self.logger.warning(f"Failed to load cache config: {e}, using defaults")
            return CacheConfig()

    def _create_cache_backend(self) -> CacheBackendInterface:
        """Create cache backend based on configuration."""
        if self.cache_config.backend == CacheBackend.MEMORY:
            return MemoryCache(self.cache_config)
        elif self.cache_config.backend == CacheBackend.REDIS:
            return RedisCache(self.cache_config)
        elif self.cache_config.backend == CacheBackend.HYBRID:
            return HybridCache(self.cache_config)
        else:
            raise CacheError(f"Unknown cache backend: {self.cache_config.backend}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.cache_backend.get(key)

    def set(
        self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache with optional tags."""
        success = self.cache_backend.set(key, value, ttl)

        if success and tags:
            # Track tags for invalidation
            for tag in tags:
                if tag not in self.invalidation_tags:
                    self.invalidation_tags[tag] = []
                self.invalidation_tags[tag].append(key)

        return success

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        return self.cache_backend.delete(key)

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.cache_backend.exists(key)

    def clear(self) -> bool:
        """Clear all cache entries."""
        success = self.cache_backend.clear()
        if success:
            self.invalidation_tags.clear()
        return success

    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all cache entries with a specific tag."""
        if tag not in self.invalidation_tags:
            return 0

        keys_to_invalidate = self.invalidation_tags[tag]
        invalidated_count = 0

        for key in keys_to_invalidate:
            if self.cache_backend.delete(key):
                invalidated_count += 1

        # Remove tag tracking
        del self.invalidation_tags[tag]

        self.logger.info(f"Invalidated {invalidated_count} cache entries with tag '{tag}'")
        return invalidated_count

    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate all cache entries matching a pattern."""
        keys = self.cache_backend.keys(pattern)
        invalidated_count = 0

        for key in keys:
            if self.cache_backend.delete(key):
                invalidated_count += 1

        self.logger.info(
            f"Invalidated {invalidated_count} cache entries matching pattern '{pattern}'"
        )
        return invalidated_count

    def warm_cache(self, warm_func: Callable, keys: List[str], ttl: Optional[int] = None) -> int:
        """Warm cache with precomputed values."""
        warmed_count = 0

        for key in keys:
            try:
                value = warm_func(key)
                if value is not None:
                    if self.set(key, value, ttl):
                        warmed_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to warm cache for key '{key}': {e}")

        self.logger.info(f"Warmed {warmed_count} cache entries")
        return warmed_count

    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        backend_stats = self.cache_backend.stats()

        return {
            **backend_stats,
            "config": {
                "backend": self.cache_config.backend.value,
                "max_size": self.cache_config.max_size,
                "default_ttl": self.cache_config.default_ttl,
                "strategy": self.cache_config.strategy.value,
            },
            "invalidation_tags": len(self.invalidation_tags),
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check."""
        try:
            # Test basic operations
            test_key = "health_check_test"
            test_value = {"timestamp": time.time(), "test": True}

            # Test set
            set_success = self.set(test_key, test_value, ttl=60)

            # Test get
            retrieved_value = self.get(test_key)
            get_success = retrieved_value is not None

            # Test delete
            delete_success = self.delete(test_key)

            return {
                "healthy": set_success and get_success and delete_success,
                "set_success": set_success,
                "get_success": get_success,
                "delete_success": delete_success,
                "backend": self.cache_config.backend.value,
                "timestamp": time.time(),
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "backend": self.cache_config.backend.value,
                "timestamp": time.time(),
            }


def cache_result(
    key_func: Optional[Callable] = None,
    ttl: Optional[int] = None,
    cache_manager: Optional[CacheManager] = None,
):
    """Decorator to cache function results."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache manager
            cm = cache_manager or get_cache_manager()

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()

            # Try to get from cache
            cached_result = cm.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cm.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def configure_cache(config_manager: ConfigurationManager) -> None:
    """Configure global cache manager."""
    global _cache_manager
    _cache_manager = CacheManager(config_manager)
