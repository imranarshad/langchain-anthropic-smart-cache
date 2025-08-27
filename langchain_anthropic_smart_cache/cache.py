"""Cache management utilities for smart caching."""

import json
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached content entry."""
    key: str
    content: Any
    cached_at: datetime
    token_count: int
    content_type: str  # 'tools', 'system', 'content'

    def is_expired(self, cache_duration: int) -> bool:
        """Check if the cache entry has expired."""
        expiry_time = self.cached_at + timedelta(seconds=cache_duration)
        return datetime.now() > expiry_time

    def age_seconds(self) -> float:
        """Get the age of this cache entry in seconds."""
        return (datetime.now() - self.cached_at).total_seconds()


class CacheStats(NamedTuple):
    """Cache performance statistics."""
    total_requests: int
    cache_hits: int
    cache_misses: int
    total_tokens_cached: int
    total_tokens_skipped: int
    cache_hit_rate: float
    estimated_savings: float


class CacheManager:
    """Manages cache storage and retrieval for content blocks."""

    def __init__(self, cache_dir: Optional[str] = None, cache_duration: int = 300):
        """Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files. If None, uses temp directory.
            cache_duration: Cache validity duration in seconds.
        """
        self.cache_duration = cache_duration
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "langchain_anthropic_smart_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for faster access
        self._memory_cache: Dict[str, CacheEntry] = {}

        # Statistics
        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_tokens_cached': 0,
            'total_tokens_skipped': 0,
        }

        # Load existing cache from disk
        self._load_cache_from_disk()

    def _get_cache_key(self, content: Any) -> str:
        """Generate a cache key for content, excluding cache_control."""
        import copy

        # Deep copy to avoid modifying original
        clean_content = copy.deepcopy(content)

        # Remove cache_control from any level
        self._remove_cache_control(clean_content)

        content_str = json.dumps(clean_content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def _remove_cache_control(self, obj: Any) -> None:
        """Recursively remove cache_control from any object."""
        if isinstance(obj, dict):
            obj.pop('cache_control', None)
            for value in obj.values():
                self._remove_cache_control(value)
        elif isinstance(obj, list):
            for item in obj:
                self._remove_cache_control(item)

    def _load_cache_from_disk(self) -> None:
        """Load cache entries from disk."""
        cache_file = self.cache_dir / "cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for key, entry_data in data.get('entries', {}).items():
                    entry = CacheEntry(
                        key=entry_data['key'],
                        content=entry_data['content'],
                        cached_at=datetime.fromisoformat(entry_data['cached_at']),
                        token_count=entry_data['token_count'],
                        content_type=entry_data['content_type']
                    )

                    # Only load non-expired entries
                    if not entry.is_expired(self.cache_duration):
                        self._memory_cache[key] = entry

                # Load statistics
                self._stats.update(data.get('stats', {}))
                logger.debug(f"Loaded {len(self._memory_cache)} cache entries from disk")

            except Exception as e:
                logger.warning(f"Failed to load cache from disk: {e}")

    def _save_cache_to_disk(self) -> None:
        """Save cache entries to disk."""
        cache_file = self.cache_dir / "cache.json"
        try:
            # Only save non-expired entries
            valid_entries = {
                key: {
                    'key': entry.key,
                    'content': entry.content,
                    'cached_at': entry.cached_at.isoformat(),
                    'token_count': entry.token_count,
                    'content_type': entry.content_type
                }
                for key, entry in self._memory_cache.items()
                if not entry.is_expired(self.cache_duration)
            }

            data = {
                'entries': valid_entries,
                'stats': self._stats,
                'last_updated': datetime.now().isoformat()
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")

    def get(self, content: Any) -> Optional[CacheEntry]:
        """Get a cache entry for content if it exists and is valid."""
        self._stats['total_requests'] += 1

        cache_key = self._get_cache_key(content)
        entry = self._memory_cache.get(cache_key)

        if entry and not entry.is_expired(self.cache_duration):
            self._stats['cache_hits'] += 1
            logger.debug(f"Cache HIT for {entry.content_type} (age: {entry.age_seconds():.1f}s)")
            return entry

        if entry:
            # Remove expired entry
            del self._memory_cache[cache_key]
            logger.debug(f"Cache entry expired for {entry.content_type}")

        self._stats['cache_misses'] += 1
        logger.debug(f"Cache MISS for content")
        return None

    def put(self, content: Any, token_count: int, content_type: str) -> str:
        """Store content in cache and return the cache key."""
        cache_key = self._get_cache_key(content)

        entry = CacheEntry(
            key=cache_key,
            content=content,
            cached_at=datetime.now(),
            token_count=token_count,
            content_type=content_type
        )

        self._memory_cache[cache_key] = entry
        self._stats['total_tokens_cached'] += token_count

        logger.debug(f"Cached {content_type} with {token_count} tokens (key: {cache_key})")

        # Periodically save to disk
        if len(self._memory_cache) % 10 == 0:
            self._save_cache_to_disk()

        return cache_key

    def is_cached(self, content: Any) -> bool:
        """Check if content is cached and valid."""
        return self.get(content) is not None

    def cleanup_expired(self) -> int:
        """Remove expired cache entries and return count of removed entries."""
        expired_keys = [
            key for key, entry in self._memory_cache.items()
            if entry.is_expired(self.cache_duration)
        ]

        for key in expired_keys:
            del self._memory_cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            self._save_cache_to_disk()

        return len(expired_keys)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()
        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_tokens_cached': 0,
            'total_tokens_skipped': 0,
        }

        cache_file = self.cache_dir / "cache.json"
        if cache_file.exists():
            cache_file.unlink()

        logger.info("Cache cleared")

    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        total_requests = self._stats['total_requests']
        cache_hits = self._stats['cache_hits']

        hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0

        # Rough cost estimation (assuming $3 per 1M input tokens for Claude)
        estimated_savings = (self._stats['total_tokens_cached'] / 1_000_000) * 3.0 * 0.9  # 90% savings

        return CacheStats(
            total_requests=total_requests,
            cache_hits=cache_hits,
            cache_misses=self._stats['cache_misses'],
            total_tokens_cached=self._stats['total_tokens_cached'],
            total_tokens_skipped=self._stats['total_tokens_skipped'],
            cache_hit_rate=hit_rate,
            estimated_savings=estimated_savings
        )

    def add_skipped_tokens(self, token_count: int) -> None:
        """Add to the count of skipped tokens for statistics."""
        self._stats['total_tokens_skipped'] += token_count

    def __del__(self):
        """Save cache to disk on destruction."""
        try:
            self._save_cache_to_disk()
        except Exception:
            pass  # Ignore errors during cleanup