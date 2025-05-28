"""Core smart cache callback handler for LangChain."""

import json
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from .cache import CacheManager, CacheEntry, CacheStats
from .utils import TokenCounter, ContentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class CacheCandidate:
    """Represents a content block that could be cached."""
    content: Any
    token_count: int
    content_type: str  # 'tools', 'system', 'content'
    priority: int      # Lower = higher priority
    is_cached: bool
    cache_entry: Optional[CacheEntry] = None

    def __str__(self) -> str:
        status = "cached" if self.is_cached else "new"
        return f"{self.content_type} (priority {self.priority}, {status}, {self.token_count} tokens)"


class SmartCacheCallbackHandler(BaseCallbackHandler):
    """Intelligent cache management callback handler for Anthropic models.

    This handler automatically optimizes cache usage by:
    1. Prioritizing tools and system prompts when not cached
    2. Managing cache slots efficiently (max 4 for Anthropic)
    3. Providing detailed analytics and logging
    4. Automatically refreshing expiring cache entries
    """

    def __init__(
        self,
        cache_duration: int = 300,
        max_cache_blocks: int = 4,
        min_token_count: int = 1024,
        enable_logging: bool = True,
        log_level: str = "INFO",
        cache_dir: Optional[str] = None,
    ):
        """Initialize the smart cache callback handler.

        Args:
            cache_duration: Cache validity duration in seconds (default: 5 minutes).
            max_cache_blocks: Maximum number of cache blocks (Anthropic limit: 4).
            min_token_count: Minimum tokens required to consider for caching.
            enable_logging: Whether to enable detailed cache logging.
            log_level: Logging level for cache operations.
            cache_dir: Directory to store cache files (default: temp directory).
        """
        super().__init__()

        self.cache_duration = cache_duration
        self.max_cache_blocks = max_cache_blocks
        self.min_token_count = min_token_count
        self.enable_logging = enable_logging

        # Initialize components
        self.cache_manager = CacheManager(cache_dir=cache_dir, cache_duration=cache_duration)
        self.token_counter = TokenCounter()
        self.content_analyzer = ContentAnalyzer(self.token_counter)

        # Configure logging
        if enable_logging:
            logging.getLogger(__name__).setLevel(getattr(logging, log_level.upper()))

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """Handle the start of a chat model invocation.

        This is where the smart caching logic is applied.
        """
        try:
            # Extract tools and flatten messages
            tools = serialized.get('kwargs', {}).get('tools', [])
            all_messages = []
            for message_list in messages:
                all_messages.extend(message_list)

            if self.enable_logging:
                logger.info(f"🚀 Smart cache processing: {len(all_messages)} messages, {len(tools)} tools")

            # Clear any existing cache control to avoid conflicts
            self._clear_existing_cache_controls(all_messages, tools)

            # Apply smart caching
            self._apply_smart_caching(all_messages, tools)

        except Exception as e:
            logger.error(f"Error in smart cache processing: {e}")
            if self.enable_logging:
                import traceback
                logger.debug(f"Full traceback: {traceback.format_exc()}")

    def _clear_existing_cache_controls(self, messages: List[BaseMessage], tools: List[Dict[str, Any]]) -> None:
        """Clear any existing cache_control tags to prevent conflicts."""
        # Clear cache controls from messages
        for message in messages:
            if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
                message.additional_kwargs.pop('cache_control', None)

            # Handle multimodal content
            if hasattr(message, 'content') and isinstance(message.content, list):
                for item in message.content:
                    if isinstance(item, dict) and 'cache_control' in item:
                        del item['cache_control']

        # Clear cache controls from tools
        for tool in tools:
            if isinstance(tool, dict) and 'cache_control' in tool:
                del tool['cache_control']

        if self.enable_logging:
            logger.debug("🧹 Cleared existing cache controls")

    def _apply_smart_caching(self, messages: List[BaseMessage], tools: List[Dict[str, Any]]) -> None:
        """Apply intelligent caching strategy to messages and tools."""
        # Collect cache candidates
        cache_candidates = []

        # 1. ANALYZE TOOLS
        if tools:
            tools_analysis = self.content_analyzer.analyze_tools(tools)
            if tools_analysis['cacheable']:
                cache_entry = self.cache_manager.get({'tools': tools})
                is_cached = cache_entry is not None

                # Tools get high priority when not cached, lower when cached
                priority = 4 if is_cached else 1

                cache_candidates.append(CacheCandidate(
                    content={'tools': tools},
                    token_count=tools_analysis['token_count'],
                    content_type='tools',
                    priority=priority,
                    is_cached=is_cached,
                    cache_entry=cache_entry
                ))

                if self.enable_logging:
                    status = f"cached (age: {cache_entry.age_seconds():.1f}s)" if is_cached else "not cached"
                    logger.info(f"🔧 Found tools: {tools_analysis['tool_count']} tools, {tools_analysis['token_count']} tokens, {status}")

        # 2. ANALYZE MESSAGES
        for i, message in enumerate(messages):
            # Convert message to dict for analysis
            message_dict = self._message_to_dict(message)
            analysis = self.content_analyzer.analyze_message(message_dict)

            if analysis['cacheable'] and analysis['token_count'] >= self.min_token_count:
                # Check if already cached
                cache_entry = self.cache_manager.get(message_dict)
                is_cached = cache_entry is not None

                # Adjust priority based on cache status
                if analysis['content_type'] == 'system':
                    priority = 5 if is_cached else 2  # High priority when not cached
                else:
                    priority = analysis['priority']

                cache_candidates.append(CacheCandidate(
                    content=message_dict,
                    token_count=analysis['token_count'],
                    content_type=analysis['content_type'],
                    priority=priority,
                    is_cached=is_cached,
                    cache_entry=cache_entry
                ))

                if self.enable_logging:
                    status = f"cached (age: {cache_entry.age_seconds():.1f}s)" if is_cached else "not cached"
                    logger.info(f"📝 Found {analysis['content_type']} message: {analysis['token_count']} tokens, {status}")

        if self.enable_logging:
            logger.info(f"🎯 Found {len(cache_candidates)} cacheable items")

        # 3. SMART PRIORITIZATION AND SLOT ALLOCATION
        self._allocate_cache_slots(cache_candidates, messages, tools)

    def _allocate_cache_slots(
        self,
        cache_candidates: List[CacheCandidate],
        messages: List[BaseMessage],
        tools: List[Dict[str, Any]]
    ) -> None:
        """Allocate cache slots using intelligent prioritization."""
        if not cache_candidates:
            if self.enable_logging:
                logger.info("🚫 No cacheable content found")
            return

        # Sort by priority (lower = higher priority), then by token count (descending)
        cache_candidates.sort(key=lambda x: (x.priority, -x.token_count))

        cached_items = []
        skipped_items = []
        used_slots = 0

        for candidate in cache_candidates:
            if used_slots >= self.max_cache_blocks:
                skipped_items.append(candidate)
                self.cache_manager.add_skipped_tokens(candidate.token_count)
                continue

            # Apply caching
            if candidate.content_type == 'tools':
                self._apply_tools_caching(tools, candidate)
            else:
                self._apply_message_caching(messages, candidate)

            # Update cache manager
            if not candidate.is_cached:
                self.cache_manager.put(
                    candidate.content,
                    candidate.token_count,
                    candidate.content_type
                )

            cached_items.append(candidate)
            used_slots += 1

        # Log results
        self._log_cache_results(cached_items, skipped_items, used_slots)

    def _apply_tools_caching(self, tools: List[Dict[str, Any]], candidate: CacheCandidate) -> None:
        """Apply cache control to tools."""
        if tools:
            # Add cache control to the last tool
            tools[-1]['cache_control'] = {'type': 'ephemeral'}

            if self.enable_logging:
                action = "MAINTAIN" if candidate.is_cached else "NEW"
                logger.info(f"💾 CACHED tools (slot {len([c for c in [candidate] if c.content_type == 'tools'])}/4) - {action} tools {'refresh' if candidate.is_cached else 'needed caching'}")

    def _apply_message_caching(self, messages: List[BaseMessage], candidate: CacheCandidate) -> None:
        """Apply cache control to a message."""
        # Find the message that matches this candidate
        message_dict = candidate.content

        for message in messages:
            if self._message_matches_candidate(message, message_dict):
                # Add cache control
                if hasattr(message, 'content') and isinstance(message.content, list):
                    # Multimodal content - add to last content block
                    if message.content:
                        if isinstance(message.content[-1], dict):
                            message.content[-1]['cache_control'] = {'type': 'ephemeral'}
                        else:
                            # Convert to dict format if needed
                            last_item = message.content[-1]
                            message.content[-1] = {
                                'type': 'text',
                                'text': str(last_item),
                                'cache_control': {'type': 'ephemeral'}
                            }
                else:
                    # Simple string content - use additional_kwargs
                    if not hasattr(message, 'additional_kwargs'):
                        message.additional_kwargs = {}
                    message.additional_kwargs['cache_control'] = {'type': 'ephemeral'}

                if self.enable_logging:
                    action = "MAINTAIN" if candidate.is_cached else "NEW"
                    if candidate.is_cached and candidate.cache_entry:
                        age = candidate.cache_entry.age_seconds()
                        if age > self.cache_duration * 0.8:  # Near expiry
                            action = "REFRESH"

                    logger.info(f"💾 CACHED {candidate.content_type} (slot X/4, {candidate.token_count} tokens) - {action} {'existing cache' if candidate.is_cached else candidate.content_type + ' block'}")
                break

    def _message_matches_candidate(self, message: BaseMessage, message_dict: Dict[str, Any]) -> bool:
        """Check if a message matches a cache candidate."""
        current_dict = self._message_to_dict(message)
        return current_dict.get('content') == message_dict.get('content') and \
               current_dict.get('role') == message_dict.get('role')

    def _message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        """Convert a LangChain message to a dictionary."""
        message_dict = {
            'role': message.type,
            'content': message.content
        }

        # Handle additional attributes
        if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
            message_dict.update(message.additional_kwargs)

        return message_dict

    def _log_cache_results(
        self,
        cached_items: List[CacheCandidate],
        skipped_items: List[CacheCandidate],
        used_slots: int
    ) -> None:
        """Log detailed cache operation results."""
        if not self.enable_logging:
            return

        # Log skipped items
        if skipped_items:
            logger.info(f"🚫 SKIPPED ITEMS ({len(skipped_items)} items):")
            for item in skipped_items:
                reason = "no slots available" if used_slots >= self.max_cache_blocks else "lower priority"
                logger.info(f"  ❌ {item} - {reason}")

        # Calculate statistics
        previously_cached = sum(1 for item in cached_items if item.is_cached)
        newly_cached = len(cached_items) - previously_cached

        cached_tokens = sum(item.token_count for item in cached_items)
        skipped_tokens = sum(item.token_count for item in skipped_items)

        cache_rate = (previously_cached / len(cached_items) * 100) if cached_items else 0

        # Summary log
        logger.info("📊 CACHE SUMMARY:")
        logger.info(f"  🎯 Slots used: {used_slots}/{self.max_cache_blocks}")
        logger.info(f"  ⚡ Previously cached: {previously_cached} items ({cache_rate:.1f}%)")
        logger.info(f"  💾 Newly cached: {newly_cached} items")
        logger.info(f"  🚫 Skipped: {len(skipped_items)} items")
        logger.info(f"  📈 Cached tokens: {cached_tokens:,} | Skipped tokens: {skipped_tokens:,}")

    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        return self.cache_manager.get_stats()

    def clear_cache(self) -> None:
        """Clear all cached content."""
        self.cache_manager.clear()
        if self.enable_logging:
            logger.info("🧹 Cache cleared manually")

    def cleanup_expired(self) -> int:
        """Clean up expired cache entries and return count of removed entries."""
        removed_count = self.cache_manager.cleanup_expired()
        if self.enable_logging and removed_count > 0:
            logger.info(f"🧹 Cleaned up {removed_count} expired cache entries")
        return removed_count