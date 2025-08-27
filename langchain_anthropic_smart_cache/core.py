"""Core smart cache callback handler for LangChain."""

import json
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from langchain_anthropic_smart_cache.cache import CacheManager, CacheEntry, CacheStats
from langchain_anthropic_smart_cache.utils import TokenCounter, ContentAnalyzer

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
            tools = kwargs.get('tools', []) or kwargs.get('invocation_params', {}).get('tools', [])
            all_messages = []
            for message_list in messages:
                all_messages.extend(message_list)

            # Clear any existing cache control to avoid conflicts
            self._clear_existing_cache_controls(all_messages, tools)

            # Apply smart caching
            self._apply_smart_caching(all_messages, tools)

            # Final summary logging
            if self.enable_logging:
                cached_indices = []
                for i, message in enumerate(all_messages):
                    if hasattr(message, 'content') and isinstance(message.content, list):
                        for item in message.content:
                            if isinstance(item, dict) and 'cache_control' in item:
                                cached_indices.append(i)
                                break

                tool_count = sum(1 for tool in tools if isinstance(tool, dict) and 'cache_control' in tool)
                total_blocks = len(cached_indices) + tool_count

                logger.info(f"🎯 FINAL: messages{cached_indices} + {tool_count} tools = {total_blocks}/4 slots")

        except Exception as e:
            logger.error(f"Error in smart cache processing: {e}")
            if self.enable_logging:
                import traceback
                logger.debug(f"Full traceback: {traceback.format_exc()}")

    def _clear_existing_cache_controls(self, messages: List[BaseMessage], tools: List[Dict[str, Any]]) -> None:
        """Clear existing cache_control tags."""
        # Clear cache controls from message content
        for message in messages:
            if hasattr(message, 'content') and isinstance(message.content, list):
                for item in message.content:
                    if isinstance(item, dict) and 'cache_control' in item:
                        del item['cache_control']

        # Clear cache controls from tools
        for tool in tools:
            if isinstance(tool, dict) and 'cache_control' in tool:
                del tool['cache_control']

    def _apply_smart_caching(self, messages: List[BaseMessage], tools: List[Dict[str, Any]]) -> None:
        """Apply intelligent caching strategy to messages and tools."""
        # Collect cache candidates
        cache_candidates = []

        # Track cache statistics
        total_tools = len(tools) if tools else 0
        cached_tools = 0
        total_messages = 0
        cached_messages = 0

        # 1. ANALYZE TOOLS
        if tools:
            tools_analysis = self.content_analyzer.analyze_tools(tools)
            if tools_analysis['cacheable']:
                cache_entry = self.cache_manager.get({'tools': tools})
                is_cached = cache_entry is not None

                if is_cached:
                    cached_tools = 1

                # Skip if already cached and not expired - no need to cache again!
                if not is_cached:
                    cache_candidates.append(CacheCandidate(
                        content={'tools': tools},
                        token_count=tools_analysis['token_count'],
                        content_type='tools',
                        priority=1,  # Always high priority when not cached
                        is_cached=False,
                        cache_entry=None
                    ))



        # 2. ANALYZE MESSAGES
        for i, message in enumerate(messages):
            # Convert message to dict for analysis
            message_dict = self._message_to_dict(message)
            analysis = self.content_analyzer.analyze_message(message_dict)

            if analysis['cacheable'] and analysis['token_count'] >= self.min_token_count:
                total_messages += 1

                # Check if already cached
                cache_entry = self.cache_manager.get(message_dict)
                is_cached = cache_entry is not None

                if is_cached:
                    cached_messages += 1

                # Skip if already cached and not expired - no need to cache again!
                if not is_cached:
                    # Only cache NEW content that needs caching
                    if analysis['content_type'] == 'system':
                        priority = 2  # High priority for new system content
                    else:
                        priority = analysis['priority']  # Use base priority for new content

                    cache_candidates.append(CacheCandidate(
                        content=message_dict,
                        token_count=analysis['token_count'],
                        content_type=analysis['content_type'],
                        priority=priority,
                        is_cached=False,
                        cache_entry=None
                    ))


        # 3. SMART PRIORITIZATION AND SLOT ALLOCATION
        if self.enable_logging:
            new_messages = total_messages - cached_messages
            new_tools = total_tools - cached_tools
            logger.info(f"🚀 CACHE: {total_messages} messages ({cached_messages} cached, {new_messages} new), {total_tools} tools ({cached_tools} cached, {new_tools} new)")

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

        if self.enable_logging:
            # Compact table showing what gets selected/skipped
            selected = cache_candidates[:self.max_cache_blocks]
            skipped = cache_candidates[self.max_cache_blocks:]

            logger.info(f"📊 CACHE SELECTION ({len(cache_candidates)} candidates → {len(selected)} selected)")

            for i, candidate in enumerate(selected):
                status = "CACHED" if candidate.is_cached else "NEW"
                age = f"{candidate.cache_entry.age_seconds():.0f}s" if candidate.cache_entry else "0s"
                logger.info(f"  ✅ {candidate.content_type} p={candidate.priority} {status}({age}) {candidate.token_count}t")

            for candidate in skipped:
                status = "CACHED" if candidate.is_cached else "NEW"
                age = f"{candidate.cache_entry.age_seconds():.0f}s" if candidate.cache_entry else "0s"
                logger.info(f"  ❌ {candidate.content_type} p={candidate.priority} {status}({age}) {candidate.token_count}t")

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
                    message.content = [{'type': 'text', 'text': str(message.content), 'cache_control': {'type': 'ephemeral'}}]

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
        """Log compact cache operation results."""
        if not self.enable_logging:
            return

        # Just the essentials - already logged above in selection
        pass

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