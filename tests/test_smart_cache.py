"""Tests for LangChain Anthropic Smart Cache."""

import pytest
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_anthropic_smart_cache import SmartCacheCallbackHandler, CacheManager, TokenCounter, ContentAnalyzer
from langchain_anthropic_smart_cache.core import CacheCandidate


class TestCacheManager:
    """Test cache manager functionality."""

    def test_cache_put_and_get(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir, cache_duration=300)

            # Test putting and getting content
            content = {"test": "content"}
            token_count = 100
            content_type = "test"

            key = cache_manager.put(content, token_count, content_type)
            assert key is not None

            entry = cache_manager.get(content)
            assert entry is not None
            assert entry.content == content
            assert entry.token_count == token_count
            assert entry.content_type == content_type

    def test_cache_expiration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir, cache_duration=1)  # 1 second

            # Put content
            content = {"test": "expiring_content"}
            cache_manager.put(content, 100, "test")

            # Should be available immediately
            assert cache_manager.get(content) is not None

            # Mock time to simulate expiration
            with patch('langchain_anthropic_smart_cache.cache.datetime') as mock_datetime:
                # Simulate 2 seconds later
                future_time = datetime.now() + timedelta(seconds=2)
                mock_datetime.now.return_value = future_time

                # Should be expired
                assert cache_manager.get(content) is None

    def test_cache_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)

            # Initial stats
            stats = cache_manager.get_stats()
            assert stats.total_requests == 0
            assert stats.cache_hits == 0
            assert stats.cache_misses == 0

            # Put and get content
            content = {"test": "stats"}
            cache_manager.put(content, 100, "test")

            # Get hit
            entry = cache_manager.get(content)
            assert entry is not None

            # Get miss
            missing_content = {"missing": "content"}
            entry = cache_manager.get(missing_content)
            assert entry is None

            # Check stats
            stats = cache_manager.get_stats()
            assert stats.total_requests == 2
            assert stats.cache_hits == 1
            assert stats.cache_misses == 1


class TestTokenCounter:
    """Test token counting functionality."""

    def test_count_tokens_string(self):
        counter = TokenCounter()

        # Test string content
        text = "Hello world, this is a test message."
        count = counter.count_tokens(text)
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_dict(self):
        counter = TokenCounter()

        # Test dict content
        content = {"message": "Hello world", "role": "user"}
        count = counter.count_tokens(content)
        assert count > 0

    def test_count_message_tokens(self):
        counter = TokenCounter()

        # Test message
        message = {
            "role": "user",
            "content": "What's the weather like today?"
        }
        count = counter.count_message_tokens(message)
        assert count > 0

        # Should include overhead
        content_only_count = counter.count_tokens(message["content"])
        assert count > content_only_count

    def test_count_tools_tokens(self):
        counter = TokenCounter()

        # Test tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ]

        count = counter.count_tools_tokens(tools)
        assert count > 0


class TestContentAnalyzer:
    """Test content analysis functionality."""

    def test_analyze_message_system(self):
        analyzer = ContentAnalyzer()

        message = {
            "role": "system",
            "content": "You are a helpful assistant that provides accurate information."
        }

        analysis = analyzer.analyze_message(message)
        assert analysis["content_type"] == "system"
        assert analysis["role"] == "system"
        assert analysis["token_count"] > 0
        assert isinstance(analysis["cacheable"], bool)

    def test_analyze_message_user(self):
        analyzer = ContentAnalyzer()

        message = {
            "role": "user",
            "content": "What's the weather like in San Francisco?"
        }

        analysis = analyzer.analyze_message(message)
        assert analysis["content_type"] == "content"
        assert analysis["role"] == "user"
        assert analysis["priority"] >= 1

    def test_analyze_tools(self):
        analyzer = ContentAnalyzer()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "Location name"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        analysis = analyzer.analyze_tools(tools)
        assert analysis["content_type"] == "tools"
        assert analysis["priority"] == 1  # High priority
        assert analysis["tool_count"] == 1
        assert analysis["token_count"] > 0


class TestCacheCandidate:
    """Test CacheCandidate dataclass functionality."""

    def test_cache_candidate_creation(self):
        candidate = CacheCandidate(
            content={"test": "content"},
            token_count=100,
            content_type="system",
            priority=2,
            is_cached=False
        )

        assert candidate.content == {"test": "content"}
        assert candidate.token_count == 100
        assert candidate.content_type == "system"
        assert candidate.priority == 2
        assert not candidate.is_cached
        assert candidate.cache_entry is None

    def test_cache_candidate_str_representation(self):
        # Test uncached candidate
        candidate = CacheCandidate(
            content={"test": "content"},
            token_count=100,
            content_type="system",
            priority=2,
            is_cached=False
        )

        expected = "system (priority 2, new, 100 tokens)"
        assert str(candidate) == expected

        # Test cached candidate
        candidate.is_cached = True
        expected = "system (priority 2, cached, 100 tokens)"
        assert str(candidate) == expected


class TestSmartCacheCallbackHandler:
    """Test the main callback handler."""

    def test_handler_initialization(self):
        handler = SmartCacheCallbackHandler(
            cache_duration=300,
            max_cache_blocks=4,
            min_token_count=1024,
            enable_logging=False
        )

        assert handler.cache_duration == 300
        assert handler.max_cache_blocks == 4
        assert handler.min_token_count == 1024
        assert not handler.enable_logging
        assert handler.cache_manager is not None
        assert handler.token_counter is not None
        assert handler.content_analyzer is not None

    def test_message_to_dict_conversion(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        # Test HumanMessage
        human_msg = HumanMessage(content="Hello!")
        human_dict = handler._message_to_dict(human_msg)
        assert human_dict["role"] == "human"
        assert human_dict["content"] == "Hello!"

        # Test SystemMessage
        system_msg = SystemMessage(content="You are helpful.")
        system_dict = handler._message_to_dict(system_msg)
        assert system_dict["role"] == "system"
        assert system_dict["content"] == "You are helpful."

    def test_clear_existing_cache_controls(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        # Create message with cache control
        message = HumanMessage(
            content="Test message",
            additional_kwargs={"cache_control": {"type": "ephemeral"}}
        )

        # Create tools with cache control
        tools = [{"name": "test", "cache_control": {"type": "ephemeral"}}]

        # Clear cache controls
        handler._clear_existing_cache_controls([message], tools)

        # Verify cache controls are removed
        assert "cache_control" not in message.additional_kwargs
        assert "cache_control" not in tools[0]

    def test_get_stats(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        stats = handler.get_stats()
        assert hasattr(stats, 'total_requests')
        assert hasattr(stats, 'cache_hits')
        assert hasattr(stats, 'cache_misses')
        assert hasattr(stats, 'cache_hit_rate')

    def test_clear_cache(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        # This should not raise any exceptions
        handler.clear_cache()

    def test_cleanup_expired(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        # This should return a count (even if 0)
        count = handler.cleanup_expired()
        assert isinstance(count, int)
        assert count >= 0


@pytest.fixture
def sample_messages():
    """Fixture providing sample messages for testing."""
    return [
        SystemMessage(content="You are a helpful assistant with access to weather tools."),
        HumanMessage(content="What's the weather like in San Francisco today?"),
    ]


@pytest.fixture
def sample_tools():
    """Fixture providing sample tools for testing."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]


def test_integration_with_sample_data(sample_messages, sample_tools):
    """Integration test with sample data."""
    handler = SmartCacheCallbackHandler(
        cache_duration=300,
        min_token_count=50,  # Lower threshold for testing
        enable_logging=False
    )

    # Mock serialized data
    serialized = {
        "kwargs": {
            "tools": sample_tools
        }
    }

    # This should not raise any exceptions
    handler.on_chat_model_start(
        serialized=serialized,
        messages=[sample_messages]
    )


class TestSmartCacheCallbackHandlerCore:
    """Test core functionality of SmartCacheCallbackHandler."""

    def test_handler_initialization_with_custom_params(self):
        handler = SmartCacheCallbackHandler(
            cache_duration=600,
            max_cache_blocks=3,
            min_token_count=500,
            enable_logging=True,
            log_level="DEBUG",
            cache_dir="/tmp/test_cache"
        )

        assert handler.cache_duration == 600
        assert handler.max_cache_blocks == 3
        assert handler.min_token_count == 500
        assert handler.enable_logging
        assert handler.cache_manager is not None
        assert handler.token_counter is not None
        assert handler.content_analyzer is not None

    def test_on_chat_model_start_basic(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        messages = [
            [SystemMessage(content="You are a helpful assistant.")]
        ]

        serialized = {
            "kwargs": {
                "tools": []
            }
        }

        # Should not raise any exceptions
        handler.on_chat_model_start(serialized=serialized, messages=messages)

    def test_on_chat_model_start_with_tools(self):
        handler = SmartCacheCallbackHandler(enable_logging=False, min_token_count=10)

        messages = [
            [SystemMessage(content="You are a helpful assistant with access to tools.")]
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather information for any location. This tool provides detailed weather data including temperature, humidity, precipitation, wind speed, and other meteorological information that users might need for planning their activities.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state or country, e.g. San Francisco, CA or London, UK"
                            },
                            "units": {
                                "type": "string",
                                "description": "Temperature units to use",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        serialized = {
            "kwargs": {
                "tools": tools
            }
        }

        handler.on_chat_model_start(serialized=serialized, messages=messages)

        # Check that cache control was added to tools
        assert "cache_control" in tools[-1]
        assert tools[-1]["cache_control"]["type"] == "ephemeral"

    def test_on_chat_model_start_with_error_handling(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        # Test with malformed data that might cause errors
        with patch.object(handler, '_apply_smart_caching', side_effect=Exception("Test error")):
            # Should not raise exception due to error handling
            handler.on_chat_model_start(serialized={}, messages=[[]])

    def test_clear_existing_cache_controls(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        # Create message with existing cache control
        message = HumanMessage(content="Test message")
        message.additional_kwargs = {"cache_control": {"type": "ephemeral"}}

        # Create tools with existing cache control
        tools = [
            {"name": "test_tool", "cache_control": {"type": "ephemeral"}},
            {"name": "tool2", "description": "Another tool"}
        ]

        handler._clear_existing_cache_controls([message], tools)

        # Verify cache controls are removed
        assert "cache_control" not in message.additional_kwargs
        assert "cache_control" not in tools[0]
        assert "cache_control" not in tools[1]

    def test_clear_existing_cache_controls_multimodal(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        # Create message with multimodal content containing cache control
        message = HumanMessage(content=[
            {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "World"}
        ])

        handler._clear_existing_cache_controls([message], [])

        # Verify cache control is removed from multimodal content
        assert "cache_control" not in message.content[0]
        assert "cache_control" not in message.content[1]

    def test_apply_smart_caching_with_large_system_message(self):
        handler = SmartCacheCallbackHandler(enable_logging=False, min_token_count=10)

        # Create a large system message that should be cached
        large_content = "You are a helpful assistant. " * 100  # Make it large enough to cache
        system_message = SystemMessage(content=large_content)

        messages = [system_message]
        tools = []

        handler._apply_smart_caching(messages, tools)

        # Check if message was modified for caching
        assert isinstance(system_message.content, list)
        assert system_message.content[-1]["cache_control"]["type"] == "ephemeral"

    def test_apply_smart_caching_no_cacheable_content(self):
        handler = SmartCacheCallbackHandler(enable_logging=False, min_token_count=10000)  # Very high threshold

        messages = [HumanMessage(content="Short message")]
        tools = []

        # Should complete without errors even with no cacheable content
        handler._apply_smart_caching(messages, tools)

    def test_apply_smart_caching_with_priority_sorting(self):
        handler = SmartCacheCallbackHandler(enable_logging=False, min_token_count=10, max_cache_blocks=2)

        # Create multiple cacheable items to test prioritization
        system_message = SystemMessage(content="System prompt. " * 50)
        user_message = HumanMessage(content="User question. " * 50)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool1",
                    "description": "First tool description. " * 20,
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

        messages = [system_message, user_message]
        handler._apply_smart_caching(messages, tools)

        # Tools and system messages should get priority
        assert "cache_control" in tools[-1]

    def test_allocate_cache_slots_no_candidates(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        # Should handle empty candidate list gracefully
        handler._allocate_cache_slots([], [], [])

    def test_allocate_cache_slots_exceeds_max_blocks(self):
        handler = SmartCacheCallbackHandler(enable_logging=False, max_cache_blocks=1)

        # Create more candidates than available slots
        candidates = [
            CacheCandidate(
                content={"test1": "content1"},
                token_count=100,
                content_type="system",
                priority=1,
                is_cached=False
            ),
            CacheCandidate(
                content={"test2": "content2"},
                token_count=200,
                content_type="content",
                priority=2,
                is_cached=False
            )
        ]

        messages = []
        tools = []

        handler._allocate_cache_slots(candidates, messages, tools)

        # Should handle slot limitation gracefully

    def test_apply_tools_caching(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        tools = [
            {"name": "tool1", "description": "First tool"},
            {"name": "tool2", "description": "Second tool"}
        ]

        candidate = CacheCandidate(
            content={"tools": tools},
            token_count=100,
            content_type="tools",
            priority=1,
            is_cached=False
        )

        handler._apply_tools_caching(tools, candidate)

        # Should add cache control to last tool
        assert "cache_control" in tools[-1]
        assert tools[-1]["cache_control"]["type"] == "ephemeral"

    def test_apply_tools_caching_already_cached(self):
        handler = SmartCacheCallbackHandler(enable_logging=True)

        tools = [{"name": "tool1", "description": "Test tool"}]

        candidate = CacheCandidate(
            content={"tools": tools},
            token_count=100,
            content_type="tools",
            priority=1,
            is_cached=True
        )

        handler._apply_tools_caching(tools, candidate)

        # Should still add cache control
        assert "cache_control" in tools[-1]

    def test_apply_message_caching_text_content(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        message = SystemMessage(content="System message content")
        messages = [message]

        candidate = CacheCandidate(
            content={"role": "system", "content": "System message content"},
            token_count=100,
            content_type="system",
            priority=1,
            is_cached=False
        )

        handler._apply_message_caching(messages, candidate)

        # Should convert content to list format with cache control
        assert isinstance(message.content, list)
        assert message.content[0]["type"] == "text"
        assert message.content[0]["text"] == "System message content"
        assert message.content[0]["cache_control"]["type"] == "ephemeral"

    def test_apply_message_caching_multimodal_content(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        message = HumanMessage(content=[
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"}
        ])
        messages = [message]

        candidate = CacheCandidate(
            content={"role": "human", "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]},
            token_count=100,
            content_type="content",
            priority=1,
            is_cached=False
        )

        handler._apply_message_caching(messages, candidate)

        # Should add cache control to last content block
        assert "cache_control" in message.content[-1]
        assert message.content[-1]["cache_control"]["type"] == "ephemeral"

    def test_apply_message_caching_multimodal_non_dict(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        message = HumanMessage(content=[
            {"type": "text", "text": "Hello"},
            "Plain text content"
        ])
        messages = [message]

        candidate = CacheCandidate(
            content={"role": "human", "content": [{"type": "text", "text": "Hello"}, "Plain text content"]},
            token_count=100,
            content_type="content",
            priority=1,
            is_cached=False
        )

        handler._apply_message_caching(messages, candidate)

        # Should convert plain text to dict format with cache control
        assert isinstance(message.content[-1], dict)
        assert message.content[-1]["type"] == "text"
        assert message.content[-1]["text"] == "Plain text content"
        assert message.content[-1]["cache_control"]["type"] == "ephemeral"

    def test_message_matches_candidate(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        message = SystemMessage(content="Test content")
        message_dict = {"role": "system", "content": "Test content"}

        assert handler._message_matches_candidate(message, message_dict)

        # Test non-matching
        different_dict = {"role": "system", "content": "Different content"}
        assert not handler._message_matches_candidate(message, different_dict)

    def test_message_to_dict_with_additional_kwargs(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        message = HumanMessage(content="Test message")
        message.additional_kwargs = {"extra_param": "value"}

        result = handler._message_to_dict(message)

        assert result["role"] == "human"
        assert result["content"] == "Test message"
        assert result["extra_param"] == "value"

    def test_log_cache_results_with_skipped_items(self):
        handler = SmartCacheCallbackHandler(enable_logging=True, max_cache_blocks=2)

        cached_items = [
            CacheCandidate(
                content={"test": "cached1"},
                token_count=100,
                content_type="system",
                priority=1,
                is_cached=True
            )
        ]

        skipped_items = [
            CacheCandidate(
                content={"test": "skipped1"},
                token_count=200,
                content_type="content",
                priority=3,
                is_cached=False
            )
        ]

        # Should not raise exceptions when logging
        handler._log_cache_results(cached_items, skipped_items, 2)

    def test_log_cache_results_disabled_logging(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        cached_items = [
            CacheCandidate(
                content={"test": "cached1"},
                token_count=100,
                content_type="system",
                priority=1,
                is_cached=False
            )
        ]

        # Should complete quickly without logging
        handler._log_cache_results(cached_items, [], 1)

    def test_log_cache_results_empty_lists(self):
        handler = SmartCacheCallbackHandler(enable_logging=True)

        # Should handle empty lists gracefully
        handler._log_cache_results([], [], 0)

    @patch('langchain_anthropic_smart_cache.core.logger')
    def test_cleanup_expired_with_logging(self, mock_logger):
        handler = SmartCacheCallbackHandler(enable_logging=True)

        with patch.object(handler.cache_manager, 'cleanup_expired', return_value=3):
            result = handler.cleanup_expired()

            assert result == 3
            mock_logger.info.assert_called_with("🧹 Cleaned up 3 expired cache entries")

    def test_cleanup_expired_no_items(self):
        handler = SmartCacheCallbackHandler(enable_logging=False)

        with patch.object(handler.cache_manager, 'cleanup_expired', return_value=0):
            result = handler.cleanup_expired()
            assert result == 0

    @patch('langchain_anthropic_smart_cache.core.logger')
    def test_clear_cache_with_logging(self, mock_logger):
        handler = SmartCacheCallbackHandler(enable_logging=True)

        handler.clear_cache()
        mock_logger.info.assert_called_with("🧹 Cache cleared manually")