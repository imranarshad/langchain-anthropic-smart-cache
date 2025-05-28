"""Tests for LangChain Anthropic Smart Cache."""

import pytest
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_anthropic_smart_cache import SmartCacheCallbackHandler, CacheManager, TokenCounter, ContentAnalyzer


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