#!/usr/bin/env python3
"""Simple integration test to verify the package works correctly."""

import sys
import os
import tempfile
from pathlib import Path

# Add the package to the path for testing
sys.path.insert(0, str(Path(__file__).parent))

from langchain_anthropic_smart_cache import SmartCacheCallbackHandler, CacheManager, TokenCounter, ContentAnalyzer
from langchain_core.messages import HumanMessage, SystemMessage


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("🧪 Testing basic functionality...")

    # Test cache manager
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_manager = CacheManager(cache_dir=temp_dir)

        # Test caching
        test_content = {"test": "content"}
        cache_manager.put(test_content, 100, "test")

        entry = cache_manager.get(test_content)
        assert entry is not None
        assert entry.content == test_content
        print("✅ Cache manager works")

    # Test token counter
    token_counter = TokenCounter()
    count = token_counter.count_tokens("Hello world!")
    assert count > 0
    print("✅ Token counter works")

    # Test content analyzer
    analyzer = ContentAnalyzer()
    analysis = analyzer.analyze_message({
        "role": "user",
        "content": "Hello world!"
    })
    assert "content_type" in analysis
    assert "token_count" in analysis
    print("✅ Content analyzer works")

    # Test callback handler initialization
    handler = SmartCacheCallbackHandler(enable_logging=False)
    assert handler.cache_manager is not None
    assert handler.token_counter is not None
    print("✅ Callback handler initializes")

    # Test message conversion
    message = HumanMessage(content="Test message")
    message_dict = handler._message_to_dict(message)
    assert message_dict["role"] == "human"
    assert message_dict["content"] == "Test message"
    print("✅ Message conversion works")

    print("🎉 All basic tests passed!")


def test_cache_prioritization():
    """Test the cache prioritization logic."""
    print("\n🧪 Testing cache prioritization...")

    handler = SmartCacheCallbackHandler(
        enable_logging=False,
        max_cache_blocks=2,  # Limited slots for testing
        min_token_count=10   # Low threshold for testing
    )

    # Create test messages
    system_msg = SystemMessage(content="You are a helpful assistant with detailed instructions for providing accurate and helpful responses.")
    user_msg = HumanMessage(content="Hello!")

    # Test tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information for a specific location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The location to get weather for"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    # Mock serialized data
    serialized = {"kwargs": {"tools": tools}}

    # Test the main caching logic
    try:
        handler.on_chat_model_start(
            serialized=serialized,
            messages=[[system_msg, user_msg]]
        )
        print("✅ Cache prioritization logic runs without errors")
    except Exception as e:
        print(f"❌ Error in cache prioritization: {e}")
        raise

    # Get stats
    stats = handler.get_stats()
    print(f"✅ Cache stats accessible: {stats.total_requests} requests")

    print("🎉 Cache prioritization tests passed!")


def main():
    """Run all integration tests."""
    print("🚀 Running LangChain Anthropic Smart Cache Integration Tests\n")

    try:
        test_basic_functionality()
        test_cache_prioritization()

        print("\n🎉 ALL TESTS PASSED! Package is ready for use.")
        return 0

    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())