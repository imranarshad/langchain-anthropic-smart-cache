"""Basic usage example for LangChain Anthropic Smart Cache."""

import logging
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_anthropic_smart_cache import SmartCacheCallbackHandler

# Configure logging to see cache operations
logging.basicConfig(level=logging.INFO)

# Initialize the smart cache handler
cache_handler = SmartCacheCallbackHandler(
    cache_duration=300,      # 5 minutes
    max_cache_blocks=4,      # Anthropic's limit
    min_token_count=1024,    # Minimum tokens to cache
    enable_logging=True,     # Enable detailed logging
    log_level="INFO"
)

# Create a tool that will be cached
@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

@tool
def calculate_distance(origin: str, destination: str) -> str:
    """Calculate distance between two locations."""
    return f"Distance from {origin} to {destination}: 150 miles"

# Initialize Anthropic model with cache handler
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    callbacks=[cache_handler]
)

# Bind tools to the model
llm_with_tools = llm.bind_tools([get_weather, calculate_distance])

# Example conversation
def run_example():
    print("🚀 Running LangChain Anthropic Smart Cache example...")

    # First call - tools and system will be cached
    print("\n📞 First call:")
    response1 = llm_with_tools.invoke([
        {"role": "system", "content": "You are a helpful travel assistant. Always provide accurate information about weather and distances. Use the available tools to get real-time data when users ask about weather or distances between locations."},
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ])
    print(f"Response: {response1.content}")

    # Second call - should use cached tools and system
    print("\n📞 Second call:")
    response2 = llm_with_tools.invoke([
        {"role": "system", "content": "You are a helpful travel assistant. Always provide accurate information about weather and distances. Use the available tools to get real-time data when users ask about weather or distances between locations."},
        {"role": "user", "content": "How far is it from New York to Boston?"}
    ])
    print(f"Response: {response2.content}")

    # Get cache statistics
    print("\n📊 Cache Statistics:")
    stats = cache_handler.get_stats()
    print(f"Cache hit rate: {stats.cache_hit_rate:.1f}%")
    print(f"Total tokens cached: {stats.total_tokens_cached:,}")
    print(f"Total tokens skipped: {stats.total_tokens_skipped:,}")
    print(f"Estimated savings: ${stats.estimated_savings:.2f}")

if __name__ == "__main__":
    run_example()