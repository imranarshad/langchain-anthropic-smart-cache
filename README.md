# LangChain Anthropic Smart Cache

🚀 **Intelligent cache management for LangChain Anthropic models with advanced optimization strategies**

[![PyPI version](https://badge.fury.io/py/langchain-anthropic-smart-cache.svg)](https://badge.fury.io/py/langchain-anthropic-smart-cache)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ⚡ What is this?

A sophisticated callback handler that automatically optimizes Anthropic Claude's cache usage to **reduce costs and improve performance**. It implements intelligent priority-based caching that ensures your most important content (tools, system prompts, large content blocks) gets cached first.

## 🎯 Key Features

- **Smart Priority System**: Tools and system prompts get priority when not cached
- **Automatic Cache Management**: 5-minute cache duration with intelligent refresh
- **Cost Optimization**: Prioritizes larger content blocks for maximum savings
- **Detailed Analytics**: Comprehensive logging and cache efficiency metrics
- **Zero Configuration**: Works out of the box with sensible defaults
- **Anthropic Optimized**: Built specifically for Claude's cache_control feature

## 📦 Installation

```bash
pip install langchain-anthropic-smart-cache[anthropic]
```

## 🚀 Quick Start

```python
from langchain_anthropic import ChatAnthropic
from langchain_anthropic_smart_cache import SmartCacheCallbackHandler

# Initialize the cache handler
cache_handler = SmartCacheCallbackHandler(
    cache_duration=300,  # 5 minutes
    max_cache_blocks=4,  # Anthropic's limit
    min_token_count=1024  # Minimum tokens to cache
)

# Add to your LangChain model
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    callbacks=[cache_handler]
)

# Use normally - caching happens automatically!
response = llm.invoke([
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Hello!"}
])
```

## 🧠 How It Works

The handler implements a sophisticated priority system:

### Priority Levels
1. **Priority 1**: Uncached tools (critical for function calling)
2. **Priority 2**: Uncached system prompts (core instructions)
3. **Priority 3**: Content blocks (user data, sorted by size)
4. **Priority 4**: Cached tools (refresh if slots available)
5. **Priority 5**: Cached system prompts (refresh if slots available)

### Smart Caching Logic
```
🔄 Cache Cycle:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Tools Expired   │ →  │ Tools Priority 1│ →  │ Gets Slot 1     │
└─────────────────┘    └─────────────────┘    └─────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ System Cached   │ →  │ System Priority │ →  │ No Slot Needed │
│ (Still Valid)   │    │ 5 (Low)         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Content Mixed   │ →  │ Content Prior.3 │ →  │ Largest Blocks  │
│ (Some Cached)   │    │ (Size Sorted)   │    │ Get Remaining   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Cache Analytics

The handler provides detailed logging:

```
💾 CACHED tools (slot 1/4) - NEW tools needed caching
⚡ CACHED content (slot 2/4, 3001 tokens) - MAINTAIN existing cache
🔄 CACHED content (slot 3/4, 2000 tokens) - REFRESH expiring cache
💾 CACHED content (slot 4/4, 1705 tokens) - NEW content block

🚫 SKIPPED ITEMS (2 items):
  ❌ content (priority 3, new, 1524 tokens) - smaller new content, larger cached content prioritized
  ❌ system (priority 5, cached, 1182 tokens) - system already cached, content got priority

📊 CACHE SUMMARY:
  🎯 Slots used: 4/4
  ⚡ Previously cached: 2 items (50.0%)
  💾 Newly cached: 2 items
  🚫 Skipped: 2 items
  📈 Cached tokens: 7,886 | Skipped tokens: 2,706
```

## ⚙️ Configuration

```python
cache_handler = SmartCacheCallbackHandler(
    cache_duration=300,      # Cache validity in seconds (default: 5 minutes)
    max_cache_blocks=4,      # Max cache blocks (Anthropic limit: 4)
    min_token_count=1024,    # Minimum tokens to consider for caching
    enable_logging=True,     # Enable detailed cache logging
    log_level="INFO",        # Logging level
    cache_dir=None,          # Custom cache directory (default: temp)
)
```

## 🎯 Advanced Usage

### With Tools
```python
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

# Tools automatically get highest priority when not cached
llm_with_tools = llm.bind_tools([get_weather])
```

### Cache Statistics
```python
# Access cache statistics
stats = cache_handler.get_stats()
print(f"Cache hit rate: {stats.cache_hit_rate:.1f}%")
print(f"Total tokens cached: {stats.total_tokens_cached:,}")
print(f"Estimated cost savings: ${stats.estimated_savings:.2f}")
```

## 🔧 Requirements

- **Python 3.8+**
- **langchain-core >= 0.1.0**
- **langchain-anthropic >= 0.1.0**
- **tiktoken >= 0.5.0**

> **Note**: This package is specifically designed for Anthropic Claude models that support the `cache_control` feature. Other providers may be added in future versions.

## 📈 Performance Benefits

- **Cost Reduction**: Up to 90% savings on repeated content
- **Latency Improvement**: Cached content loads ~10x faster
- **Smart Prioritization**: Ensures most valuable content stays cached
- **Automatic Management**: No manual cache invalidation needed

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for the [LangChain](https://github.com/langchain-ai/langchain) ecosystem
- Optimized for [Anthropic Claude](https://www.anthropic.com/claude) models
- Inspired by modern caching strategies and cost optimization principles