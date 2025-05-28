# LangChain Anthropic Smart Cache

🚀 **Intelligent cache management for LangChain Anthropic models with advanced optimization strategies**

[![PyPI version](https://badge.fury.io/py/langchain-anthropic-smart-cache.svg)](https://badge.fury.io/py/langchain-anthropic-smart-cache)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **📚 Learn about Anthropic's prompt caching:** [Official Documentation](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)

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
pip install langchain-anthropic-smart-cache
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

## 🧠 How Smart Caching Works

### 🎯 Priority-Based Cache Management

The system uses a sophisticated 5-level priority system to ensure the most valuable content gets cached first:

```mermaid
graph TD
    A[Incoming Request] --> B{Analyze Content}
    B --> C[Tools Available?]
    B --> D[System Prompts?]
    B --> E[User Content]

    C --> F{Tools Cached?}
    F -->|No| G[Priority 1: Cache Tools]
    F -->|Yes, Fresh| H[Skip Tools]
    F -->|Yes, Expiring| I[Priority 4: Refresh Tools]

    D --> J{System Cached?}
    J -->|No| K[Priority 2: Cache System]
    J -->|Yes, Fresh| L[Skip System]
    J -->|Yes, Expiring| M[Priority 5: Refresh System]

    E --> N[Priority 3: Cache Content by Size]

    G --> O[Allocate Cache Slots]
    K --> O
    N --> O
    I --> O
    M --> O

    O --> P{Slots Available?}
    P -->|Yes| Q[Apply Cache Control]
    P -->|No| R[Skip Lower Priority Items]
```

### 🔄 Cache Lifecycle Flow

```mermaid
sequenceDiagram
    participant User
    participant Handler as SmartCacheHandler
    participant Cache as Cache Storage
    participant Claude as Anthropic API

    User->>Handler: Send Request with Tools + System + Content
    Handler->>Cache: Check existing cache status
    Cache-->>Handler: Return cache metadata

    Note over Handler: Priority Analysis
    Handler->>Handler: Priority 1: Uncached Tools
    Handler->>Handler: Priority 2: Uncached System
    Handler->>Handler: Priority 3: Content (by size)
    Handler->>Handler: Priority 4: Expiring Tools
    Handler->>Handler: Priority 5: Expiring System

    Note over Handler: Slot Allocation (Max 4)
    Handler->>Handler: Assign cache_control headers
    Handler->>Claude: Send optimized request
    Claude-->>Handler: Response with cache info
    Handler->>Cache: Update cache metadata
    Handler-->>User: Response + Cache Analytics
```

### 🎲 Decision Algorithm

The cache decision algorithm follows this logic:

```mermaid
flowchart TD
    Start([New Request]) --> Clear[Clear Previous Cache Controls]
    Clear --> Parse[Parse Messages for Tools/System/Content]

    Parse --> CheckTools{Tools Present?}
    CheckTools -->|Yes| ToolsCached{Tools Cached & Fresh?}
    CheckTools -->|No| CheckSystem{System Prompts?}

    ToolsCached -->|No| AddTools[Add Tools - Priority 1]
    ToolsCached -->|Yes| CheckSystem

    CheckSystem -->|Yes| SystemCached{System Cached & Fresh?}
    CheckSystem -->|No| ProcessContent[Process Content Blocks]

    SystemCached -->|No| AddSystem[Add System - Priority 2]
    SystemCached -->|Yes| ProcessContent

    ProcessContent --> SortContent[Sort Content by Token Count]
    SortContent --> AddContent[Add Content - Priority 3]

    AddTools --> CheckSlots{Slots < 4?}
    AddSystem --> CheckSlots
    AddContent --> CheckSlots

    CheckSlots -->|Yes| MoreContent{More Items?}
    CheckSlots -->|No| RefreshCheck{Expired Items to Refresh?}

    MoreContent -->|Yes| AddContent
    MoreContent -->|No| RefreshCheck

    RefreshCheck -->|Yes| AddRefresh[Add Refresh - Priority 4/5]
    RefreshCheck -->|No| Complete[Complete Cache Assignment]

    AddRefresh --> FinalCheck{Slots < 4?}
    FinalCheck -->|Yes| RefreshCheck
    FinalCheck -->|No| Complete

    Complete --> SendRequest[Send to Anthropic API]
    SendRequest --> UpdateCache[Update Cache Metadata]
    UpdateCache --> End([Return Response])

    style AddTools fill:#ff6b6b
    style AddSystem fill:#4ecdc4
    style AddContent fill:#45b7d1
    style AddRefresh fill:#96ceb4
```

### 💡 Priority System Explained

| Priority | Type | Condition | Why? |
|----------|------|-----------|------|
| **1** 🔴 | Tools | Not cached or expired | Critical for function calling - failures break functionality |
| **2** 🟠 | System | Not cached or expired | Core instructions that define AI behavior |
| **3** 🟡 | Content | Always evaluated | User data, sorted by size for maximum cache efficiency |
| **4** 🟢 | Tools | Cached but expiring soon | Refresh tools proactively to avoid cache misses |
| **5** 🔵 | System | Cached but expiring soon | Refresh system prompts when slots available |

### 📊 Cache Efficiency Example

```mermaid
pie title Cache Slot Allocation Example
    "Tools (Priority 1)" : 25
    "System (Priority 2)" : 25
    "Large Content (Priority 3)" : 35
    "Medium Content (Priority 3)" : 15
```

**Scenario**: 4 available slots, competing content
- 🔴 **Slot 1**: Tools (3,000 tokens) - Priority 1 (uncached)
- 🟠 **Slot 2**: System prompt (1,200 tokens) - Priority 2 (uncached)
- 🟡 **Slot 3**: Large content (5,000 tokens) - Priority 3 (new)
- 🟡 **Slot 4**: Medium content (2,000 tokens) - Priority 3 (new)
- ❌ **Skipped**: Small content (800 tokens) - Priority 3 (below minimum)
- ❌ **Skipped**: Cached system refresh (1,200 tokens) - Priority 5 (no slots left)

**Result**: 11,200 tokens cached, optimizing for both functionality and cost savings.

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
- **langchain-core >= 0.3.62**
- **langchain-anthropic >= 0.3.14**
- **tiktoken >= 0.8.0**

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
