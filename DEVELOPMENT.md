# Development Guide

## Package Structure

```
langchain-anthropic-smart-cache/
├── langchain_anthropic_smart_cache/    # Main package
│   ├── __init__.py                     # Package exports
│   ├── core.py                         # SmartCacheCallbackHandler
│   ├── cache.py                        # CacheManager and utilities
│   └── utils.py                        # TokenCounter and ContentAnalyzer
├── tests/                              # Unit tests
│   └── test_smart_cache.py
├── examples/                           # Usage examples
│   └── basic_usage.py
├── .github/workflows/                  # CI/CD
│   └── test.yml
├── README.md                           # Package documentation
├── pyproject.toml                      # Modern Python packaging
├── setup.py                            # Legacy packaging support
├── requirements.txt                    # Runtime dependencies
├── requirements-dev.txt                # Development dependencies
├── LICENSE                             # MIT license
├── MANIFEST.in                         # Package manifest
├── Makefile                            # Development commands
└── test_integration.py                 # Integration tests
```

## Key Features

### 🧠 Smart Prioritization
- **Priority 1**: Uncached tools (critical for function calling)
- **Priority 2**: Uncached system prompts (core instructions)
- **Priority 3**: Content blocks (sorted by size)
- **Priority 4**: Cached tools (refresh if slots available)
- **Priority 5**: Cached system prompts (refresh if slots available)

### 💾 Intelligent Cache Management
- 5-minute cache duration with automatic refresh
- Maximum 4 cache blocks (Anthropic limit)
- Persistent disk storage with in-memory performance
- Automatic cleanup of expired entries

### 📊 Detailed Analytics
- Cache hit/miss rates
- Token usage statistics
- Cost estimation
- Comprehensive logging

## Development Commands

```bash
# Install development dependencies
make install-dev

# Run tests
make test

# Format code
make format

# Lint code
make lint

# Build package
make build

# Upload to TestPyPI
make upload-test
```

## Testing

### Unit Tests
```bash
pytest tests/ -v --cov=langchain_anthropic_smart_cache
```

### Integration Tests
```bash
python test_integration.py
```

### Manual Testing
```bash
python examples/basic_usage.py
```

## Architecture

### Core Components

1. **SmartCacheCallbackHandler**: Main LangChain callback handler
   - Implements `on_chat_model_start` hook
   - Manages cache slot allocation
   - Applies cache controls to messages and tools

2. **CacheManager**: Handles cache storage and retrieval
   - In-memory cache with disk persistence
   - Automatic expiration handling
   - Statistics tracking

3. **TokenCounter**: Estimates token counts
   - Uses tiktoken when available
   - Fallback estimation for any content type
   - Handles multimodal content

4. **ContentAnalyzer**: Analyzes content for caching decisions
   - Determines content type and complexity
   - Calculates caching priority
   - Provides caching recommendations

### Cache Priority Logic

```python
def calculate_priority(content_type, is_cached, token_count):
    if content_type == 'tools':
        return 1 if not is_cached else 4
    elif content_type == 'system':
        return 2 if not is_cached else 5
    else:
        return 3  # Content blocks sorted by size
```

### Cache Control Application

- **Tools**: Applied to last tool in the list
- **Messages**: Applied based on content structure
  - Multimodal: Cache control on last content block
  - Text: Cache control in additional_kwargs

## Performance Considerations

### Cache Hit Optimization
- Tools and system prompts cached first when missing
- Large content blocks prioritized
- Smart refresh of near-expiry items

### Memory Management
- Automatic cleanup of expired entries
- Periodic disk persistence
- Configurable cache limits

### Token Efficiency
- Minimum token thresholds prevent micro-caching
- Size-based prioritization maximizes savings
- Intelligent content analysis

## Publishing

### To TestPyPI
```bash
make upload-test
```

### To PyPI
```bash
make upload
```

## Configuration Options

```python
SmartCacheCallbackHandler(
    cache_duration=300,      # Cache validity (seconds)
    max_cache_blocks=4,      # Max cache slots
    min_token_count=1024,    # Min tokens to cache
    enable_logging=True,     # Detailed logging
    log_level="INFO",        # Log level
    cache_dir=None,          # Cache directory
)
```

## Anthropic-Specific Features

- Designed for Claude's `cache_control` feature
- Respects 4-block cache limit
- Optimized for Claude's token pricing
- Handles ephemeral cache type

## Future Enhancements

- Support for other providers with cache features
- Advanced cache warming strategies
- Dynamic cache duration based on content
- Cache sharing across sessions
- Performance profiling tools