import copy
from datetime import datetime

from langchain_anthropic_smart_cache.cache import CacheManager, CacheEntry
from langchain_anthropic_smart_cache.core import SmartCacheCallbackHandler


def test_cache_key_ignores_cache_control():
    cm = CacheManager()

    content_clean = {"tools": [{"name": "search", "description": "tool"}]}
    content_with_cc = copy.deepcopy(content_clean)
    content_with_cc["tools"][0]["cache_control"] = {"type": "ephemeral"}

    k1 = cm._get_cache_key(content_clean)
    k2 = cm._get_cache_key(content_with_cc)

    assert k1 == k2, "cache key must ignore cache_control"


class _Msg:
    def __init__(self, role: str, content):
        self.type = role
        self.content = content
        self.additional_kwargs = {}


def test_skip_cached_excludes_valid_cached_from_candidates():
    handler = SmartCacheCallbackHandler(enable_logging=False)
    # Make everything eligible regardless of size
    handler.min_token_count = 0

    # Message 1 (cached) and Message 2 (new)
    msg1_dict = {"role": "system", "content": [{"type": "text", "text": "A" * 10}]}
    # Ensure enough length to be considered cacheable by analyzer (>=50 tokens approx)
    msg2_dict = {"role": "content", "content": [{"type": "text", "text": "B" * 400}]}

    m1 = _Msg("system", msg1_dict["content"])  # cached
    m2 = _Msg("human", msg2_dict["content"])   # new

    # Monkeypatch cache_manager.get to return cached entry for msg1 only
    orig_get = handler.cache_manager.get

    def _fake_get(content):
        # content is a dict with keys role/content as created by _message_to_dict
        if content.get("content") == msg1_dict["content"] and content.get("role") == msg1_dict["role"]:
            return CacheEntry(
                key="k",
                content=content,
                cached_at=datetime.now(),
                token_count=100,
                content_type="system",
            )
        return None

    handler.cache_manager.get = _fake_get

    try:
        # Invoke with two messages; expect only m2 to receive cache_control
        serialized = {"kwargs": {}}
        handler.on_chat_model_start(serialized, [[m1, m2]])

        # m1 should NOT have cache_control
        assert all(
            not (isinstance(it, dict) and "cache_control" in it) for it in m1.content
        ), "cached message must be skipped and not tagged"

        # m2 SHOULD have cache_control on last block
        assert any(
            isinstance(it, dict) and it.get("cache_control") for it in m2.content
        ), "new message should be tagged with cache_control"
    finally:
        handler.cache_manager.get = orig_get


