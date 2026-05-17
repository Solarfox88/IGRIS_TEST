"""Tests for chat streaming + session tier selector (Sprint 16)."""

from __future__ import annotations

import json

import pytest

from igris.core.chat_streaming import (
    AVAILABLE_TIERS,
    StreamChunk,
    TierConfig,
    chat_stream_sync,
    get_current_tier,
    get_tier_availability,
    set_tier,
)


class TestTierConfig:
    def test_default_tier(self):
        t = get_current_tier()
        assert t.tier in AVAILABLE_TIERS

    def test_to_dict(self):
        t = TierConfig(tier="auto", description="test")
        d = t.to_dict()
        assert d["tier"] == "auto"
        assert "available_tiers" in d
        assert set(d["available_tiers"]) == set(AVAILABLE_TIERS)

    def test_set_valid_tier(self):
        original = get_current_tier().tier
        try:
            for tier in AVAILABLE_TIERS:
                result = set_tier(tier)
                assert result.tier == tier
                assert get_current_tier().tier == tier
        finally:
            set_tier(original)

    def test_set_invalid_tier(self):
        with pytest.raises(ValueError):
            set_tier("vast")

    def test_set_invalid_tier_empty(self):
        with pytest.raises(ValueError):
            set_tier("")


class TestStreamChunk:
    def test_content_chunk(self):
        c = StreamChunk(type="content", text="hello")
        sse = c.to_sse()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        parsed = json.loads(sse[6:].strip())
        assert parsed["type"] == "content"
        assert parsed["text"] == "hello"

    def test_done_chunk(self):
        c = StreamChunk(type="done", metadata={"provider": "test"})
        sse = c.to_sse()
        parsed = json.loads(sse[6:].strip())
        assert parsed["type"] == "done"
        assert parsed["metadata"]["provider"] == "test"


@pytest.mark.slow
class TestChatStreamSync:
    """Marked slow: makes real LLM calls via chat_stream_sync."""

    def test_returns_chunks(self):
        chunks = chat_stream_sync("hello")
        assert len(chunks) >= 1
        # Last chunk should be done
        assert chunks[-1].type == "done"
        assert "provider" in chunks[-1].metadata

    def test_content_chunks_have_text(self):
        chunks = chat_stream_sync("help me with tests")
        content_chunks = [c for c in chunks if c.type == "content"]
        assert len(content_chunks) >= 1
        full_text = "".join(c.text for c in content_chunks)
        assert len(full_text) > 0

    def test_metadata_in_done_chunk(self):
        chunks = chat_stream_sync("status")
        done = chunks[-1]
        assert done.type == "done"
        assert "tier" in done.metadata
        assert "latency_ms" in done.metadata
        assert "routing_reason" in done.metadata

    def test_no_secrets_in_response(self):
        chunks = chat_stream_sync("show me sk-abcdefghijklmnopqrstuvwxyz")
        full_text = "".join(c.text for c in chunks if c.type == "content")
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in full_text

    def test_with_history(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        chunks = chat_stream_sync("what was my first message?", history=history)
        assert len(chunks) >= 1


class TestTierAvailability:
    def test_availability_structure(self):
        avail = get_tier_availability()
        assert "current_tier" in avail
        assert "tiers" in avail
        assert "auto" in avail["tiers"]
        assert "local" in avail["tiers"]
        assert "fallback" in avail["tiers"]

    def test_auto_always_available(self):
        avail = get_tier_availability()
        assert avail["tiers"]["auto"]["available"] is True
