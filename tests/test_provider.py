"""Tests for LLM provider parsing and edge cases."""
from __future__ import annotations
import pytest
from rec_sim.llm.provider import MockProvider, LLMResponse, OpenAICompatibleProvider


def test_parse_clean_json():
    provider = MockProvider()
    text = '{"watch_pct": 0.7, "liked": true, "commented": false, "shared": false, "reason": "good content"}'
    result = provider._parse_response(text)
    assert result.watch_pct == 0.7
    assert result.liked is True
    assert result.reason == "good content"


def test_parse_json_in_markdown():
    provider = MockProvider()
    text = '```json\n{"watch_pct": 0.8, "liked": false, "reason": "ok"}\n```'
    result = provider._parse_response(text)
    assert result.watch_pct == 0.8


def test_parse_invalid_json_fallback():
    provider = MockProvider()
    text = "I think the user would watch about 60% of the video"
    result = provider._parse_response(text)
    assert result.watch_pct == 0.5  # fallback
    assert "parse_error" in result.reason


def test_parse_partial_json():
    provider = MockProvider()
    text = '{"watch_pct": 0.9}'
    result = provider._parse_response(text)
    assert result.watch_pct == 0.9
    assert result.liked is False  # default


def test_openai_provider_requires_key():
    with pytest.raises(ValueError, match="API key required"):
        import os
        old = os.environ.pop("LLM_API_KEY", None)
        try:
            OpenAICompatibleProvider(api_key="")
        finally:
            if old:
                os.environ["LLM_API_KEY"] = old
