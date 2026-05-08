"""LLM provider interface and implementations."""
from __future__ import annotations
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    watch_pct: float
    liked: bool
    commented: bool
    shared: bool
    reason: str
    raw_response: str = ""


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    def decide(self, prompt: str) -> LLMResponse:
        ...

    def _parse_response(self, text: str) -> LLMResponse:
        """Parse JSON response from LLM, with fallback."""
        text = text.strip()
        # Try to extract JSON from markdown code blocks
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

        try:
            data = json.loads(text)
            return LLMResponse(
                watch_pct=float(data.get("watch_pct", 0.5)),
                liked=bool(data.get("liked", False)),
                commented=bool(data.get("commented", False)),
                shared=bool(data.get("shared", False)),
                reason=str(data.get("reason", "")),
                raw_response=text,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return LLMResponse(
                watch_pct=0.5, liked=False, commented=False,
                shared=False, reason=f"parse_error: {text[:100]}",
                raw_response=text,
            )


class MockProvider(LLMProvider):
    """Mock provider for testing. Returns deterministic responses based on prompt keywords."""

    def decide(self, prompt: str) -> LLMResponse:
        prompt_lower = prompt.lower()
        # Simulate context-sensitive decisions
        watch_pct = 0.5
        liked = False

        if "first" in prompt_lower or "first_visit" in prompt_lower:
            watch_pct = 0.3  # new users are impatient
        if "stall" in prompt_lower or "stall_count" in prompt_lower:
            watch_pct *= 0.7
        if "360p" in prompt_lower or "480p" in prompt_lower:
            watch_pct *= 0.8
        if "1080p" in prompt_lower:
            watch_pct *= 1.1

        watch_pct = max(0.0, min(1.0, watch_pct))
        liked = watch_pct > 0.6

        return LLMResponse(
            watch_pct=watch_pct, liked=liked, commented=False, shared=False,
            reason="mock_decision", raw_response="{}",
        )


class OpenAICompatibleProvider(LLMProvider):
    """Provider for any OpenAI-compatible API (OpenAI, DeepSeek, Ollama, etc.)."""

    def __init__(
        self,
        base_url: str = "https://api.deepseek.com",
        api_key: str | None = None,
        model: str = "deepseek-chat",
        max_tokens: int = 150,
        temperature: float = 0.7,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        if not self.api_key:
            raise ValueError("API key required. Set LLM_API_KEY env var or pass api_key.")

    def decide(self, prompt: str) -> LLMResponse:
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/chat/completions"
        if "/v1/" not in url and not url.endswith("/v1/chat/completions"):
            if "/chat/completions" not in url:
                url = f"{self.base_url}/v1/chat/completions"

        # For DeepSeek, the endpoint is /chat/completions (no /v1/)
        # For OpenAI, it's /v1/chat/completions
        # We handle both by checking the base_url
        if "deepseek" in self.base_url:
            url = f"{self.base_url}/chat/completions"
        elif "openai" in self.base_url:
            url = f"{self.base_url}/chat/completions"
            if "/v1" not in self.base_url:
                url = f"{self.base_url}/v1/chat/completions"
        else:
            url = f"{self.base_url}/v1/chat/completions"

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a user behavior simulator. Always respond with valid JSON only, no markdown."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = result["choices"][0]["message"]["content"]
                return self._parse_response(text)
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as e:
            return LLMResponse(
                watch_pct=0.5, liked=False, commented=False, shared=False,
                reason=f"api_error: {str(e)[:100]}", raw_response=str(e),
            )


class ClaudeProvider(LLMProvider):
    """Provider for Anthropic Claude API (non-OpenAI format)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 150,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key.")

    def decide(self, prompt: str) -> LLMResponse:
        import urllib.request
        import urllib.error

        url = "https://api.anthropic.com/v1/messages"
        payload = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "user", "content": f"You are a user behavior simulator. Always respond with valid JSON only, no markdown.\n\n{prompt}"},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = result["content"][0]["text"]
                return self._parse_response(text)
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as e:
            return LLMResponse(
                watch_pct=0.5, liked=False, commented=False, shared=False,
                reason=f"claude_api_error: {str(e)[:100]}", raw_response=str(e),
            )


def create_provider(provider_type: str = "mock", **kwargs) -> LLMProvider:
    """Factory function to create LLM providers.

    provider_type: "mock", "deepseek", "openai", "ollama", or "custom"
    """
    if provider_type == "mock":
        return MockProvider()
    elif provider_type == "deepseek":
        return OpenAICompatibleProvider(
            base_url="https://api.deepseek.com",
            model=kwargs.get("model", "deepseek-chat"),
            api_key=kwargs.get("api_key"),
            **{k: v for k, v in kwargs.items() if k not in ("model", "api_key")},
        )
    elif provider_type == "openai":
        return OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            model=kwargs.get("model", "gpt-4o-mini"),
            api_key=kwargs.get("api_key"),
            **{k: v for k, v in kwargs.items() if k not in ("model", "api_key")},
        )
    elif provider_type == "ollama":
        return OpenAICompatibleProvider(
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            model=kwargs.get("model", "qwen2.5"),
            api_key=kwargs.get("api_key", "ollama"),
            **{k: v for k, v in kwargs.items() if k not in ("base_url", "model", "api_key")},
        )
    elif provider_type == "claude":
        return ClaudeProvider(
            model=kwargs.get("model", "claude-sonnet-4-6"),
            api_key=kwargs.get("api_key"),
        )
    elif provider_type == "custom":
        return OpenAICompatibleProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")
