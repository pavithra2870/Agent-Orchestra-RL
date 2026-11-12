"""Thin client for a local Ollama model (default: phi4-mini). No API key, no cloud calls."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple

import httpx

from ecommerce.config import get_settings


class OllamaClient:
    def __init__(self, url: Optional[str] = None, model: Optional[str] = None, timeout: Optional[float] = None,
                 temperature: Optional[float] = None):
        cfg = get_settings().llm
        self.url = (url or cfg.ollama_url).rstrip("/")
        self.model = model or cfg.model
        self.timeout = timeout or cfg.timeout_seconds
        self.temperature = cfg.temperature if temperature is None else temperature
        self.last_error: Optional[str] = None

    def available(self) -> bool:
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=3.0)
            return any(m["name"] == self.model for m in r.json().get("models", []))
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def _chat(self, system: str, user: str, schema: Optional[Dict[str, Any]], max_tokens: int) -> Tuple[Optional[str], float]:
        payload: Dict[str, Any] = {
            "model": self.model, "stream": False, "keep_alive": "30m",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": self.temperature, "num_predict": max_tokens, "num_ctx": 2048, "seed": 7},
        }
        if schema is not None:
            payload["format"] = schema
        t = time.time()
        try:
            r = httpx.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["message"]["content"], time.time() - t
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None, time.time() - t

    def chat_json(self, system: str, user: str, schema: Dict[str, Any], max_tokens: int = 160):
        text, latency = self._chat(system, user, schema, max_tokens)
        if text is None:
            return None, latency
        try:
            return json.loads(text), latency
        except json.JSONDecodeError:
            self.last_error = "invalid JSON from model"
            return None, latency

    def chat_text(self, system: str, user: str, max_tokens: int = 200):
        return self._chat(system, user, None, max_tokens)
