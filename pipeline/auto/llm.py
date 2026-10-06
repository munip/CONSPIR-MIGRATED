"""Local-model clients that return schema-checked JSON. Standard library plus jsonschema.

Local-only by default: a base URL that is not loopback is refused unless allow_remote=True, so a
contributor's source text and drafts are not sent to a hosted API by accident.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

import jsonschema

DEFAULT_MODEL = "qwen2.5:7b"
LOOPBACK = ("localhost", "127.0.0.1", "::1")


class LLMError(RuntimeError):
    """The model could not produce valid JSON for the schema."""


def _parse(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 < end:
            return json.loads(raw[start:end + 1])
        raise


class LLM:
    """Subclasses implement _chat(system, user, schema, temperature) -> str."""
    name = "base"

    def __init__(self, retries: int = 2):
        self.retries = retries
        self.calls = 0

    def _chat(self, system: str, user: str, schema: dict, temperature: float) -> str:
        raise NotImplementedError

    def json(self, system: str, user: str, schema: dict, *, temperature: float = 0.2) -> dict:
        problem = ""
        for _ in range(self.retries + 1):
            self.calls += 1
            suffix = f"\n\nYour previous reply was rejected: {problem}\nReply with ONLY valid JSON matching the schema." if problem else ""
            raw = self._chat(system, user + suffix, schema, temperature)
            try:
                data = _parse(raw)
                jsonschema.validate(data, schema)
                return data
            except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
                problem = getattr(exc, "message", str(exc))[:300]
        raise LLMError(f"{self.name}: no valid JSON after {self.retries + 1} tries ({problem})")


def _post(url: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise LLMError(f"{url}: HTTP {exc.code}: {exc.read()[:300].decode('utf-8', 'replace')}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMError(f"cannot reach {url}: {exc}. Is the model server running?") from exc


def _guard_local(base_url: str, allow_remote: bool) -> None:
    host = urllib.parse.urlparse(base_url).hostname
    if host not in LOOPBACK and not allow_remote:
        raise LLMError(f"{base_url} is not a local address; pass --allow-remote-llm if you really want to send source text there")


class OllamaLLM(LLM):
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = "http://localhost:11434", *, num_ctx: int = 8192,
                 timeout: int = 600, allow_remote: bool = False, retries: int = 2):
        super().__init__(retries)
        _guard_local(base_url, allow_remote)
        self.model, self.base, self.num_ctx, self.timeout = model, base_url.rstrip("/"), num_ctx, timeout
        self.name = f"ollama:{model}"

    def _chat(self, system, user, schema, temperature):
        body = {"model": self.model, "stream": False, "format": schema, "think": False,
                "options": {"temperature": temperature, "num_ctx": self.num_ctx},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        try:
            out = _post(self.base + "/api/chat", body, self.timeout)
        except LLMError as exc:
            if "think" not in str(exc):
                raise
            body.pop("think")
            out = _post(self.base + "/api/chat", body, self.timeout)
        return out["message"]["content"]


class OpenAICompatLLM(LLM):
    """llama.cpp server, vLLM, LM Studio: anything exposing /v1/chat/completions."""

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", *, timeout: int = 600,
                 allow_remote: bool = False, retries: int = 2):
        super().__init__(retries)
        _guard_local(base_url, allow_remote)
        self.model, self.base, self.timeout = model, base_url.rstrip("/"), timeout
        self.name = f"openai-compat:{model}"

    def _chat(self, system, user, schema, temperature):
        body = {"model": self.model, "temperature": temperature, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system + "\nJSON schema of the reply:\n" + json.dumps(schema)},
                             {"role": "user", "content": user}]}
        return _post(self.base + "/chat/completions", body, self.timeout)["choices"][0]["message"]["content"]


class ScriptedLLM(LLM):
    """Deterministic stand-in for tests: handler(system, user, schema) -> dict."""
    name = "scripted"

    def __init__(self, handler: Callable[[str, str, dict], dict]):
        super().__init__(retries=0)
        self.handler = handler

    def _chat(self, system, user, schema, temperature):
        return json.dumps(self.handler(system, user, schema))


def make_llm(backend: str, model: str, base_url: str | None, allow_remote: bool = False) -> LLM:
    if backend == "ollama":
        return OllamaLLM(model, base_url or "http://localhost:11434", allow_remote=allow_remote)
    if backend == "openai":
        return OpenAICompatLLM(model, base_url or "http://localhost:8000/v1", allow_remote=allow_remote)
    raise LLMError(f"unknown backend {backend!r} (ollama | openai)")
