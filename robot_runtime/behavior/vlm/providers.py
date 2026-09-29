"""Vendor adapters.

A provider owns two things and nothing else: how to shape a request, and how to
dig the answer out of a response. Prompt construction, validation, deadlines,
and fallback all live above this layer, so adding a vendor never touches the
robot's decision logic.

The important shared behaviour is **structured output**. Every adapter must
constrain the `behavior` field to an enum of exactly the candidate ids the
registry produced, so a valid response is structurally incapable of naming an
action the robot does not have. The policy still validates the answer -- a
model can ignore a schema, an API can change, and this is the last thing
between a language model and an actuator -- but the schema means that check is
a guard rail rather than the only line of defence.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Mapping, Protocol, Sequence

from .config import VLMConfig

SYSTEM_PROMPT = (
    "You are the decision layer of a small interactive robot. "
    "Choose exactly one behavior from the provided list for the robot to perform now. "
    "You may only choose an id from that list; never invent one. "
    "If an image is provided, use it to judge what the person is doing and which "
    "greeting fits. Prefer the simplest behavior that fits. "
    "Answer with JSON only."
)


class VLMProtocolError(RuntimeError):
    """The response did not have the shape the adapter expects."""


class VLMProvider(Protocol):
    name: str

    def request(
        self, prompt: str, candidates: Sequence[str], image_jpeg: bytes | None
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Return (url, headers, payload)."""

    def parse(self, response: Mapping[str, Any]) -> tuple[str, str]:
        """Return (behavior_id, one-line reason)."""


class GeminiProvider:
    name = "gemini"
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, config: VLMConfig) -> None:
        self._config = config

    def request(
        self, prompt: str, candidates: Sequence[str], image_jpeg: bytes | None
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        parts: list[dict[str, Any]] = [{"text": prompt}]
        if image_jpeg:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(image_jpeg).decode("ascii"),
                    }
                }
            )

        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        # The guarantee: the model cannot name a behavior the
                        # registry did not produce without violating the schema.
                        "behavior": {"type": "STRING", "enum": list(candidates)},
                        "reason": {"type": "STRING"},
                    },
                    "required": ["behavior", "reason"],
                },
                "temperature": 0.2,
                "maxOutputTokens": self._config.max_output_tokens,
            },
        }
        # The key goes in a header, never in the query string: URLs end up in
        # logs, proxies, and crash reports.
        headers = {"x-goog-api-key": self._config.api_key}
        return self.ENDPOINT.format(model=self._config.model), headers, payload

    def parse(self, response: Mapping[str, Any]) -> tuple[str, str]:
        try:
            text = response["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VLMProtocolError(f"unexpected Gemini response shape: {_summarise(response)}") from exc

        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise VLMProtocolError(f"model did not return JSON: {text[:200]!r}") from exc

        if not isinstance(decoded, dict) or "behavior" not in decoded:
            raise VLMProtocolError(f"model JSON has no 'behavior' field: {decoded!r}")

        return str(decoded["behavior"]), str(decoded.get("reason", ""))


class OpenAIProvider:
    """Not implemented yet; the seam is here so adding it is a contained change.

    The equivalent guarantee is `response_format={"type": "json_schema", ...}`
    with `strict: true`, and images go in as a `image_url` content part holding
    a `data:image/jpeg;base64,...` URI.
    """

    name = "openai"
    ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def __init__(self, config: VLMConfig) -> None:
        raise NotImplementedError(
            "The OpenAI adapter is not implemented. Use VLM_PROVIDER=gemini, "
            "or implement OpenAIProvider.request/parse in behavior/vlm/providers.py."
        )


class AnthropicProvider:
    """Not implemented yet; the seam is here so adding it is a contained change.

    The equivalent guarantee is a single forced tool whose `input_schema`
    carries the enum, and images go in as a `source` block of type `base64`.
    """

    name = "anthropic"
    ENDPOINT = "https://api.anthropic.com/v1/messages"

    def __init__(self, config: VLMConfig) -> None:
        raise NotImplementedError(
            "The Anthropic adapter is not implemented. Use VLM_PROVIDER=gemini, "
            "or implement AnthropicProvider.request/parse in behavior/vlm/providers.py."
        )


PROVIDERS: dict[str, type] = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def build_provider(config: VLMConfig) -> VLMProvider:
    provider_class = PROVIDERS.get(config.provider)
    if provider_class is None:  # pragma: no cover - VLMConfig validates first
        raise NotImplementedError(f"no adapter for provider {config.provider!r}")
    return provider_class(config)


def _summarise(response: Mapping[str, Any]) -> str:
    try:
        return json.dumps(response)[:200]
    except (TypeError, ValueError):
        return repr(response)[:200]
