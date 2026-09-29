"""Provider-agnostic configuration, read from the environment.

Three variables, so switching vendors is a deployment decision rather than a
code change:

    VLM_PROVIDER   gemini | openai | anthropic   (default: gemini)
    VLM_API_KEY    the credential                (required)
    VLM_MODEL      model id                      (default: per provider)

Nothing here ever logs or repr-s the key. `VLMConfig.__repr__` is defined by
hand for that reason: a dataclass repr would print the credential into the
first stack trace that touched it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

SUPPORTED_PROVIDERS = ("gemini", "openai", "anthropic")

DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
}

# Kept close to the decision deadline on purpose. The deadline is what protects
# the robot; this timeout is what stops a worker thread outliving the decision
# it was serving.
DEFAULT_TIMEOUT_S = 2.0
DEFAULT_MAX_OUTPUT_TOKENS = 128


class VLMConfigError(RuntimeError):
    """Raised for a misconfigured or absent credential.

    Always carries an actionable message: the caller's job is to print it and
    fall back, not to interpret it.
    """


@dataclass(frozen=True)
class VLMConfig:
    provider: str
    api_key: str
    model: str
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    def __repr__(self) -> str:
        return f"VLMConfig(provider={self.provider!r}, model={self.model!r}, api_key=<redacted>)"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides) -> VLMConfig:
        source = os.environ if env is None else env

        provider = (source.get("VLM_PROVIDER") or "gemini").strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise VLMConfigError(
                f"VLM_PROVIDER={provider!r} is not supported. "
                f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
            )

        api_key = (source.get("VLM_API_KEY") or "").strip()
        if not api_key:
            raise VLMConfigError(
                "VLM_API_KEY is not set, so the real vision-language policy cannot start.\n"
                f"  export VLM_PROVIDER={provider}\n"
                "  export VLM_API_KEY=<your key>\n"
                f"  export VLM_MODEL={DEFAULT_MODELS[provider]}   # optional"
            )

        model = (source.get("VLM_MODEL") or "").strip() or DEFAULT_MODELS[provider]
        return cls(provider=provider, api_key=api_key, model=model, **overrides)
