"""A real vision-language model as the robot's deliberative policy.

This is an *addition*, not a replacement of any mechanism. It implements the
same `Policy` protocol as `RulePolicy` and `MockVLMPolicy`, so it drops into
the existing `DeadlineArbiter` untouched and inherits, for free, every
protection the fake one was written to exercise:

  * the decision is raced against a deadline, and a slow model loses;
  * an exception is a lost race, not a crash;
  * an answer outside the candidate list is discarded;
  * in every one of those cases the deterministic `RulePolicy` answers instead.

So the worst a misbehaving model can do is make the robot boring. It cannot
make the robot wrong, and it cannot make it late.

The policy's own job is therefore narrow: build a prompt, make the call, and
refuse anything that is not one of the candidates it was given.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ...contracts.protocols import Tracer
from .config import VLMConfig
from .providers import VLMProvider, build_provider
from .transport import HttpTransport, UrllibTransport

# A frame older than this describes a scene that has moved on. Better to decide
# from text than from a stale picture of an empty room.
MAX_FRAME_AGE_S = 3.0


class VLMRejected(RuntimeError):
    """The model answered, but not with something the robot can do."""


class RealVLMPolicy:
    """Asks a hosted vision-language model which behavior to run."""

    def __init__(
        self,
        config: VLMConfig,
        transport: HttpTransport | None = None,
        provider: VLMProvider | None = None,
        tracer: Tracer | None = None,
        max_frame_age_s: float = MAX_FRAME_AGE_S,
    ) -> None:
        self._config = config
        self._provider = provider or build_provider(config)
        self._transport = transport or UrllibTransport()
        self._tracer = tracer
        self._max_frame_age_s = max_frame_age_s
        self.name = f"vlm-{config.provider}"
        self.last_reason = ""

    async def decide(self, candidates: Sequence[str], context: Mapping[str, Any]) -> str:
        if not candidates:
            raise ValueError("no candidates")

        image = self._frame_for(context)
        prompt = build_prompt(candidates, context, has_image=image is not None)
        url, headers, payload = self._provider.request(prompt, candidates, image)

        response = await self._transport.post_json(url, headers, payload, self._config.timeout_s)
        behavior, reason = self._provider.parse(response)

        # The schema should already have made this impossible. Checking anyway,
        # because this is the last thing standing between a language model and
        # a motor, and "should" is not a safety property.
        if behavior not in candidates:
            raise VLMRejected(
                f"model chose {behavior!r}, which is not among {list(candidates)}"
            )

        self.last_reason = reason
        if self._tracer is not None:
            self._tracer.record(
                "policy.vlm", "-", provider=self._config.provider, behavior=behavior, reason=reason
            )
        return behavior

    def _frame_for(self, context: Mapping[str, Any]) -> bytes | None:
        frame = context.get("frame_jpeg")
        if not isinstance(frame, (bytes, bytearray)):
            return None
        age = context.get("frame_age_s")
        if isinstance(age, (int, float)) and age > self._max_frame_age_s:
            return None
        return bytes(frame)


def build_prompt(
    candidates: Sequence[str], context: Mapping[str, Any], has_image: bool
) -> str:
    """Describe the situation, then the menu.

    Only fields the decision may legitimately turn on go in here. The robot's
    state is included because the same perception event means different things
    depending on what the robot is already doing, and the candidate
    descriptions are included because ids alone ("greet_returning") tell a
    model far less than the registry already knows.
    """
    descriptions = context.get("candidates") or {}
    options = [
        {"id": name, "description": str(descriptions.get(name, ""))} for name in candidates
    ]

    situation = {
        "robot_state": context.get("state"),
        "trigger": context.get("event"),
        "person_present": context.get("person_present"),
        "returning_visitor": context.get("returning_visitor"),
        "scene": context.get("scene") or None,
    }

    lines = [
        "Situation:",
        json.dumps(situation, indent=2, default=str),
        "",
        "Available behaviors:",
        json.dumps(options, indent=2),
        "",
    ]
    lines.append(
        "A camera frame of the current scene is attached. Use it to judge what the "
        "person is doing and pick the behavior that best fits."
        if has_image
        else "No camera frame is available; decide from the description above."
    )
    lines.append('Reply as JSON: {"behavior": "<id>", "reason": "<one sentence>"}')
    return "\n".join(lines)
