"""The real policy, without a network.

There are deliberately no tests here that call a hosted model. A unit test that
needs an API key and the internet is slow, costs money, fails for reasons that
have nothing to do with this repository, and -- because a language model is
non-deterministic -- cannot assert much anyway. What *is* worth pinning down is
everything between the socket and the robot: the request we build, the answer
we accept, and what happens to every answer we do not. So the HTTP transport is
the seam, and it is stubbed.

The safety argument these tests make is that a real model is no more dangerous
than the fake one, because it reaches the actuators through exactly the same
deadline, validation, and fallback path.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import RecordingTracer, run

from robot_runtime.behavior.policies import DeadlineArbiter, RulePolicy
from robot_runtime.behavior.vlm import (
    RealVLMPolicy,
    VLMConfig,
    VLMConfigError,
    VLMRejected,
    build_prompt,
)
from robot_runtime.behavior.vlm.providers import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
    VLMProtocolError,
)
from robot_runtime.behavior.vlm.transport import VLMTransportError
from robot_runtime.runtime.clock import FakeClock

CANDIDATES = ("greet_returning_visitor", "greet_visitor")
CONTEXT = {
    "state": "idle",
    "event": "PersonAppeared",
    "person_present": True,
    "returning_visitor": True,
    "scene": "one person, close",
    "candidates": {
        "greet_returning_visitor": "welcome someone we have already met",
        "greet_visitor": "greet a new arrival",
    },
}

API_KEY = "test-key-do-not-log"


def config(**overrides) -> VLMConfig:
    env = {"VLM_PROVIDER": "gemini", "VLM_API_KEY": API_KEY, "VLM_MODEL": "gemini-test"}
    return VLMConfig.from_env(env, **overrides)


def gemini_reply(behavior: str, reason: str = "because") -> dict:
    """The shape the real endpoint returns, down to the nested JSON string."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps({"behavior": behavior, "reason": reason})}]
                }
            }
        ]
    }


class StubTransport:
    """Stands in for the network. Records what we would have sent."""

    def __init__(self, response=None, error=None, clock=None, latency_s=0.0):
        self._response = response
        self._error = error
        self._clock = clock
        self._latency_s = latency_s
        self.calls: list[dict] = []

    async def post_json(self, url, headers, payload, timeout_s):
        self.calls.append(
            {"url": url, "headers": dict(headers), "payload": payload, "timeout_s": timeout_s}
        )
        if self._clock is not None and self._latency_s:
            await self._clock.sleep(self._latency_s)
        if self._error is not None:
            raise self._error
        return self._response


def decide(transport, context=CONTEXT, candidates=CANDIDATES, **kwargs):
    policy = RealVLMPolicy(config(), transport=transport, **kwargs)

    async def main():
        return await policy.decide(candidates, context)

    return run(main), policy


def arbitrate(transport, clock, deadline_s=0.25, context=CONTEXT):
    """Run the real policy through the unmodified arbiter."""
    tracer = RecordingTracer()
    policy = RealVLMPolicy(config(), transport=transport)
    arbiter = DeadlineArbiter(policy, RulePolicy(), clock, tracer, deadline_s=deadline_s)

    async def main():
        task = asyncio.create_task(arbiter.choose(CANDIDATES, context, trace_id="t1"))
        await clock.advance(10.0)
        return await task

    return run(main)


# --------------------------------------------------------------- happy path


def test_a_valid_answer_is_used_and_its_reason_kept():
    transport = StubTransport(gemini_reply("greet_returning_visitor", "we have met before"))
    chosen, policy = decide(transport)

    assert chosen == "greet_returning_visitor"
    assert policy.last_reason == "we have met before"
    assert policy.name == "vlm-gemini"


def test_the_schema_pins_the_answer_to_exactly_the_registered_behaviors():
    """The first line of defence is structural: an illegal id violates the schema."""
    transport = StubTransport(gemini_reply("greet_visitor"))
    decide(transport)

    schema = transport.calls[0]["payload"]["generationConfig"]["responseSchema"]
    assert schema["properties"]["behavior"]["enum"] == list(CANDIDATES)
    assert schema["required"] == ["behavior", "reason"]
    assert transport.calls[0]["payload"]["generationConfig"]["responseMimeType"] == "application/json"


def test_the_prompt_carries_the_state_and_what_each_behavior_means():
    prompt = build_prompt(CANDIDATES, CONTEXT, has_image=False)

    assert "idle" in prompt and "PersonAppeared" in prompt
    assert "welcome someone we have already met" in prompt
    assert "No camera frame" in prompt


# ------------------------------------------------------------- the picture


def test_a_camera_frame_is_sent_along_when_one_is_available():
    transport = StubTransport(gemini_reply("greet_visitor"))
    decide(transport, context={**CONTEXT, "frame_jpeg": b"\xff\xd8jpeg-bytes", "frame_age_s": 0.3})

    parts = transport.calls[0]["payload"]["contents"][0]["parts"]
    inline = [part for part in parts if "inline_data" in part]
    assert len(inline) == 1
    assert inline[0]["inline_data"]["mime_type"] == "image/jpeg"


def test_a_stale_frame_is_dropped_rather_than_describing_a_room_that_has_emptied():
    transport = StubTransport(gemini_reply("greet_visitor"))
    decide(transport, context={**CONTEXT, "frame_jpeg": b"\xff\xd8old", "frame_age_s": 30.0})

    parts = transport.calls[0]["payload"]["contents"][0]["parts"]
    assert all("inline_data" not in part for part in parts)


def test_a_text_only_decision_works_when_there_is_no_camera():
    transport = StubTransport(gemini_reply("greet_visitor"))
    chosen, _ = decide(transport)

    parts = transport.calls[0]["payload"]["contents"][0]["parts"]
    assert chosen == "greet_visitor"
    assert len(parts) == 1 and "text" in parts[0]


# --------------------------------------------------------------- rejection


def test_an_unregistered_behavior_is_refused_by_the_policy():
    transport = StubTransport(gemini_reply("perform_a_backflip"))

    with pytest.raises(VLMRejected, match="perform_a_backflip"):
        decide(transport)


def test_a_response_that_is_not_json_is_refused():
    reply = {"candidates": [{"content": {"parts": [{"text": "sure! let's wave"}]}}]}

    with pytest.raises(VLMProtocolError):
        decide(StubTransport(reply))


def test_a_response_missing_the_expected_fields_is_refused():
    with pytest.raises(VLMProtocolError):
        decide(StubTransport({"promptFeedback": {"blockReason": "SAFETY"}}))


# ------------------------------------------- the same fallback as the fake


def test_an_unregistered_behavior_falls_back_to_the_deterministic_policy():
    clock = FakeClock()
    decision = arbitrate(StubTransport(gemini_reply("perform_a_backflip")), clock)

    assert decision.policy == "rule"
    assert decision.behavior in CANDIDATES
    assert "perform_a_backflip" in decision.fallback_reason


def test_a_slow_model_loses_to_the_deadline():
    """The robot's responsiveness cannot depend on someone else's p99 latency."""
    clock = FakeClock()
    transport = StubTransport(gemini_reply("greet_returning_visitor"), clock=clock, latency_s=4.0)
    decision = arbitrate(transport, clock, deadline_s=0.25)

    assert decision.policy == "rule"
    assert decision.behavior == CANDIDATES[0]
    assert "deadline" in decision.fallback_reason


def test_an_unreachable_api_degrades_instead_of_crashing_the_robot():
    clock = FakeClock()
    transport = StubTransport(error=VLMTransportError("could not reach the model API"))
    decision = arbitrate(transport, clock)

    assert decision.policy == "rule"
    assert "could not reach" in decision.fallback_reason


# ------------------------------------------------------------------ config


def test_a_missing_key_raises_something_a_human_can_act_on():
    with pytest.raises(VLMConfigError) as excinfo:
        VLMConfig.from_env({"VLM_PROVIDER": "gemini"})

    assert "VLM_API_KEY" in str(excinfo.value)


def test_an_unknown_provider_is_rejected_at_configuration_time():
    with pytest.raises(VLMConfigError, match="not supported"):
        VLMConfig.from_env({"VLM_PROVIDER": "skynet", "VLM_API_KEY": "x"})


def test_the_model_defaults_per_provider_so_only_a_key_is_required():
    assert VLMConfig.from_env({"VLM_API_KEY": "x"}).model.startswith("gemini")


def test_the_credential_stays_out_of_logs_and_urls():
    """Keys leak through reprs and query strings; both paths are closed here."""
    transport = StubTransport(gemini_reply("greet_visitor"))
    _, policy = decide(transport)
    call = transport.calls[0]

    assert API_KEY not in repr(config())
    assert API_KEY not in call["url"]
    assert API_KEY not in json.dumps(call["payload"])
    assert call["headers"]["x-goog-api-key"] == API_KEY


def test_the_unimplemented_providers_say_so_instead_of_failing_obscurely():
    for provider in (OpenAIProvider, AnthropicProvider):
        with pytest.raises(NotImplementedError, match="gemini"):
            provider(VLMConfig(provider="gemini", api_key="x", model="m"))


def test_the_gemini_adapter_targets_the_configured_model():
    transport = StubTransport(gemini_reply("greet_visitor"))
    decide(transport)

    assert "gemini-test:generateContent" in transport.calls[0]["url"]
    assert GeminiProvider.name == "gemini"


# ---------------------------------------------------------------- the app


def test_the_app_falls_back_to_the_fake_policy_when_no_key_is_configured(monkeypatch):
    """A deployment mistake must not stop a robot from working."""
    from robot_runtime.perception.detectors.simulated import ManualDetector
    from robot_runtime.runtime.app import AppConfig, RobotApp

    for name in ("VLM_API_KEY", "VLM_PROVIDER", "VLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    lines: list[str] = []
    app = RobotApp(
        detector=ManualDetector(),
        clock=FakeClock(),
        config=AppConfig(vlm_policy="real", verbose=False),
        sink=lines.append,
    )

    assert app.policy.name == "mock-vlm"
    assert any("VLM_API_KEY" in line for line in lines)
