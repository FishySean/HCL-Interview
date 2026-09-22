"""A language model in the decision path is only safe if being wrong, slow, or
malformed all degrade to the deterministic policy. These tests assert each of
those three failure modes individually, because "the fallback works" is the
whole argument for letting a model near a robot at all."""

from __future__ import annotations

import asyncio

from conftest import RecordingTracer, run

from robot_runtime.behavior.policies import DeadlineArbiter, MockVLMPolicy, RulePolicy
from robot_runtime.runtime.clock import FakeClock

CANDIDATES = ("greet_returning_visitor", "greet_visitor")
CONTEXT = {"returning_visitor": True}


class StubPolicy:
    name = "stub"

    def __init__(self, clock, latency_s=0.0, answer=None, error=None):
        self._clock = clock
        self._latency_s = latency_s
        self._answer = answer
        self._error = error
        self.calls = 0

    async def decide(self, candidates, context):
        self.calls += 1
        await self._clock.sleep(self._latency_s)
        if self._error is not None:
            raise self._error
        return self._answer if self._answer is not None else candidates[0]


def arbitrate(primary, clock, deadline_s=0.25, candidates=CANDIDATES, context=CONTEXT):
    tracer = RecordingTracer()
    arbiter = DeadlineArbiter(primary, RulePolicy(), clock, tracer, deadline_s=deadline_s)

    async def main():
        task = asyncio.create_task(arbiter.choose(candidates, context, trace_id="t1"))
        await clock.advance(10.0)
        return await task, tracer

    return run(main)


def test_a_single_candidate_never_spends_the_latency_budget():
    clock = FakeClock()
    primary = StubPolicy(clock, latency_s=0.2)
    decision, _ = arbitrate(primary, clock, candidates=("greet_visitor",))

    assert decision.behavior == "greet_visitor"
    assert decision.policy == "trivial"
    assert primary.calls == 0


def test_an_answer_within_budget_is_used():
    clock = FakeClock()
    primary = StubPolicy(clock, latency_s=0.1, answer="greet_returning_visitor")
    decision, _ = arbitrate(primary, clock, deadline_s=0.25)

    assert decision.behavior == "greet_returning_visitor"
    assert decision.policy == "stub"
    assert decision.fallback_reason == ""


def test_a_late_answer_is_discarded_for_the_deterministic_one():
    """A decision that arrives after the deadline is wrong even if it is correct."""
    clock = FakeClock()
    primary = StubPolicy(clock, latency_s=3.0, answer="greet_returning_visitor")
    decision, tracer = arbitrate(primary, clock, deadline_s=0.2)

    assert decision.policy == "rule"
    assert decision.behavior == CANDIDATES[0]
    assert "deadline" in decision.fallback_reason
    assert tracer.of_kind("policy.fallback")


def test_a_hallucinated_action_cannot_reach_the_motion_layer():
    clock = FakeClock()
    primary = StubPolicy(clock, latency_s=0.05, answer="perform_a_backflip")
    decision, _ = arbitrate(primary, clock)

    assert decision.policy == "rule"
    assert decision.behavior in CANDIDATES
    assert "perform_a_backflip" in decision.fallback_reason


def test_a_crashing_policy_degrades_instead_of_propagating():
    clock = FakeClock()
    primary = StubPolicy(clock, latency_s=0.05, error=RuntimeError("vlm offline"))
    decision, _ = arbitrate(primary, clock)

    assert decision.policy == "rule"
    assert "vlm offline" in decision.fallback_reason


def test_the_rule_policy_prefers_the_highest_priority_candidate():
    """The registry sorts best-first, so the deterministic floor is priority order."""

    async def main():
        return await RulePolicy().decide(["return_to_idle", "ambient_breathing"], {})

    assert run(main) == "return_to_idle"


def test_the_mock_vlm_prefers_the_warmer_greeting_for_a_returning_visitor():
    clock = FakeClock()
    primary = MockVLMPolicy(clock, latency_s=0.05)
    decision, _ = arbitrate(primary, clock)

    assert decision.behavior == "greet_returning_visitor"
    assert decision.policy == "mock-vlm"


def test_the_mock_vlm_falls_back_for_a_first_time_visitor():
    clock = FakeClock()
    primary = MockVLMPolicy(clock, latency_s=0.05)
    decision, _ = arbitrate(primary, clock, context={"returning_visitor": False})

    assert decision.behavior == CANDIDATES[0]
