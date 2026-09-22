"""Arbitration between behaviors, and the seam where a VLM plugs in.

The design problem: a language model makes a robot flexible and unpredictable
at the same time, and unpredictable is a safety property, not a style choice.
Two constraints make it safe here.

1. Constrained output. A policy may only return a name drawn from the
   candidate list the registry produced. Anything else -- a hallucinated
   action, a malformed reply, an exception -- is discarded by `DeadlineArbiter`
   and the deterministic `RulePolicy` decides instead. The model can be wrong;
   it cannot be dangerous.

2. A deadline. This is a real-time system: a decision that arrives late is
   wrong even if it is correct. `DeadlineArbiter` races the policy against the
   clock and falls back the instant the budget is spent.

Paired with the reflex arc (see `reflexes.py`), which dispatches the safe part
of the reaction *before* deliberation starts, the user-visible latency is the
reflex latency while the thinking happens underneath it.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..contracts.protocols import Clock, Tracer

DEFAULT_DEADLINE_S = 0.25


@dataclass(frozen=True)
class Decision:
    behavior: str
    policy: str
    latency_ms: float
    fallback_reason: str = ""


class RulePolicy:
    """Deterministic, instant, and always available. The floor under everything."""

    name = "rule"

    async def decide(self, candidates: Sequence[str], context: Mapping[str, Any]) -> str:
        if not candidates:
            raise ValueError("no candidates")
        return candidates[0]  # registry already sorted them best-first


class MockVLMPolicy:
    """Stands in for a vision-language model.

    It exists to make the *failure modes* testable, not to be smart: latency,
    hallucinated action names, and outright errors are all injectable and
    seeded, so the deadline-and-fallback path is covered by real tests rather
    than by an argument in a README.
    """

    name = "mock-vlm"

    def __init__(
        self,
        clock: Clock,
        latency_s: float = 0.12,
        invalid_rate: float = 0.0,
        error_rate: float = 0.0,
        seed: int = 11,
    ) -> None:
        self._clock = clock
        self._latency_s = latency_s
        self._invalid_rate = invalid_rate
        self._error_rate = error_rate
        self._random = random.Random(seed)
        self.calls = 0

    async def decide(self, candidates: Sequence[str], context: Mapping[str, Any]) -> str:
        self.calls += 1
        await self._clock.sleep(self._latency_s)

        if self._random.random() < self._error_rate:
            raise RuntimeError("vlm backend unavailable")
        if self._random.random() < self._invalid_rate:
            return "perform_a_backflip"  # exactly what the guard rail is for

        # A real VLM would reason over the frame and the dialogue history. The
        # shape of the answer is what matters at this seam: a registered name.
        if context.get("returning_visitor") and "greet_returning_visitor" in candidates:
            return "greet_returning_visitor"
        return candidates[0]


class DeadlineArbiter:
    """Runs `primary` under a time budget and validates whatever comes back."""

    def __init__(
        self,
        primary: Any,
        fallback: Any,
        clock: Clock,
        tracer: Tracer,
        deadline_s: float = DEFAULT_DEADLINE_S,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._clock = clock
        self._tracer = tracer
        self._deadline_s = deadline_s

    async def choose(
        self,
        candidates: Sequence[str],
        context: Mapping[str, Any],
        trace_id: str = "-",
    ) -> Decision:
        if not candidates:
            raise ValueError("no candidates")
        if len(candidates) == 1:
            # Nothing to arbitrate: do not spend the latency budget.
            return Decision(behavior=candidates[0], policy="trivial", latency_ms=0.0)

        started = self._clock.now()
        decision_task = asyncio.create_task(self._primary.decide(list(candidates), dict(context)))
        timer = asyncio.create_task(self._clock.sleep(self._deadline_s))

        try:
            done, _ = await asyncio.wait({decision_task, timer}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            decision_task.cancel()
            timer.cancel()
            raise

        latency_ms = (self._clock.now() - started) * 1000.0

        if decision_task not in done:
            decision_task.cancel()
            await asyncio.gather(decision_task, return_exceptions=True)
            return await self._fall_back(candidates, context, latency_ms, "deadline exceeded", trace_id)

        timer.cancel()
        error = decision_task.exception()
        if error is not None:
            return await self._fall_back(candidates, context, latency_ms, repr(error), trace_id)

        chosen = decision_task.result()
        if chosen not in candidates:
            return await self._fall_back(
                candidates, context, latency_ms, f"unregistered behavior {chosen!r}", trace_id
            )

        self._tracer.record(
            "policy.decide", trace_id, policy=self._primary.name, behavior=chosen, latency_ms=round(latency_ms, 1)
        )
        return Decision(behavior=chosen, policy=self._primary.name, latency_ms=latency_ms)

    async def _fall_back(
        self,
        candidates: Sequence[str],
        context: Mapping[str, Any],
        latency_ms: float,
        reason: str,
        trace_id: str,
    ) -> Decision:
        chosen = await self._fallback.decide(list(candidates), dict(context))
        self._tracer.record(
            "policy.fallback",
            trace_id,
            policy=self._fallback.name,
            behavior=chosen,
            latency_ms=round(latency_ms, 1),
            reason=reason,
        )
        return Decision(
            behavior=chosen,
            policy=self._fallback.name,
            latency_ms=latency_ms,
            fallback_reason=reason,
        )
