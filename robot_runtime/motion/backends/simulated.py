"""Simulated hardware.

The injectable `failure_rate` and `slow_rate` are the point of this class. The
brief says motion "may occasionally fail or take longer than expected", and a
backend that always succeeds cannot prove the system survives that. With a
seeded RNG, flakiness becomes a reproducible test fixture.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Callable, Mapping

from ...contracts.protocols import Clock


class MotionFailure(RuntimeError):
    """Raised by a backend when the hardware refuses or aborts a motion."""


class SimulatedBackend:
    name = "simulated"

    def __init__(
        self,
        clock: Clock,
        failure_rate: float = 0.0,
        slow_rate: float = 0.0,
        slow_factor: float = 4.0,
        seed: int = 7,
        sink: Callable[[str], None] | None = None,
    ) -> None:
        self._clock = clock
        self._failure_rate = failure_rate
        self._slow_rate = slow_rate
        self._slow_factor = slow_factor
        self._random = random.Random(seed)
        self._sink = sink
        self.executed: list[str] = []

    async def run(self, primitive: str, params: Mapping[str, Any], duration_s: float) -> str:
        roll_fail = self._random.random()
        roll_slow = self._random.random()

        if roll_fail < self._failure_rate:
            # Fail partway through, the way a stalled servo would.
            await self._clock.sleep(duration_s * 0.3)
            raise MotionFailure(f"{primitive}: simulated actuator fault")

        effective = duration_s * (self._slow_factor if roll_slow < self._slow_rate else 1.0)
        self._emit(primitive, params, effective, overrun=effective > duration_s)
        try:
            await self._clock.sleep(effective)
        except asyncio.CancelledError:
            self._log(f"  [motion] {primitive}: interrupted")
            raise
        self.executed.append(primitive)
        return f"{primitive} done in {effective:.2f}s"

    def _emit(self, primitive: str, params: Mapping[str, Any], duration: float, overrun: bool) -> None:
        if primitive == "say":
            self._log(f'  [voice] "{params.get("text", "")}"')
            return
        detail = ", ".join(f"{k}={v}" for k, v in params.items() if k != "text")
        suffix = "  (running slow)" if overrun else ""
        self._log(f"  [motion] {primitive}({detail}) ~{duration:.1f}s{suffix}")

    def _log(self, message: str) -> None:
        if self._sink is not None:
            self._sink(message)
