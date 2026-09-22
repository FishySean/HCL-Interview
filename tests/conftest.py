"""Test helpers.

`run` lets the whole suite be plain synchronous pytest functions, which keeps
the project free of a pytest-asyncio dependency. Combined with `FakeClock`,
every test below runs on virtual time: a 1.5-second presence timeout or a
3-second motion hang costs no wall-clock time and cannot flake on a loaded
machine.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from robot_runtime.runtime.clock import FakeClock


REAL_TIME_GUARD_S = 5.0


def run(main: Callable[[], Awaitable[Any]], guard_s: float = REAL_TIME_GUARD_S) -> Any:
    """Run a coroutine on the event loop under a wall-clock guard.

    Every test below runs on virtual time, so any real elapsed time at all means
    something is genuinely stuck. The guard turns that into a fast failure
    rather than a suite that hangs forever in CI.
    """

    async def guarded() -> Any:
        return await asyncio.wait_for(main(), timeout=guard_s)

    return asyncio.run(guarded())


class RecordingTracer:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def record(self, kind: str, trace_id: str, **fields: Any) -> None:
        self.records.append((kind, trace_id, fields))

    def kinds(self) -> list[str]:
        return [kind for kind, _, _ in self.records]

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [fields for k, _, fields in self.records if k == kind]


def fake_clock(start: float = 0.0) -> FakeClock:
    return FakeClock(start)
