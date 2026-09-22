"""Time is injected, never imported.

`FakeClock` is the reason the whole test suite runs in milliseconds and is
deterministic: a 1.5s presence-exit timeout or a 3s motion hang costs no real
time and never flakes on a loaded CI machine.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time


class RealClock:
    """Monotonic, but rebased to zero at construction so timestamps in logs and
    scripted timelines are readable and comparable across runs."""

    def __init__(self) -> None:
        self._origin = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._origin

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


class FakeClock:
    """Virtual time driven explicitly by `advance`."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self._waiters: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()

    def now(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (self._now + seconds, next(self._seq), future))
        await future

    async def advance(self, seconds: float) -> None:
        """Move virtual time forward, waking sleepers in due order and letting
        each woken task run to its next suspension before time moves again.

        The drain happens *before* each decision to pop, not only after, so a
        sleep registered by a task that was itself only just scheduled -- a
        motion restarted after a preemption, say -- is seen within this same
        advance instead of being stranded past the deadline.
        """
        target = self._now + seconds
        while True:
            await _drain()
            if not (self._waiters and self._waiters[0][0] <= target):
                break
            due, _, future = heapq.heappop(self._waiters)
            self._now = max(self._now, due)
            if not future.done():
                future.set_result(None)
        self._now = target
        await _drain()

    @property
    def pending_sleepers(self) -> int:
        return len(self._waiters)


_MAX_DRAIN_ROUNDS = 1000


async def _drain() -> None:
    """Yield until the event loop has no runnable callbacks left.

    A fixed number of yields is not enough: waking one sleeper can start a
    chain -- a motion times out, cancels its worker, awaits the cancellation,
    then starts a fallback primitive that registers a *new* sleep. If we stop
    yielding before that new sleep is registered, it is stranded past the
    advance deadline and the test hangs on a motion that can never complete.

    Reading the loop's ready queue is a private API, but this class is a test
    utility and the alternative -- guessing a yield count -- is exactly the bug
    described above. The round cap keeps a busy-looping task from wedging us.
    """
    loop = asyncio.get_running_loop()
    ready = getattr(loop, "_ready", None)
    for _ in range(_MAX_DRAIN_ROUNDS):
        await asyncio.sleep(0)
        if ready is not None and len(ready) == 0:
            return
    if ready is None:  # pragma: no cover - only on a non-CPython event loop
        for _ in range(16):
            await asyncio.sleep(0)
