"""Bridges the executor to the bus.

It subscribes to `ApprovedMotion` and never to `MotionCommand`, which is the
mechanism behind the claim that the safety gate cannot be bypassed: a command
that the gate did not approve has no subscriber in this layer.

Approved motions are executed in their own tasks rather than serially, because
the gate already arbitrates per limb. Serialising here would make "speak while
waving" impossible for no safety gain.
"""

from __future__ import annotations

import asyncio

from ..contracts.events import ApprovedMotion
from ..contracts.protocols import Tracer
from ..runtime.bus import EventBus
from .executor import MotionExecutor


class MotionService:
    def __init__(self, executor: MotionExecutor, bus: EventBus, tracer: Tracer) -> None:
        self._executor = executor
        self._bus = bus
        self._tracer = tracer
        self._subscription = bus.subscribe(ApprovedMotion, name="motion")
        self._task: asyncio.Task[None] | None = None
        self._inflight: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="motion-service")

    async def stop(self) -> None:
        for task in list(self._inflight):
            task.cancel()
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._bus.unsubscribe(self._subscription)

    async def _run(self) -> None:
        while True:
            approved = await self._subscription.get()
            assert isinstance(approved, ApprovedMotion)
            task = asyncio.create_task(
                self._dispatch(approved), name=f"cmd:{approved.command.primitive}"
            )
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)

    async def _dispatch(self, approved: ApprovedMotion) -> None:
        command = approved.command
        self._tracer.record(
            "motion.command",
            command.trace_id,
            primitive=command.primitive,
            priority=command.priority,
            limbs=list(approved.limbs),
            issued_by=command.issued_by,
        )
        try:
            result = await self._executor.execute(approved)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._tracer.record("motion.crash", command.trace_id, error=repr(exc))
            return
        self._tracer.record(
            "motion.result",
            result.trace_id,
            primitive=result.primitive,
            status=result.status.value,
            attempts=list(result.attempts),
        )
        self._bus.publish(result)
