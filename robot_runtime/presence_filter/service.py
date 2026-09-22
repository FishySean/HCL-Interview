"""The presence filter as a pipeline stage.

Subscribes to raw `PersonObserved` frames and publishes only debounced
`PersonAppeared` / `PersonLeft` events. Nothing downstream subscribes to raw
observations, so the behavior layer structurally cannot react to a single
flickering frame.

This costs one extra hop and one event per frame on the bus. That is the price
of the boundary being enforceable instead of merely documented, and the bus
drops the oldest observation under overflow, which for a 30Hz sensor stream is
the right thing to lose.
"""

from __future__ import annotations

import asyncio

from ..contracts.events import PersonObserved
from ..contracts.protocols import Clock, Tracer
from ..runtime.bus import EventBus
from .filter import PresenceConfig, PresenceDebouncer


class PresenceFilterService:
    def __init__(
        self,
        bus: EventBus,
        clock: Clock,
        tracer: Tracer,
        config: PresenceConfig | None = None,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._tracer = tracer
        self._debouncer = PresenceDebouncer(config)
        self._subscription = bus.subscribe(PersonObserved, maxsize=256, name="presence-filter")
        self._task: asyncio.Task[None] | None = None
        self.observations = 0

    @property
    def present(self) -> bool:
        return self._debouncer.present

    @property
    def config(self) -> PresenceConfig:
        return self._debouncer.config

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="presence-filter")

    async def stop(self) -> None:
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
            raw = await self._subscription.get()
            assert isinstance(raw, PersonObserved)
            self.observations += 1
            for event in self._debouncer.update(raw.observation, raw.timestamp, raw.trace_id):
                self._tracer.record(
                    "presence.event",
                    event.trace_id,
                    event=type(event).__name__,
                    confidence=round(raw.observation.confidence, 3),
                    source=raw.observation.source,
                )
                self._bus.publish(event)
