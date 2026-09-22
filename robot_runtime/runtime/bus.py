"""Typed asynchronous publish/subscribe bus.

Why a bus and not direct calls: the three layers have different natural
frequencies -- perception ~30Hz, behavior event-driven ~1-5Hz, motion ~50Hz.
Wiring them together with direct calls forces all three down to the slowest one,
so a slow decision would stall the camera loop. Here every subscriber owns a
queue and runs as its own task, and a slow consumer degrades only itself.

`publish` is synchronous and never blocks, so a real-time producer can never be
back-pressured by a slow consumer. When a queue overflows the *oldest* event is
dropped: for a robot, a stale perception frame is worth less than a fresh one.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable

from ..contracts.events import Event


class Subscription:
    def __init__(self, types: tuple[type, ...], maxsize: int, name: str) -> None:
        self.types = types
        self.name = name
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def accepts(self, event: Event) -> bool:
        return isinstance(event, self.types)

    def offer(self, event: Event) -> None:
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self.queue.put_nowait(event)

    async def get(self) -> Event:
        return await self.queue.get()


class EventBus:
    def __init__(self, tracer: Any | None = None) -> None:
        self._subscriptions: list[Subscription] = []
        self._tracer = tracer
        self.published = 0

    def subscribe(self, *types: type, maxsize: int = 128, name: str = "anon") -> Subscription:
        subscription = Subscription(types=types or (Event,), maxsize=maxsize, name=name)
        self._subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)

    def publish(self, event: Event) -> None:
        self.published += 1
        if self._tracer is not None:
            self._tracer.record("bus.publish", event.trace_id, event=type(event).__name__)
        for subscription in self._subscriptions:
            if subscription.accepts(event):
                subscription.offer(event)

    def publish_all(self, events: Iterable[Event]) -> None:
        for event in events:
            self.publish(event)

    @property
    def dropped(self) -> int:
        return sum(s.dropped for s in self._subscriptions)
