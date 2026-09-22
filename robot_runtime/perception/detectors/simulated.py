"""Simulated fast-tier detectors.

They implement exactly the same `PersonDetector` protocol as the real YOLO
detector, including its noise characteristics: `noise` injects dropped and
spurious frames so the debouncer is exercised the way real hardware would
exercise it, not the way a clean mock would.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from ...contracts.events import PersonObservation
from ...contracts.protocols import Clock


@dataclass(frozen=True)
class Segment:
    """`present` holds from `start_s` until the next segment begins."""

    start_s: float
    present: bool


class ScriptedDetector:
    """Replays a timeline against the injected clock. The backbone of the tests:
    a whole visit-and-leave scenario becomes three lines of data."""

    name = "scripted"

    def __init__(
        self,
        timeline: Sequence[Segment],
        clock: Clock,
        noise: float = 0.0,
        confidence: float = 0.92,
        seed: int = 0,
    ) -> None:
        if not timeline:
            raise ValueError("timeline must not be empty")
        self._timeline = sorted(timeline, key=lambda s: s.start_s)
        self._clock = clock
        self._noise = noise
        self._confidence = confidence
        self._random = random.Random(seed)

    async def detect(self) -> PersonObservation:
        now = self._clock.now()
        present = False
        for segment in self._timeline:
            if segment.start_s <= now:
                present = segment.present
            else:
                break
        if self._noise and self._random.random() < self._noise:
            present = not present
        return PersonObservation(
            present=present,
            confidence=self._confidence if present else 0.0,
            bbox=(0.35, 0.2, 0.65, 0.95) if present else None,
            track_id=1 if present else None,
            source=self.name,
        )

    async def close(self) -> None:
        return None


class ManualDetector:
    """Driven by the operator from the CLI ('p' / 'l'), for a live demo."""

    name = "manual"

    def __init__(self, present: bool = False, confidence: float = 0.95) -> None:
        self._present = present
        self._confidence = confidence

    def set_present(self, present: bool) -> None:
        self._present = present

    async def detect(self) -> PersonObservation:
        return PersonObservation(
            present=self._present,
            confidence=self._confidence if self._present else 0.0,
            bbox=(0.35, 0.2, 0.65, 0.95) if self._present else None,
            track_id=1 if self._present else None,
            source=self.name,
        )

    async def close(self) -> None:
        return None
