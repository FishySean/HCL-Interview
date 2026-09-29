"""The perception loop: sample the detector, publish what it saw.

It reports observations and draws no conclusions. Debouncing lives in the
presence filter downstream, which keeps this layer to a single job -- turning
sensor data into a timestamped observation -- and lets the detector be swapped
(scripted, webcam, a real robot's sensor stack) without any of the temporal
reasoning moving with it.
"""

from __future__ import annotations

import asyncio

from ..contracts.events import CameraFrame, PersonObserved, new_id
from ..contracts.protocols import Clock, FrameSource, PersonDetector, Tracer
from ..runtime.bus import EventBus


class PerceptionService:
    def __init__(
        self,
        detector: PersonDetector,
        bus: EventBus,
        clock: Clock,
        tracer: Tracer,
        fps: float = 30.0,
        frame_hz: float = 1.0,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._detector = detector
        self._bus = bus
        self._clock = clock
        self._tracer = tracer
        self._period = 1.0 / fps
        # Keyframes for the deliberative tier go out at their own, much lower
        # rate. A detector without an image simply never produces any.
        self._frame_period = 1.0 / frame_hz if frame_hz > 0 else None
        self._last_frame_at: float | None = None
        self._task: asyncio.Task[None] | None = None
        self.frames = 0
        self.keyframes = 0

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="perception")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._detector.close()

    async def _run(self) -> None:
        while True:
            await self._tick()
            await self._clock.sleep(self._period)

    async def _tick(self) -> None:
        """A detector fault must degrade to 'no observation', never kill the loop:
        a crashed camera driver should not take the robot's brain with it."""
        try:
            observation = await self._detector.detect()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._tracer.record("perception.error", "-", error=repr(exc))
            return

        self.frames += 1
        self._bus.publish(
            PersonObserved(
                trace_id=new_id(),
                timestamp=self._clock.now(),
                observation=observation,
            )
        )
        self._maybe_publish_keyframe(observation.source)

    def _maybe_publish_keyframe(self, source: str) -> None:
        if self._frame_period is None or not isinstance(self._detector, FrameSource):
            return
        now = self._clock.now()
        if self._last_frame_at is not None and (now - self._last_frame_at) < self._frame_period:
            return
        try:
            jpeg = self._detector.latest_frame_jpeg()
        except Exception as exc:  # noqa: BLE001
            self._tracer.record("perception.frame_error", "-", error=repr(exc))
            return
        if not jpeg:
            return
        self._last_frame_at = now
        self.keyframes += 1
        self._bus.publish(
            CameraFrame(trace_id=new_id(), timestamp=now, jpeg=jpeg, source=source)
        )
