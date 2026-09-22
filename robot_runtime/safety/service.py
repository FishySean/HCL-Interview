"""The gate as a mandatory pipeline stage.

The behavior layer publishes `MotionCommand`. The motion layer subscribes to
`ApprovedMotion` and to nothing else. Only this service converts one into the
other, so there is no code path from a decision to an actuator that skips the
safety checks -- not a forgotten call site, not a new behavior written by
someone who never read this file, not a language model picking an action.

A denial is published twice on purpose: as a `MotionResult(REJECTED)` so the
waiting plan gets its answer immediately, and as a `MotionDenied` carrying the
reason, so denials are observable as their own signal. A rising rate of
JOINT_LIMIT denials means a behavior is misparameterised, and that is worth
alerting on; a plain rejection count would bury it among ordinary contention.
"""

from __future__ import annotations

import asyncio

from ..contracts.enums import MotionStatus
from ..contracts.events import ApprovedMotion, MotionCommand, MotionDenied, MotionResult
from ..contracts.protocols import Clock, Tracer
from ..runtime.bus import EventBus
from .gate import SafetyGate


class SafetyGateService:
    def __init__(self, gate: SafetyGate, bus: EventBus, clock: Clock, tracer: Tracer) -> None:
        self._gate = gate
        self._bus = bus
        self._clock = clock
        self._tracer = tracer
        self._subscription = bus.subscribe(MotionCommand, name="safety-gate")
        self._task: asyncio.Task[None] | None = None
        self.screened = 0

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="safety-gate")

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
            command = await self._subscription.get()
            assert isinstance(command, MotionCommand)
            self._screen(command)

    def _screen(self, command: MotionCommand) -> None:
        self.screened += 1
        verdict = self._gate.screen(command.primitive, command.params, command.trace_id)

        if not verdict.allowed:
            self._bus.publish(
                MotionDenied(
                    trace_id=command.trace_id,
                    timestamp=self._clock.now(),
                    command_id=command.command_id,
                    primitive=command.primitive,
                    reason=verdict.reason,
                    detail=verdict.detail,
                )
            )
            self._bus.publish(
                MotionResult(
                    trace_id=command.trace_id,
                    timestamp=self._clock.now(),
                    command_id=command.command_id,
                    primitive=command.primitive,
                    status=MotionStatus.REJECTED,
                    attempts=(command.primitive,),
                    detail=f"safety gate: {verdict.reason.value}: {verdict.detail}",
                )
            )
            return

        self._tracer.record(
            "safety.approved",
            command.trace_id,
            primitive=command.primitive,
            limbs=list(verdict.limbs),
            joints={name: round(angle, 3) for name, angle in verdict.joint_targets.items()},
        )
        self._bus.publish(
            ApprovedMotion(
                trace_id=command.trace_id,
                timestamp=self._clock.now(),
                command=command,
                limbs=verdict.limbs,
                joint_targets=dict(verdict.joint_targets),
            )
        )
