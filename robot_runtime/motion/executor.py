"""Executes approved motions, enforces deadlines, and tells the truth.

What this class no longer does is as important as what it does: it does not
decide whether a motion is safe, and it does not own the arbitration policy.
Both moved to the safety gate. What is left here is execution:

1. Deadlines. Every primitive runs under a watchdog, so a motion that hangs
   becomes TIMED_OUT instead of a robot frozen mid-gesture forever.

2. Degradation. On a retryable outcome the executor walks the primitive's
   fallback ladder (wave -> nod). A substituted primitive is screened by the
   gate exactly like the original, so the invariant holds: nothing reaches the
   backend without a safety verdict.

3. Preemption mechanics. The gate decides *whether* a motion may be preempted;
   the executor owns the tasks, so it performs the cancellation.

Failure is always a returned `MotionResult`, never an exception that escapes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..contracts.enums import MotionStatus
from ..contracts.events import ApprovedMotion, MotionCommand, MotionResult
from ..contracts.primitives import PrimitiveRegistry
from ..contracts.protocols import Arbiter, Clock, MotionBackend, Tracer
from ..contracts.safety import Denial, Screened
from .backends.simulated import MotionFailure


@dataclass
class _Running:
    task: asyncio.Task[str]
    command_id: str
    primitive: str


class MotionExecutor:
    def __init__(
        self,
        backend: MotionBackend,
        gate: Arbiter,
        registry: PrimitiveRegistry,
        clock: Clock,
        tracer: Tracer,
    ) -> None:
        self._backend = backend
        self._gate = gate
        self._registry = registry
        self._clock = clock
        self._tracer = tracer
        self._running: dict[str, _Running] = {}

    async def execute(self, approved: ApprovedMotion) -> MotionResult:
        command = approved.command
        primitive = self._registry.require(command.primitive)
        first = Screened(
            primitive=primitive,
            params=primitive.with_defaults(command.params),
            limbs=approved.limbs,
            joint_targets=approved.joint_targets,
        )

        attempts: list[str] = []
        result: MotionResult | None = None
        chain: tuple[str, ...] = (command.primitive, *first.primitive.fallbacks)

        for index, name in enumerate(chain):
            # The head of the chain was screened by the pipeline stage; every
            # substitution has to earn its own verdict.
            screened = first if index == 0 else self._gate.screen(name, command.params, command.trace_id)
            if isinstance(screened, Denial):
                attempts.append(name)
                result = self._result(
                    command, name, MotionStatus.REJECTED, (), 0.0, tuple(attempts),
                    f"safety gate: {screened.reason.value}: {screened.detail}",
                )
                break

            attempts.append(name)
            status, duration, detail = await self._run_once(command, screened)
            result = self._result(
                command, name, status, screened.limbs, duration, tuple(attempts), detail
            )
            self._tracer.record(
                "motion.attempt",
                command.trace_id,
                primitive=name,
                status=status.value,
                limbs=list(screened.limbs),
                duration_s=round(duration, 3),
                detail=detail,
            )
            if status.is_success or not status.is_retryable:
                return result

        assert result is not None
        return result

    async def _run_once(
        self, command: MotionCommand, screened: Screened
    ) -> tuple[MotionStatus, float, str]:
        acquisition = self._gate.acquire(
            command_id=command.command_id,
            priority=command.priority,
            limbs=screened.limbs,
            primitive=screened.primitive.name,
            trace_id=command.trace_id,
        )
        if isinstance(acquisition, Denial):
            return MotionStatus.REJECTED, 0.0, f"safety gate: {acquisition.reason.value}: {acquisition.detail}"

        for command_id in acquisition.preempted:
            victim = self._running.get(command_id)
            if victim is not None and not victim.task.done():
                victim.task.cancel()
                await asyncio.gather(victim.task, return_exceptions=True)

        primitive = screened.primitive
        params = dict(screened.params)
        duration = primitive.duration_for(params)
        timeout = command.timeout_s if command.timeout_s is not None else primitive.timeout_s
        started = self._clock.now()

        work = asyncio.create_task(
            self._backend.run(primitive.name, params, duration), name=f"motion:{primitive.name}"
        )
        watchdog = asyncio.create_task(self._clock.sleep(timeout), name=f"watchdog:{primitive.name}")
        self._running[command.command_id] = _Running(
            task=work, command_id=command.command_id, primitive=primitive.name
        )

        try:
            done, _ = await asyncio.wait({work, watchdog}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            work.cancel()
            watchdog.cancel()
            raise
        finally:
            self._running.pop(command.command_id, None)
            self._gate.release(command.command_id)

        elapsed = self._clock.now() - started

        if work in done:
            watchdog.cancel()
            if work.cancelled():
                return MotionStatus.PREEMPTED, elapsed, "preempted by a higher-priority command"
            error = work.exception()
            if error is None:
                return MotionStatus.COMPLETED, elapsed, work.result()
            if isinstance(error, MotionFailure):
                return MotionStatus.FAILED, elapsed, str(error)
            return MotionStatus.FAILED, elapsed, repr(error)

        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        return MotionStatus.TIMED_OUT, elapsed, f"exceeded {timeout:.1f}s deadline"

    def _result(
        self,
        command: MotionCommand,
        primitive: str,
        status: MotionStatus,
        limbs: tuple[str, ...],
        duration: float,
        attempts: tuple[str, ...],
        detail: str,
    ) -> MotionResult:
        return MotionResult(
            trace_id=command.trace_id,
            timestamp=self._clock.now(),
            command_id=command.command_id,
            primitive=primitive,
            status=status,
            limbs=limbs,
            duration_s=duration,
            attempts=attempts,
            detail=detail,
        )
