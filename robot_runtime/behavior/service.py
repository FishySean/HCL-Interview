"""The behavior layer's runtime: reflex, deliberate, plan, react to outcomes.

Two cooperating tasks, on purpose:

  * `_pump` only reads the bus and resolves motion results. It never awaits
    anything slow, so a motion result can always land, even while the policy is
    deliberating or a plan is mid-flight.
  * `_deliberate` does the thinking. If it blocked the bus reader, a plan
    waiting on a motion result and a decision waiting on the VLM would stall
    each other.

The loop is closed: every motion result comes back here, and every non-success
outcome has an explicit path. That is what the brief's "motion may fail or take
longer than expected" actually demands -- not a try/except, but a state machine
that cannot get stuck in GREETING because a wave never finished.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..contracts.enums import MotionStatus, RobotState
from ..contracts.events import (
    BehaviorFinished,
    BehaviorSelected,
    Event,
    MotionCommand,
    MotionResult,
    PerceptionEvent,
    PersonAppeared,
    PersonLeft,
    SceneContext,
)
from ..contracts.protocols import Clock, Tracer
from ..runtime.bus import EventBus
from .behaviors import register_builtins
from .policies import DeadlineArbiter, Decision, RulePolicy
from .registry import Behavior, BehaviorContext, BehaviorRegistry
from .reflexes import DEFAULT_REFLEXES, Reflex
from .state import IllegalTransition, RobotStateMachine

COMMAND_HARD_LIMIT_S = 15.0


class BehaviorService:
    def __init__(
        self,
        bus: EventBus,
        clock: Clock,
        tracer: Tracer,
        registry: BehaviorRegistry | None = None,
        arbiter: DeadlineArbiter | None = None,
        reflexes: tuple[Reflex, ...] = DEFAULT_REFLEXES,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._tracer = tracer
        self._registry = registry or register_builtins()
        self._arbiter = arbiter or DeadlineArbiter(RulePolicy(), RulePolicy(), clock, tracer)
        self._reflexes = reflexes

        self._fsm = RobotStateMachine()
        self._subscription = bus.subscribe(
            PerceptionEvent, MotionResult, BehaviorFinished, name="behavior"
        )
        self._inbox: asyncio.Queue[Event] = asyncio.Queue(maxsize=64)
        self._pending: dict[str, asyncio.Future[MotionResult]] = {}
        self._memory: dict[str, Any] = {}
        self._scene: SceneContext | None = None
        self._present = False

        self._plan_task: asyncio.Task[None] | None = None
        self._running: Behavior | None = None
        self._tasks: list[asyncio.Task[None]] = []

    # ----------------------------------------------------------------- state
    @property
    def state(self) -> RobotState:
        return self._fsm.state

    @property
    def registry(self) -> BehaviorRegistry:
        return self._registry

    # --------------------------------------------------------------- control
    async def start(self) -> None:
        if self._tasks:
            return
        self._tasks = [
            asyncio.create_task(self._pump(), name="behavior-pump"),
            asyncio.create_task(self._deliberate(), name="behavior-deliberate"),
        ]

    async def stop(self) -> None:
        await self._cancel_plan("shutdown")
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        self._bus.unsubscribe(self._subscription)

    # ------------------------------------------------------------------ loops
    async def _pump(self) -> None:
        while True:
            event = await self._subscription.get()
            if isinstance(event, MotionResult):
                self._resolve(event)
                continue
            if isinstance(event, PersonAppeared):
                self._present = True
            elif isinstance(event, PersonLeft):
                self._present = False
                self._memory["last_seen_at"] = event.timestamp
                self._memory["last_track_id"] = event.track_id
            elif isinstance(event, SceneContext):
                self._scene = event
                continue
            try:
                self._inbox.put_nowait(event)
            except asyncio.QueueFull:
                self._tracer.record("behavior.inbox_full", event.trace_id, event=type(event).__name__)

    async def _deliberate(self) -> None:
        while True:
            event = await self._inbox.get()
            try:
                await self._handle(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._tracer.record("behavior.error", event.trace_id, error=repr(exc))

    # ---------------------------------------------------------------- handling
    async def _handle(self, event: Event) -> None:
        self._fire_reflexes(event)

        context = self._context(event)
        candidates = self._registry.candidates(context)
        if not candidates:
            return

        if self._running is not None and self._plan_task is not None and not self._plan_task.done():
            if candidates[0].priority <= self._running.priority:
                self._tracer.record(
                    "behavior.suppressed",
                    event.trace_id,
                    candidate=candidates[0].name,
                    running=self._running.name,
                )
                return
            await self._cancel_plan(f"preempted by {candidates[0].name}")

        names = tuple(behavior.name for behavior in candidates)
        decision = await self._arbiter.choose(names, self._policy_context(context), event.trace_id)
        behavior = self._registry.get(decision.behavior)
        if behavior is None:  # defence in depth; the arbiter already validated
            self._tracer.record("behavior.unknown", event.trace_id, behavior=decision.behavior)
            return

        # The world may have moved on while the policy was thinking.
        fresh = self._context(event)
        if not self._registry.candidates(fresh) or behavior.name not in {
            b.name for b in self._registry.candidates(fresh)
        }:
            self._tracer.record("behavior.stale", event.trace_id, behavior=behavior.name)
            return

        self._announce(behavior, decision, fresh, names)
        self._registry.mark_ran(behavior.name, self._clock.now())
        self._running = behavior
        self._plan_task = asyncio.create_task(
            self._run_plan(behavior, fresh), name=f"plan:{behavior.name}"
        )

    def _fire_reflexes(self, event: Event) -> None:
        for reflex in self._reflexes:
            if not reflex.applies(event, self._fsm.state):
                continue
            if reflex.enter_state is not None:
                self._transition(reflex.enter_state, f"reflex:{reflex.name}", event.trace_id)
            command = MotionCommand(
                trace_id=event.trace_id,
                primitive=reflex.primitive,
                params=dict(reflex.params),
                priority=reflex.priority,
                issued_by=f"reflex:{reflex.name}",
            )
            self._tracer.record("behavior.reflex", event.trace_id, reflex=reflex.name, primitive=reflex.primitive)
            self._bus.publish(command)

    async def _run_plan(self, behavior: Behavior, context: BehaviorContext) -> None:
        trace_id = context.event.trace_id if context.event else "-"
        succeeded = True
        detail = ""
        try:
            for step in behavior.plan(context):
                if step.enter_state is not None and self._fsm.state is not step.enter_state:
                    self._transition(step.enter_state, f"{behavior.name}:step", trace_id)
                result = await self._send(step.command)
                if result.status.is_success or step.optional:
                    continue
                succeeded = False
                detail = f"{result.primitive} -> {result.status.value}: {result.detail}"
                break
        except asyncio.CancelledError:
            self._tracer.record("behavior.cancelled", trace_id, behavior=behavior.name)
            raise

        target = behavior.success_state if succeeded else behavior.failure_state
        self._transition(target, f"{behavior.name}:{'done' if succeeded else 'failed'}", trace_id)
        self._tracer.record(
            "behavior.finished", trace_id, behavior=behavior.name, succeeded=succeeded, detail=detail
        )
        self._running = None
        self._bus.publish(
            BehaviorFinished(
                trace_id=trace_id,
                timestamp=self._clock.now(),
                behavior=behavior.name,
                succeeded=succeeded,
                state_after=self._fsm.state,
                detail=detail,
            )
        )

    async def _cancel_plan(self, reason: str) -> None:
        task = self._plan_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._plan_task = None
        self._running = None
        for future in list(self._pending.values()):
            if not future.done():
                future.cancel()
        self._pending.clear()
        if reason:
            self._tracer.record("behavior.plan_cancelled", "-", reason=reason)

    # ------------------------------------------------------------ motion i/o
    async def _send(self, command: MotionCommand) -> MotionResult:
        """Publish a command and wait for its result, with a hard ceiling so a
        motion layer that never answers cannot wedge the behavior layer."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[MotionResult] = loop.create_future()
        self._pending[command.command_id] = future
        self._bus.publish(command)
        guard = asyncio.create_task(self._clock.sleep(COMMAND_HARD_LIMIT_S))
        try:
            done, _ = await asyncio.wait({future, guard}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            guard.cancel()
            raise
        finally:
            self._pending.pop(command.command_id, None)

        if future in done and not future.cancelled():
            guard.cancel()
            return future.result()

        future.cancel()
        self._tracer.record("behavior.lost_command", command.trace_id, primitive=command.primitive)
        return MotionResult(
            trace_id=command.trace_id,
            timestamp=self._clock.now(),
            command_id=command.command_id,
            primitive=command.primitive,
            status=MotionStatus.TIMED_OUT,
            detail="no result returned by the motion layer",
        )

    def _resolve(self, result: MotionResult) -> None:
        future = self._pending.pop(result.command_id, None)
        if future is not None and not future.done():
            future.set_result(result)

    # ---------------------------------------------------------------- helpers
    def _context(self, event: Event | None) -> BehaviorContext:
        return BehaviorContext(
            state=self._fsm.state,
            event=event,
            now=self._clock.now(),
            person_present=self._present,
            scene=self._scene,
            memory=self._memory,
        )

    def _policy_context(self, context: BehaviorContext) -> dict[str, Any]:
        event = context.event
        returning = False
        if isinstance(event, PersonAppeared) and event.track_id is not None:
            returning = self._memory.get("last_track_id") == event.track_id
        return {
            "state": context.state.value,
            "event": type(event).__name__ if event else None,
            "person_present": context.person_present,
            "returning_visitor": returning,
            "scene": context.scene.description if context.scene else "",
        }

    def _announce(
        self, behavior: Behavior, decision: Decision, context: BehaviorContext, considered: tuple[str, ...]
    ) -> None:
        self._tracer.record(
            "behavior.selected",
            context.event.trace_id if context.event else "-",
            behavior=behavior.name,
            policy=decision.policy,
            latency_ms=round(decision.latency_ms, 1),
            fallback=decision.fallback_reason,
        )
        self._bus.publish(
            BehaviorSelected(
                trace_id=context.event.trace_id if context.event else "-",
                timestamp=self._clock.now(),
                behavior=behavior.name,
                policy=decision.policy,
                state_before=context.state,
                latency_ms=decision.latency_ms,
                fallback_reason=decision.fallback_reason,
                considered=considered,
            )
        )

    def _transition(self, target: RobotState, reason: str, trace_id: str) -> None:
        try:
            change = self._fsm.transition(
                target, reason=reason, trace_id=trace_id, now=self._clock.now()
            )
        except IllegalTransition as exc:
            self._tracer.record("behavior.illegal_transition", trace_id, error=str(exc), reason=reason)
            return
        self._tracer.record("state.changed", trace_id, old=change.old.value, new=change.new.value, reason=reason)
        self._bus.publish(change)
