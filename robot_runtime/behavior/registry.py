"""The behavior plugin system: the extension point for "more behaviors".

A behavior declares, as data, when it applies and what it wants done. Adding
one is a new subclass plus a `register(...)` call -- no edit to the service,
the state machine, the policy, or the motion layer. That is the concrete answer
to the brief's requirement that future behaviors must not require rewriting the
application.

Gating (state legality, cooldown, trigger type) lives here rather than inside
each behavior, so every behavior gets it right by construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..contracts.enums import Priority, RobotState
from ..contracts.events import Event, MotionCommand, PlanStep, SceneContext


@dataclass
class BehaviorContext:
    state: RobotState
    event: Event | None
    now: float
    person_present: bool
    scene: SceneContext | None = None
    memory: dict[str, Any] = field(default_factory=dict)


class Behavior(ABC):
    name: str = "unnamed"
    description: str = ""
    priority: int = Priority.NORMAL
    cooldown_s: float = 0.0
    triggers: tuple[type[Event], ...] = ()
    valid_states: tuple[RobotState, ...] = ()
    success_state: RobotState = RobotState.IDLE
    failure_state: RobotState = RobotState.IDLE

    def matches(self, context: BehaviorContext) -> bool:
        """Extra conditions beyond trigger type and state. Default: none."""
        return True

    @abstractmethod
    def plan(self, context: BehaviorContext) -> list[PlanStep]:
        """The hierarchical output: a behavior emits *named primitives with
        parameters*, never joint trajectories. A hallucinated primitive name is
        rejected by the registry; a hallucinated trajectory would not be."""

    def command(self, primitive: str, trace_id: str, **params: Any) -> MotionCommand:
        return MotionCommand(
            trace_id=trace_id,
            primitive=primitive,
            params=params,
            priority=self.priority,
            issued_by=self.name,
        )


class BehaviorRegistry:
    def __init__(self) -> None:
        self._behaviors: dict[str, Behavior] = {}
        self._last_run: dict[str, float] = {}

    def register(self, behavior: Behavior) -> Behavior:
        if behavior.name in self._behaviors:
            raise ValueError(f"behavior {behavior.name!r} already registered")
        self._behaviors[behavior.name] = behavior
        return behavior

    def get(self, name: str) -> Behavior | None:
        return self._behaviors.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._behaviors))

    def mark_ran(self, name: str, now: float) -> None:
        self._last_run[name] = now

    def cooling_down(self, behavior: Behavior, now: float) -> bool:
        last = self._last_run.get(behavior.name)
        return last is not None and (now - last) < behavior.cooldown_s

    def candidates(self, context: BehaviorContext) -> list[Behavior]:
        """Everything eligible right now, best first. The policy picks one."""
        eligible = [
            behavior
            for behavior in self._behaviors.values()
            if self._eligible(behavior, context)
        ]
        eligible.sort(key=lambda b: (-b.priority, b.name))
        return eligible

    def _eligible(self, behavior: Behavior, context: BehaviorContext) -> bool:
        if behavior.valid_states and context.state not in behavior.valid_states:
            return False
        if behavior.triggers and not isinstance(context.event, behavior.triggers):
            return False
        if self.cooling_down(behavior, context.now):
            return False
        return behavior.matches(context)
