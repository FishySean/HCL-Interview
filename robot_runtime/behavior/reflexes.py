"""The reflex arc: the fast path that runs before any deliberation.

Waking up when someone walks in is always safe and always correct, so there is
nothing to decide -- and deciding would cost the one thing a greeting cannot
spend, which is the first 200 milliseconds. The reflex is dispatched
immediately and the policy keeps thinking about *how* to greet underneath it.

This is how the latency-versus-expressiveness trade-off is paid for
structurally rather than by shortening a timeout: the person perceives an
instant reaction, and the expensive thought is hidden behind it.

A reflex is intentionally hard to write: no planning, no arbitration, one
primitive. Anything that needs a choice belongs in a behavior instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..contracts.enums import Priority, RobotState
from ..contracts.events import Event, PersonAppeared


@dataclass(frozen=True)
class Reflex:
    name: str
    trigger: type[Event]
    states: tuple[RobotState, ...]
    primitive: str
    enter_state: RobotState | None = None
    priority: int = Priority.REFLEX
    params: Mapping[str, Any] = field(default_factory=dict)

    def applies(self, event: Event, state: RobotState) -> bool:
        return isinstance(event, self.trigger) and state in self.states


DEFAULT_REFLEXES: tuple[Reflex, ...] = (
    Reflex(
        name="wake_on_person",
        trigger=PersonAppeared,
        states=(RobotState.IDLE,),
        primitive="wake_up",
        enter_state=RobotState.WAKING,
    ),
)
