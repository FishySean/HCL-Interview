"""The state machine: the spine of the behavior layer.

The transition table is the single source of truth for what the robot is
allowed to do, and it is data rather than control flow, so it can be reviewed
by a person and asserted on exhaustively by a test. Behaviors are plugins that
*request* transitions; the machine is what refuses illegal ones. That division
is what keeps an extensible behavior set from turning into an unpredictable
robot -- including when the behavior was chosen by a language model.
"""

from __future__ import annotations

from ..contracts.enums import RobotState
from ..contracts.events import StateChanged

ALLOWED: dict[RobotState, frozenset[RobotState]] = {
    RobotState.IDLE: frozenset({RobotState.WAKING, RobotState.ERROR}),
    RobotState.WAKING: frozenset(
        {RobotState.GREETING, RobotState.ENGAGED, RobotState.DISENGAGING, RobotState.IDLE, RobotState.ERROR}
    ),
    RobotState.GREETING: frozenset({RobotState.ENGAGED, RobotState.DISENGAGING, RobotState.ERROR}),
    RobotState.ENGAGED: frozenset({RobotState.ENGAGED, RobotState.DISENGAGING, RobotState.ERROR}),
    # A visitor who comes back mid-farewell must be re-greetable, not stranded.
    RobotState.DISENGAGING: frozenset({RobotState.IDLE, RobotState.WAKING, RobotState.ERROR}),
    RobotState.ERROR: frozenset({RobotState.IDLE}),
}


class IllegalTransition(RuntimeError):
    pass


class RobotStateMachine:
    def __init__(self, state: RobotState = RobotState.IDLE) -> None:
        self._state = state
        self.history: list[tuple[RobotState, RobotState, str]] = []

    @property
    def state(self) -> RobotState:
        return self._state

    def can(self, target: RobotState) -> bool:
        return target in ALLOWED[self._state]

    def transition(
        self, target: RobotState, reason: str = "", trace_id: str = "-", now: float = 0.0
    ) -> StateChanged:
        if not self.can(target):
            raise IllegalTransition(f"{self._state.value} -> {target.value} is not allowed")
        old, self._state = self._state, target
        self.history.append((old, target, reason))
        return StateChanged(trace_id=trace_id, timestamp=now, old=old, new=target, reason=reason)

    def force(self, target: RobotState, reason: str = "") -> StateChanged:
        """Escape hatch for recovery paths only; always audited in `history`."""
        old, self._state = self._state, target
        self.history.append((old, target, f"forced:{reason}"))
        return StateChanged(old=old, new=target, reason=f"forced:{reason}")
