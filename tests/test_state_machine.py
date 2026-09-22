"""The transition table is data, so it can be asserted on exhaustively."""

from __future__ import annotations

import pytest

from robot_runtime.behavior.state import ALLOWED, IllegalTransition, RobotStateMachine
from robot_runtime.contracts.enums import RobotState


def test_every_state_has_an_entry_and_only_names_real_states():
    assert set(ALLOWED) == set(RobotState)
    for targets in ALLOWED.values():
        assert targets <= set(RobotState)


def test_every_state_can_reach_idle():
    """No state may be a dead end: a robot that cannot get back to idle is stuck."""
    reachable = {RobotState.IDLE}
    changed = True
    while changed:
        changed = False
        for state, targets in ALLOWED.items():
            if state not in reachable and targets & reachable:
                reachable.add(state)
                changed = True
    assert reachable == set(RobotState)


def test_the_happy_path_is_legal():
    fsm = RobotStateMachine()
    for target in (
        RobotState.WAKING,
        RobotState.GREETING,
        RobotState.ENGAGED,
        RobotState.DISENGAGING,
        RobotState.IDLE,
    ):
        fsm.transition(target, reason="test")
    assert fsm.state is RobotState.IDLE
    assert len(fsm.history) == 5


def test_illegal_transitions_are_refused_and_leave_state_untouched():
    fsm = RobotStateMachine()
    with pytest.raises(IllegalTransition):
        fsm.transition(RobotState.GREETING)
    assert fsm.state is RobotState.IDLE


def test_a_visitor_who_returns_mid_farewell_can_be_greeted_again():
    fsm = RobotStateMachine(RobotState.DISENGAGING)
    fsm.transition(RobotState.WAKING, reason="came back")
    assert fsm.state is RobotState.WAKING


def test_force_is_audited():
    fsm = RobotStateMachine(RobotState.GREETING)
    fsm.force(RobotState.IDLE, reason="estop")
    assert fsm.state is RobotState.IDLE
    assert fsm.history[-1][2] == "forced:estop"


def test_transition_carries_trace_and_time():
    fsm = RobotStateMachine()
    change = fsm.transition(RobotState.WAKING, reason="r", trace_id="abc", now=12.5)
    assert (change.trace_id, change.timestamp, change.old, change.new) == (
        "abc",
        12.5,
        RobotState.IDLE,
        RobotState.WAKING,
    )
