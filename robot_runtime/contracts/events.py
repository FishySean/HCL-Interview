"""Every message that crosses a layer boundary is defined here, and only here.

Layers depend on this module; they never depend on each other. That single rule is
what makes the decoupling verifiable instead of aspirational.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from .enums import DenialReason, MotionStatus, Priority, RobotState


def new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass(frozen=True, kw_only=True)
class Event:
    """Base for anything published on the bus."""

    trace_id: str = field(default_factory=new_id)
    timestamp: float = 0.0


# --------------------------------------------------------------------------- #
# Perception
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, kw_only=True)
class PersonObservation:
    """A single raw frame result: noisy, high-frequency, and never actionable on
    its own. Only the presence filter is allowed to consume these."""

    present: bool
    confidence: float = 0.0
    bbox: tuple[float, float, float, float] | None = None
    track_id: int | None = None
    source: str = "unknown"


@dataclass(frozen=True, kw_only=True)
class PersonObserved(Event):
    """Raw output of the perception layer, addressed to the presence filter."""

    observation: PersonObservation


@dataclass(frozen=True, kw_only=True)
class PerceptionEvent(Event):
    """Debounced, actionable facts. Produced by the presence filter, not by the
    perception layer: the distinction is what stops a flickering detector from
    reaching the behavior layer at all."""


@dataclass(frozen=True, kw_only=True)
class PersonAppeared(PerceptionEvent):
    track_id: int | None = None
    confidence: float = 0.0


@dataclass(frozen=True, kw_only=True)
class PersonLeft(PerceptionEvent):
    track_id: int | None = None
    absent_for_s: float = 0.0


@dataclass(frozen=True, kw_only=True)
class CameraFrame(Event):
    """A compressed keyframe for the slow, deliberative tier.

    Deliberately *not* a `PerceptionEvent`: it is context for a decision, never
    a trigger for one, and nothing in the system is allowed to act because a
    frame arrived. It is published at a low rate rather than per detection
    frame -- this exists so a policy can look at the person, not so the bus
    carries video.
    """

    jpeg: bytes
    width: int = 0
    height: int = 0
    source: str = "unknown"


@dataclass(frozen=True, kw_only=True)
class SceneContext(PerceptionEvent):
    """Output of the slow perception tier (VLM). Enriches decisions but is never
    required for correctness: the robot behaves properly if this never arrives."""

    description: str = ""
    tags: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Motion
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, kw_only=True)
class MotionCommand(Event):
    command_id: str = field(default_factory=new_id)
    primitive: str
    params: Mapping[str, Any] = field(default_factory=dict)
    priority: int = Priority.NORMAL
    timeout_s: float | None = None
    issued_by: str = "unknown"


@dataclass(frozen=True, kw_only=True)
class ApprovedMotion(Event):
    """A command that has passed the safety gate.

    The motion layer subscribes to this and never to `MotionCommand`, so there
    is no code path from a behavior to an actuator that skips the gate.
    """

    command: MotionCommand
    limbs: tuple[str, ...]
    joint_targets: Mapping[str, float] = field(default_factory=dict)
    preempted_command_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class MotionDenied(Event):
    """Published alongside the rejection so denials are observable in their own
    right -- a spike in JOINT_LIMIT denials means a behavior is misparameterised,
    which a plain REJECTED result would bury."""

    command_id: str
    primitive: str
    reason: DenialReason
    detail: str = ""


@dataclass(frozen=True, kw_only=True)
class MotionResult(Event):
    command_id: str
    primitive: str
    status: MotionStatus
    limbs: tuple[str, ...] = ()
    duration_s: float = 0.0
    attempts: tuple[str, ...] = ()
    detail: str = ""


# --------------------------------------------------------------------------- #
# Behavior
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, kw_only=True)
class PlanStep:
    """A plan step pairs a motion command with the state the robot enters while it
    runs. The motion layer only ever receives `command`, so it stays ignorant of
    robot state -- the separation of concerns survives the convenience."""

    command: MotionCommand
    enter_state: RobotState | None = None
    optional: bool = False


@dataclass(frozen=True, kw_only=True)
class BehaviorSelected(Event):
    behavior: str
    policy: str
    state_before: RobotState
    latency_ms: float = 0.0
    fallback_reason: str = ""
    considered: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class BehaviorFinished(Event):
    behavior: str
    succeeded: bool
    state_after: RobotState
    detail: str = ""


@dataclass(frozen=True, kw_only=True)
class StateChanged(Event):
    old: RobotState
    new: RobotState
    reason: str = ""
