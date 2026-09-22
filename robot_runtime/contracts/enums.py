"""Shared vocabulary. This module must not import anything from the layers."""

from __future__ import annotations

from enum import Enum


class RobotState(str, Enum):
    """High-level state of the robot. Owned exclusively by the behavior FSM."""

    IDLE = "idle"
    WAKING = "waking"
    GREETING = "greeting"
    ENGAGED = "engaged"
    DISENGAGING = "disengaging"
    ERROR = "error"


class MotionStatus(str, Enum):
    """Outcome of a motion command. Failure is a value, never an escaping exception."""

    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    PREEMPTED = "preempted"
    REJECTED = "rejected"

    @property
    def is_success(self) -> bool:
        return self is MotionStatus.COMPLETED

    @property
    def is_retryable(self) -> bool:
        """Preemption and rejection are deliberate arbitration outcomes, not faults,
        so retrying them on a fallback primitive would fight the arbiter."""
        return self in (MotionStatus.FAILED, MotionStatus.TIMED_OUT)


class DenialReason(str, Enum):
    """Why the safety gate refused a motion. Distinguished because they are not
    the same kind of event: a limit violation is a bug in a behavior, while a
    busy limb is normal contention."""

    UNKNOWN_PRIMITIVE = "unknown_primitive"
    JOINT_LIMIT = "joint_limit"
    SELF_COLLISION = "self_collision"
    LIMB_BUSY = "limb_busy"


class Priority(int, Enum):
    """Higher preempts lower within a resource group."""

    AMBIENT = 10
    NORMAL = 50
    REFLEX = 70
    URGENT = 90
