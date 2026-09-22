"""The gate is the last thing between a decision and an actuator, so its rules
get tested directly rather than only through the pipeline. Everything here is
synchronous: the gate does no I/O and holds no state beyond limb ownership,
which is what makes that possible."""

from __future__ import annotations

import pytest
from conftest import RecordingTracer

from robot_runtime.contracts.enums import DenialReason, Priority
from robot_runtime.contracts.primitives import JointTarget, Primitive, default_registry
from robot_runtime.safety.collision import approximate_self_collision, never_collides
from robot_runtime.safety.gate import SafetyGate
from robot_runtime.safety.model import RobotModel, RobotModelError


def build(collision=approximate_self_collision, registry=None):
    registry = registry or default_registry()
    return SafetyGate(RobotModel.load(), registry, RecordingTracer(), collision_check=collision)


# --------------------------------------------------------------------------- #
# The robot model
# --------------------------------------------------------------------------- #


def test_the_shipped_model_loads_and_is_self_consistent():
    model = RobotModel.load()
    assert model.name == "demo_interactive_robot"
    assert model.limb_of("right_elbow") == "right_arm"
    assert set(model.limbs) == {"voice", "head", "torso", "left_arm", "right_arm"}


def test_every_shipped_primitive_fits_the_shipped_robot():
    """Startup validation: a joint typo should stop the robot from booting."""
    build().validate_registry()


def test_a_joint_claimed_by_two_limbs_is_a_boot_failure():
    with pytest.raises(RobotModelError, match="belongs to both"):
        RobotModel.from_mapping(
            {
                "limbs": {"left_arm": {"joints": ["elbow"]}, "right_arm": {"joints": ["elbow"]}},
                "joints": {"elbow": {"min": 0.0, "max": 1.0}},
            }
        )


def test_an_empty_joint_range_is_a_boot_failure():
    with pytest.raises(RobotModelError, match="empty range"):
        RobotModel.from_mapping(
            {
                "limbs": {"head": {"joints": ["head_yaw"]}},
                "joints": {"head_yaw": {"min": 1.0, "max": 1.0}},
            }
        )


# --------------------------------------------------------------------------- #
# Static checks: joint limits and self-collision
# --------------------------------------------------------------------------- #


def test_a_motion_that_exceeds_a_joint_limit_is_refused():
    gate = build()
    verdict = gate.screen("wave", {"amplitude": 2.0})

    assert not verdict.allowed
    assert verdict.reason is DenialReason.JOINT_LIMIT
    assert "right_shoulder_pitch" in verdict.detail
    assert gate.occupancy() == {}


def test_the_same_motion_inside_the_limits_is_allowed():
    verdict = build().screen("wave", {"amplitude": 0.8})

    assert verdict.allowed
    assert verdict.limbs == ("right_arm",)
    assert verdict.joint_targets["right_shoulder_pitch"] == pytest.approx(1.12)


def test_the_limit_check_follows_the_hand_parameter():
    """`wave(hand=left)` must be checked against the *left* joints. Waving left
    with an amplitude the left shoulder cannot reach is not made legal by the
    right shoulder being able to."""
    gate = build()
    assert gate.screen("wave", {"hand": "left", "amplitude": 0.8}).allowed
    denied = gate.screen("wave", {"hand": "left", "amplitude": 2.0})
    assert denied.reason is DenialReason.JOINT_LIMIT
    assert "left_shoulder_pitch" in denied.detail


def test_a_self_colliding_pose_is_refused_even_within_joint_limits():
    registry = default_registry()
    registry.register(
        Primitive(
            name="hug",
            limbs=("right_arm",),
            duration_s=1.0,
            timeout_s=2.0,
            targets=(
                JointTarget("right_shoulder_roll", 0.0),
                JointTarget("right_elbow", 2.20),
            ),
        )
    )
    gate = build(registry=registry)

    # Both joints are individually legal ...
    assert gate.model.limit_violations({"right_shoulder_roll": 0.0, "right_elbow": 2.20}) == []
    # ... but the combination puts the hand in the torso.
    verdict = gate.screen("hug", {})
    assert verdict.reason is DenialReason.SELF_COLLISION
    assert "right_hand_into_torso" in verdict.detail


def test_an_unknown_primitive_never_gets_a_limb():
    gate = build()
    verdict = gate.screen("perform_a_backflip", {})
    assert verdict.reason is DenialReason.UNKNOWN_PRIMITIVE
    assert gate.occupancy() == {}


def test_a_missing_required_parameter_is_refused_rather_than_crashing():
    registry = default_registry()
    registry.register(
        Primitive(
            name="point",
            limbs=("{hand}_arm",),
            duration_s=0.5,
            timeout_s=1.0,
            targets=(JointTarget("{hand}_elbow", 0.5),),
        )
    )
    verdict = build(registry=registry).screen("point", {})
    assert verdict.reason is DenialReason.UNKNOWN_PRIMITIVE
    assert "hand" in verdict.detail


# --------------------------------------------------------------------------- #
# Dynamic checks: limb ownership and priority
# --------------------------------------------------------------------------- #


def test_a_lower_priority_motion_cannot_interrupt_a_higher_priority_one():
    """The rule that protects an in-progress urgent motion: ambient behavior
    must never be able to take an arm away from a safety-relevant gesture."""
    gate = build(collision=never_collides)
    held = gate.acquire("urgent-1", Priority.URGENT, ("right_arm",), primitive="wave")
    assert held.allowed

    denied = gate.acquire("ambient-1", Priority.AMBIENT, ("right_arm",), primitive="idle_breathe")

    assert not denied.allowed
    assert denied.reason is DenialReason.LIMB_BUSY
    assert "does not outrank" in denied.detail
    assert gate.owner_of("right_arm") == "urgent-1"


def test_an_equal_priority_motion_is_also_refused():
    """Equal rank is contention to resolve upstream, not a race for whoever
    published last."""
    gate = build()
    gate.acquire("first", Priority.NORMAL, ("right_arm",), primitive="wave")
    denied = gate.acquire("second", Priority.NORMAL, ("right_arm",), primitive="wave")

    assert denied.reason is DenialReason.LIMB_BUSY
    assert gate.owner_of("right_arm") == "first"


def test_a_higher_priority_motion_takes_the_limb_and_names_its_victim():
    gate = build()
    gate.acquire("ambient-1", Priority.AMBIENT, ("right_arm",), primitive="idle_breathe")
    taken = gate.acquire("reflex-1", Priority.REFLEX, ("right_arm",), primitive="lower_arms")

    assert taken.allowed
    assert taken.preempted == ("ambient-1",)
    assert gate.owner_of("right_arm") == "reflex-1"


def test_a_multi_limb_motion_is_all_or_nothing():
    gate = build()
    gate.acquire("urgent-1", Priority.URGENT, ("right_arm",), primitive="wave")
    denied = gate.acquire("normal-1", Priority.NORMAL, ("left_arm", "right_arm"), primitive="lower_arms")

    assert denied.reason is DenialReason.LIMB_BUSY
    assert gate.owner_of("left_arm") is None  # the free arm was not half-taken


def test_a_preempted_motion_cannot_release_its_successors_limb():
    """The release path is where a use-after-preempt bug would live: the victim
    finishes *after* the preemptor already owns the limb."""
    gate = build()
    gate.acquire("victim", Priority.AMBIENT, ("right_arm",), primitive="idle_breathe")
    gate.acquire("winner", Priority.REFLEX, ("right_arm",), primitive="lower_arms")

    gate.release("victim")

    assert gate.owner_of("right_arm") == "winner"


def test_releasing_frees_every_limb_the_command_held():
    gate = build()
    gate.acquire("both", Priority.NORMAL, ("left_arm", "right_arm"), primitive="lower_arms")
    gate.release("both")
    assert gate.occupancy() == {}


def test_denials_are_counted_by_reason():
    """Denial reasons are kept apart because they mean different things: a limit
    violation is a misparameterised behavior, a busy limb is normal contention."""
    gate = build()
    gate.screen("wave", {"amplitude": 2.0})
    gate.screen("perform_a_backflip", {})

    assert gate.stats.denied[DenialReason.JOINT_LIMIT] == 1
    assert gate.stats.denied[DenialReason.UNKNOWN_PRIMITIVE] == 1
