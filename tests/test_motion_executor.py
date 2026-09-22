"""The executor is where the brief's hardest requirement lands: motion "may
occasionally fail or take longer than expected". These tests assert that each
of those outcomes is a value the behavior layer can react to, that the fallback
ladder runs and is itself safety-screened, and that a limb is released after
every possible outcome."""

from __future__ import annotations

import asyncio

from conftest import RecordingTracer, run

from robot_runtime.contracts.enums import MotionStatus, Priority
from robot_runtime.contracts.events import ApprovedMotion, MotionCommand
from robot_runtime.contracts.primitives import JointTarget, Primitive, default_registry
from robot_runtime.motion.backends.simulated import MotionFailure
from robot_runtime.motion.executor import MotionExecutor
from robot_runtime.runtime.clock import FakeClock
from robot_runtime.safety.collision import never_collides
from robot_runtime.safety.gate import SafetyGate
from robot_runtime.safety.model import RobotModel


class FlakyBackend:
    """Explicit about which primitive misbehaves, so no test depends on an RNG."""

    name = "flaky"

    def __init__(self, clock, fail: set[str] | None = None, hang: set[str] | None = None) -> None:
        self._clock = clock
        self._fail = fail or set()
        self._hang = hang or set()
        self.executed: list[str] = []

    async def run(self, primitive: str, params, duration_s: float) -> str:
        if primitive in self._hang:
            await self._clock.sleep(10_000.0)
        if primitive in self._fail:
            await self._clock.sleep(duration_s * 0.2)
            raise MotionFailure(f"{primitive}: stalled servo")
        await self._clock.sleep(duration_s)
        self.executed.append(primitive)
        return "ok"


def make(fail=None, hang=None, collision=never_collides):
    clock = FakeClock()
    registry = default_registry()
    gate = SafetyGate(RobotModel.load(), registry, RecordingTracer(), collision_check=collision)
    backend = FlakyBackend(clock, fail=fail, hang=hang)
    executor = MotionExecutor(backend, gate, registry, clock, RecordingTracer())
    return executor, clock, backend, gate


def command(primitive: str, priority: int = Priority.NORMAL, **params) -> MotionCommand:
    return MotionCommand(primitive=primitive, params=params, priority=priority)


def approve(gate: SafetyGate, cmd: MotionCommand) -> ApprovedMotion:
    """Stands in for the safety gate service, which is what produces these in
    the running system."""
    screened = gate.screen(cmd.primitive, cmd.params, cmd.trace_id)
    assert screened.allowed, getattr(screened, "detail", "")
    return ApprovedMotion(
        trace_id=cmd.trace_id,
        command=cmd,
        limbs=screened.limbs,
        joint_targets=dict(screened.joint_targets),
    )


def test_a_healthy_motion_completes():
    async def main():
        executor, clock, backend, gate = make()
        task = asyncio.create_task(executor.execute(approve(gate, command("wave"))))
        await clock.advance(5.0)
        result = await task
        assert result.status is MotionStatus.COMPLETED
        assert result.limbs == ("right_arm",)
        assert backend.executed == ["wave"]

    run(main)


def test_the_hand_parameter_selects_the_limb():
    async def main():
        executor, clock, _, gate = make()
        task = asyncio.create_task(executor.execute(approve(gate, command("wave", hand="left"))))
        await clock.advance(5.0)
        assert (await task).limbs == ("left_arm",)

    run(main)


def test_a_failed_wave_degrades_to_a_nod():
    """Failing to move must not mean failing to interact."""

    async def main():
        executor, clock, backend, gate = make(fail={"wave"})
        task = asyncio.create_task(executor.execute(approve(gate, command("wave"))))
        await clock.advance(5.0)
        result = await task
        assert result.status is MotionStatus.COMPLETED
        assert result.primitive == "nod"
        assert result.attempts == ("wave", "nod")
        assert backend.executed == ["nod"]

    run(main)


def test_a_substituted_primitive_is_safety_screened_too():
    """The invariant: nothing reaches the backend without a gate verdict, and a
    fallback is not exempt just because the original was approved.

    Parameters carry over to the substitute, and the substitute moves different
    joints -- so an amplitude that is harmless for the original can be out of
    range for its fallback.
    """
    registry = default_registry()
    registry.register(
        Primitive(
            name="deep_bow",
            limbs=("torso",),
            duration_s=1.0,
            timeout_s=2.0,
            targets=(JointTarget("torso_pitch", 0.30, scales_with="amplitude"),),
            defaults={"amplitude": 1.0},
        )
    )
    registry.register(
        Primitive(
            name="flourish",
            limbs=("{hand}_arm",),
            duration_s=1.0,
            timeout_s=2.0,
            fallbacks=("deep_bow",),
            targets=(JointTarget("{hand}_elbow", 0.50, scales_with="amplitude"),),
            defaults={"hand": "right", "amplitude": 1.0},
        )
    )

    async def main():
        clock = FakeClock()
        gate = SafetyGate(RobotModel.load(), registry, RecordingTracer(), collision_check=never_collides)
        backend = FlakyBackend(clock, fail={"flourish"})
        executor = MotionExecutor(backend, gate, registry, clock, RecordingTracer())

        # elbow 1.00 rad is fine; torso_pitch 0.60 rad is past the 0.35 limit.
        cmd = command("flourish", amplitude=2.0)
        task = asyncio.create_task(executor.execute(approve(gate, cmd)))
        await clock.advance(5.0)
        result = await task

        assert result.status is MotionStatus.REJECTED
        assert result.attempts == ("flourish", "deep_bow")
        assert "joint_limit" in result.detail
        assert backend.executed == []
        assert gate.occupancy() == {}

    run(main)


def test_a_hung_motion_times_out_instead_of_freezing_the_robot():
    async def main():
        executor, clock, backend, gate = make(hang={"wave", "nod"})
        task = asyncio.create_task(executor.execute(approve(gate, command("wave"))))
        await clock.advance(30.0)
        result = await task
        assert result.status is MotionStatus.TIMED_OUT
        assert result.attempts == ("wave", "nod")  # the ladder was walked, then exhausted
        assert backend.executed == []

    run(main)


def test_a_higher_priority_command_preempts_the_same_limb():
    async def main():
        executor, clock, backend, gate = make()
        slow = asyncio.create_task(executor.execute(approve(gate, command("wave"))))
        await clock.advance(0.2)
        urgent = asyncio.create_task(
            executor.execute(approve(gate, command("lower_arms", priority=Priority.REFLEX)))
        )
        await clock.advance(5.0)

        assert (await slow).status is MotionStatus.PREEMPTED
        assert (await urgent).status is MotionStatus.COMPLETED
        assert backend.executed == ["lower_arms"]

    run(main)


def test_preemption_is_not_retried_on_the_fallback_ladder():
    """Preemption is the arbiter working as intended; retrying would fight it."""

    async def main():
        executor, clock, _, gate = make()
        slow = asyncio.create_task(executor.execute(approve(gate, command("wave"))))
        await clock.advance(0.2)
        urgent = asyncio.create_task(
            executor.execute(approve(gate, command("lower_arms", priority=Priority.REFLEX)))
        )
        await clock.advance(5.0)

        result = await slow
        await urgent
        assert result.attempts == ("wave",)

    run(main)


def test_an_equal_priority_command_is_rejected_not_queued():
    async def main():
        executor, clock, _, gate = make()
        first = asyncio.create_task(executor.execute(approve(gate, command("wave"))))
        await clock.advance(0.2)
        second = asyncio.create_task(executor.execute(approve(gate, command("wave"))))
        await clock.advance(5.0)

        assert (await first).status is MotionStatus.COMPLETED
        rejected = await second
        assert rejected.status is MotionStatus.REJECTED
        assert "limb_busy" in rejected.detail

    run(main)


def test_different_limbs_run_at_the_same_time():
    """Speaking while waving must stay possible; the gate arbitrates per limb."""

    async def main():
        executor, clock, backend, gate = make()
        wave = asyncio.create_task(executor.execute(approve(gate, command("wave"))))
        speak = asyncio.create_task(executor.execute(approve(gate, command("say", text="Hi"))))
        await clock.advance(5.0)

        assert (await wave).status is MotionStatus.COMPLETED
        assert (await speak).status is MotionStatus.COMPLETED
        assert sorted(backend.executed) == ["say", "wave"]

    run(main)


def test_both_arms_are_taken_together_or_not_at_all():
    """`lower_arms` needs two limbs; a wave already holding one must block it
    when it does not outrank the incumbent."""

    async def main():
        executor, clock, _, gate = make()
        wave = asyncio.create_task(
            executor.execute(approve(gate, command("wave", priority=Priority.URGENT)))
        )
        await clock.advance(0.2)
        assert gate.owner_of("right_arm") is not None
        assert gate.owner_of("left_arm") is None

        both = asyncio.create_task(
            executor.execute(approve(gate, command("lower_arms", priority=Priority.NORMAL)))
        )
        await clock.advance(5.0)

        assert (await both).status is MotionStatus.REJECTED
        assert (await wave).status is MotionStatus.COMPLETED
        # The free arm was not left locked by the failed all-or-nothing attempt.
        assert gate.owner_of("left_arm") is None

    run(main)


def test_the_limb_is_released_after_every_outcome():
    async def main():
        executor, clock, _, gate = make(fail={"wave", "nod"})
        task = asyncio.create_task(executor.execute(approve(gate, command("wave"))))
        await clock.advance(5.0)
        await task
        assert gate.occupancy() == {}

    run(main)


def test_the_result_carries_the_trace_id_of_the_command():
    async def main():
        executor, clock, _, gate = make()
        cmd = MotionCommand(trace_id="abc123", primitive="nod")
        task = asyncio.create_task(executor.execute(approve(gate, cmd)))
        await clock.advance(5.0)
        assert (await task).trace_id == "abc123"

    run(main)
