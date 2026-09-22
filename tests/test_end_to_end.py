"""The behaviour the brief actually asks for, driven through the real wiring:
a scripted camera feed goes in at one end and motion commands come out the
other, with nothing mocked in between except the hardware itself."""

from __future__ import annotations

from conftest import run

from robot_runtime.behavior.behaviors import register_builtins
from robot_runtime.behavior.registry import Behavior
from robot_runtime.contracts.enums import DenialReason, Priority, RobotState
from robot_runtime.contracts.events import PersonAppeared, PlanStep
from robot_runtime.contracts.primitives import JointTarget, Primitive, default_registry
from robot_runtime.perception.detectors.simulated import ScriptedDetector, Segment
from robot_runtime.presence_filter.filter import PresenceConfig
from robot_runtime.runtime.app import AppConfig, RobotApp
from robot_runtime.runtime.clock import FakeClock

VISIT = [
    Segment(start_s=0.0, present=False),
    Segment(start_s=1.0, present=True),
    Segment(start_s=8.0, present=False),
]

GREETING_PRIMITIVES = ("wake_up", "say", "wave", "nod")


def build(timeline=VISIT, backend=None, behaviors=None, primitives=None, **overrides):
    clock = FakeClock()
    detector = ScriptedDetector(timeline, clock=clock, noise=0.0)
    config = AppConfig(
        fps=20.0,
        presence=PresenceConfig(enter_frames=2, exit_frames=10),
        verbose=False,
        use_vlm=overrides.pop("use_vlm", False),
        **overrides,
    )
    app = RobotApp(
        detector=detector,
        clock=clock,
        config=config,
        sink=lambda _: None,
        backend=backend,
        behaviors=behaviors,
        primitives=primitives,
    )
    return app, clock


def greeting_sequence(app) -> list[str]:
    return [p for p in app.backend.executed if p in GREETING_PRIMITIVES]


def test_the_robot_wakes_greets_and_waves_then_returns_to_idle():
    async def main():
        app, clock = build()
        await app.start()

        await clock.advance(6.0)
        assert greeting_sequence(app) == ["wake_up", "say", "wave"]
        assert app.state is RobotState.ENGAGED

        await clock.advance(8.0)
        assert app.state is RobotState.IDLE

        await app.stop()

    run(main)


def test_nothing_happens_while_nobody_is_there():
    async def main():
        app, clock = build(timeline=[Segment(start_s=0.0, present=False)])
        await app.start()
        await clock.advance(10.0)

        assert app.backend.executed == []
        assert app.state is RobotState.IDLE
        await app.stop()

    run(main)


def test_a_brief_occlusion_does_not_trigger_a_second_greeting():
    timeline = [
        Segment(start_s=0.0, present=False),
        Segment(start_s=1.0, present=True),
        Segment(start_s=3.0, present=False),  # shorter than the exit window
        Segment(start_s=3.3, present=True),
        Segment(start_s=10.0, present=False),
    ]

    async def main():
        app, clock = build(timeline=timeline)
        await app.start()
        await clock.advance(9.0)

        assert app.backend.executed.count("say") == 1
        await app.stop()

    run(main)


def test_the_reflex_moves_before_the_policy_has_decided():
    """The latency-hiding claim, asserted rather than argued: `wake_up` is
    dispatched while a slow policy is still deliberating.

    A second behavior competing for the same event is what forces a real
    arbitration -- with a single candidate the arbiter short-circuits and there
    is no latency to hide.
    """

    class AlternativeGreeting(Behavior):
        name = "alternative_greeting"
        triggers = (PersonAppeared,)
        valid_states = (RobotState.IDLE, RobotState.WAKING)
        success_state = RobotState.ENGAGED
        failure_state = RobotState.ENGAGED

        def plan(self, context):
            return [PlanStep(command=self.command("nod", context.event.trace_id))]

    behaviors = register_builtins()
    behaviors.register(AlternativeGreeting())

    async def main():
        app, clock = build(
            behaviors=behaviors, use_vlm=True, vlm_latency_s=0.2, decision_deadline_s=0.5
        )
        await app.start()
        await clock.advance(1.15)  # presence confirmed, decision still in flight

        reflexes = app.tracer.of_kind("behavior.reflex")
        assert reflexes and reflexes[0].fields["primitive"] == "wake_up"
        assert app.state is RobotState.WAKING
        assert app.tracer.of_kind("behavior.selected") == []

        await clock.advance(0.5)
        selected = app.tracer.of_kind("behavior.selected")
        assert selected
        assert reflexes[0].at < selected[0].at

        await app.stop()

    run(main)


def test_a_failing_wave_degrades_to_a_nod_and_the_interaction_survives():
    from test_motion_executor import FlakyBackend

    async def main():
        clock = FakeClock()
        detector = ScriptedDetector(VISIT, clock=clock, noise=0.0)
        backend = FlakyBackend(clock, fail={"wave"})
        config = AppConfig(
            fps=20.0,
            presence=PresenceConfig(enter_frames=2, exit_frames=10),
            verbose=False,
            use_vlm=False,
        )
        app = RobotApp(
            detector=detector, clock=clock, config=config, sink=lambda _: None, backend=backend
        )
        await app.start()
        await clock.advance(7.0)

        assert "wave" not in backend.executed
        assert "nod" in backend.executed
        assert app.state is RobotState.ENGAGED

        await app.stop()

    run(main)


def test_a_departure_preempts_a_gesture_still_in_flight():
    timeline = [
        Segment(start_s=0.0, present=False),
        Segment(start_s=1.0, present=True),
        Segment(start_s=3.4, present=False),  # leaves while the wave is running
    ]

    async def main():
        app, clock = build(timeline=timeline)
        await app.start()
        await clock.advance(12.0)

        assert "wave" not in app.backend.executed  # cut short, never completed
        assert app.state is RobotState.IDLE
        await app.stop()

    run(main)


def test_one_trace_id_spans_the_whole_pipeline():
    """The optional traceability requirement: a single id reconstructs the chain
    from the frame that confirmed the person all the way to the actuator."""

    async def main():
        app, clock = build()
        await app.start()
        await clock.advance(6.0)

        appeared = [
            r for r in app.tracer.of_kind("presence.event") if r.fields["event"] == "PersonAppeared"
        ]
        assert appeared
        trace_id = appeared[0].trace_id

        kinds = {record.kind for record in app.tracer.chain(trace_id)}
        assert {
            "presence.event",
            "behavior.selected",
            "motion.command",
            "safety.approved",
            "motion.result",
        } <= kinds

        await app.stop()

    run(main)


def test_a_misparameterised_behavior_is_stopped_by_the_gate_not_by_the_hardware():
    """A behavior that asks for an impossible pose must be refused before the
    backend sees it, and the interaction must still terminate cleanly."""

    class OverreachingGreeting(Behavior):
        name = "overreaching_greeting"
        priority = Priority.URGENT
        triggers = (PersonAppeared,)
        valid_states = (RobotState.IDLE, RobotState.WAKING)
        success_state = RobotState.ENGAGED
        failure_state = RobotState.ENGAGED

        def plan(self, context):
            return [
                PlanStep(
                    command=self.command("wave", context.event.trace_id, amplitude=2.0),
                    enter_state=RobotState.GREETING,
                )
            ]

    behaviors = register_builtins()
    behaviors.register(OverreachingGreeting())

    async def main():
        app, clock = build(behaviors=behaviors)
        await app.start()
        await clock.advance(6.0)

        assert "wave" not in app.backend.executed
        denials = app.tracer.of_kind("safety.denied")
        assert any(r.fields["reason"] == DenialReason.JOINT_LIMIT.value for r in denials)
        assert app.state is RobotState.ENGAGED  # degraded, not wedged

        await app.stop()

    run(main)


def test_a_new_behavior_and_a_new_motion_need_no_change_to_the_core():
    """The extensibility constraint, as an executable claim: a third-party
    behavior and primitive are registered from outside the package and run."""

    primitives = default_registry()
    primitives.register(
        Primitive(
            name="bow",
            limbs=("torso",),
            duration_s=0.8,
            timeout_s=2.0,
            targets=(JointTarget("torso_pitch", 0.30),),
        )
    )

    class BowToVisitor(Behavior):
        name = "bow_to_visitor"
        priority = Priority.URGENT
        triggers = (PersonAppeared,)
        valid_states = (RobotState.IDLE, RobotState.WAKING)
        success_state = RobotState.ENGAGED
        failure_state = RobotState.ENGAGED

        def plan(self, context):
            trace = context.event.trace_id
            return [
                PlanStep(
                    command=self.command("bow", trace), enter_state=RobotState.GREETING
                )
            ]

    behaviors = register_builtins()
    behaviors.register(BowToVisitor())

    async def main():
        app, clock = build(behaviors=behaviors, primitives=primitives)
        await app.start()
        await clock.advance(6.0)

        assert "bow" in app.backend.executed
        assert app.state is RobotState.ENGAGED
        await app.stop()

    run(main)
