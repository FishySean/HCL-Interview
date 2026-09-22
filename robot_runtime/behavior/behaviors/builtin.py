"""The behaviors shipped by default.

Two of them can fire on the same event (`greet_visitor` and
`greet_returning_visitor`), which is deliberate: it gives the policy layer a
real arbitration decision to make, and is where a VLM adds value -- choosing
the socially appropriate variant -- without ever being able to invent a motion.

Adding a behavior means adding a class here (or in a new module) and one
`register` call in `register_builtins`. Nothing else in the system changes.
"""

from __future__ import annotations

from ...contracts.enums import Priority, RobotState
from ...contracts.events import PersonAppeared, PersonLeft, PlanStep
from ...contracts.events import BehaviorFinished
from ..registry import Behavior, BehaviorContext, BehaviorRegistry

RETURNING_VISITOR_WINDOW_S = 60.0


class GreetVisitor(Behavior):
    name = "greet_visitor"
    description = "Wake up, greet the person, and wave."
    priority = Priority.NORMAL
    triggers = (PersonAppeared,)
    valid_states = (RobotState.IDLE, RobotState.WAKING, RobotState.DISENGAGING)
    success_state = RobotState.ENGAGED
    failure_state = RobotState.ENGAGED

    greeting = "Hello there! Nice to meet you."

    def plan(self, context: BehaviorContext) -> list[PlanStep]:
        trace = context.event.trace_id if context.event else "-"
        steps: list[PlanStep] = []
        # Normally the reflex arc has already dispatched `wake_up` while the
        # policy was still thinking. If reflexes are disabled, do it here so the
        # behavior remains correct on its own.
        if context.state is not RobotState.WAKING:
            steps.append(
                PlanStep(command=self.command("wake_up", trace), enter_state=RobotState.WAKING)
            )
        steps.append(
            PlanStep(
                command=self.command("say", trace, text=self.greeting),
                enter_state=RobotState.GREETING,
            )
        )
        steps.append(PlanStep(command=self.command("wave", trace, hand="right", cycles=3)))
        return steps


class GreetReturningVisitor(GreetVisitor):
    name = "greet_returning_visitor"
    description = "A warmer greeting for someone the robot saw moments ago."
    priority = Priority.NORMAL
    greeting = "Welcome back! Good to see you again."

    def matches(self, context: BehaviorContext) -> bool:
        event = context.event
        if not isinstance(event, PersonAppeared) or event.track_id is None:
            return False
        last_seen = context.memory.get("last_seen_at")
        last_track = context.memory.get("last_track_id")
        if last_seen is None or last_track != event.track_id:
            return False
        return (context.now - float(last_seen)) <= RETURNING_VISITOR_WINDOW_S


class SayFarewell(Behavior):
    name = "return_to_idle"
    description = "Stop gesturing and settle back into the resting pose."
    # Above NORMAL so it preempts a wave that is still in flight: the person has
    # already gone, and finishing the gesture would only delay the reset.
    priority = Priority.REFLEX
    triggers = (PersonLeft,)
    valid_states = (RobotState.WAKING, RobotState.GREETING, RobotState.ENGAGED)
    success_state = RobotState.IDLE
    failure_state = RobotState.IDLE

    def plan(self, context: BehaviorContext) -> list[PlanStep]:
        trace = context.event.trace_id if context.event else "-"
        return [
            PlanStep(
                command=self.command("lower_arms", trace),
                enter_state=RobotState.DISENGAGING,
                optional=True,
            ),
            PlanStep(command=self.command("relax", trace)),
        ]


class AmbientBreathing(Behavior):
    name = "ambient_breathing"
    description = "Keep a subtle motion alive while engaged so the robot never looks frozen."
    priority = Priority.AMBIENT
    cooldown_s = 2.0
    triggers = (BehaviorFinished,)
    valid_states = (RobotState.ENGAGED,)
    success_state = RobotState.ENGAGED
    failure_state = RobotState.ENGAGED

    def matches(self, context: BehaviorContext) -> bool:
        return context.person_present

    def plan(self, context: BehaviorContext) -> list[PlanStep]:
        trace = context.event.trace_id if context.event else "-"
        return [PlanStep(command=self.command("idle_breathe", trace), optional=True)]


class RecoverFromError(Behavior):
    name = "recover_from_error"
    description = "Reset to a known-good pose after an unrecoverable motion fault."
    priority = Priority.URGENT
    valid_states = (RobotState.ERROR,)
    success_state = RobotState.IDLE
    failure_state = RobotState.IDLE

    def plan(self, context: BehaviorContext) -> list[PlanStep]:
        trace = context.event.trace_id if context.event else "-"
        return [PlanStep(command=self.command("relax", trace), optional=True)]


def register_builtins(registry: BehaviorRegistry | None = None) -> BehaviorRegistry:
    registry = registry or BehaviorRegistry()
    registry.register(GreetVisitor())
    registry.register(GreetReturningVisitor())
    registry.register(SayFarewell())
    registry.register(AmbientBreathing())
    registry.register(RecoverFromError())
    return registry
