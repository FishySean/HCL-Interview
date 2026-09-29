"""Composition root.

Every dependency is constructed here and injected downwards, which is why no
layer ever imports another. Swapping the scripted detector for a webcam, the
simulated actuators for real ones, the hand-written self-collision stub for a
geometric one, or the mock VLM for a real endpoint is a change to this file
alone.

The pipeline, in order:

    detector -> perception -> presence_filter -> behavior -> safety -> motion
                                    ^                                    |
                                    +------------ MotionResult ----------+
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TextIO

from ..behavior.behaviors import register_builtins
from ..behavior.policies import DeadlineArbiter, MockVLMPolicy, RulePolicy
from ..behavior.registry import BehaviorRegistry
from ..behavior.service import BehaviorService
from ..behavior.vlm import RealVLMPolicy, VLMConfig, VLMConfigError
from ..contracts.enums import RobotState
from ..contracts.events import (
    BehaviorSelected,
    MotionDenied,
    MotionResult,
    PerceptionEvent,
    StateChanged,
)
from ..contracts.primitives import PrimitiveRegistry, default_registry
from ..contracts.protocols import Clock, PersonDetector, Policy
from ..motion.backends.simulated import SimulatedBackend
from ..motion.executor import MotionExecutor
from ..motion.service import MotionService
from ..perception.service import PerceptionService
from ..presence_filter.filter import PresenceConfig
from ..presence_filter.service import PresenceFilterService
from ..safety.collision import CollisionCheck, approximate_self_collision
from ..safety.gate import SafetyGate
from ..safety.model import RobotModel
from ..safety.service import SafetyGateService
from .bus import EventBus
from .clock import RealClock
from .tracing import Tracer


@dataclass
class AppConfig:
    fps: float = 20.0
    frame_hz: float = 1.0
    presence: PresenceConfig = field(default_factory=PresenceConfig)
    robot_config: Path | str | None = None
    motion_failure_rate: float = 0.0
    motion_slow_rate: float = 0.0
    use_vlm: bool = True
    # "fake" is the default everywhere, including tests: a hosted model is a
    # network dependency, and the robot's logic must be reproducible without one.
    vlm_policy: str = "fake"
    vlm_latency_s: float = 0.12
    vlm_invalid_rate: float = 0.0
    vlm_error_rate: float = 0.0
    vlm_timeout_s: float = 2.0
    decision_deadline_s: float = 0.25
    seed: int = 7
    verbose: bool = True


class RobotApp:
    def __init__(
        self,
        detector: PersonDetector,
        clock: Clock | None = None,
        config: AppConfig | None = None,
        trace_stream: TextIO | None = None,
        sink: Callable[[str], None] = print,
        behaviors: BehaviorRegistry | None = None,
        primitives: PrimitiveRegistry | None = None,
        backend: Any | None = None,
        model: RobotModel | None = None,
        collision_check: CollisionCheck = approximate_self_collision,
    ) -> None:
        self.config = config or AppConfig()
        self.clock = clock or RealClock()
        self.tracer = Tracer(self.clock, stream=trace_stream)
        self.bus = EventBus(tracer=None)
        self._sink = sink

        # ------------------------------------------------------------ safety
        self.primitives = primitives or default_registry()
        self.model = model or RobotModel.load(self.config.robot_config)
        self.gate = SafetyGate(
            model=self.model,
            registry=self.primitives,
            tracer=self.tracer,
            collision_check=collision_check,
        )
        # Fail at boot, not mid-greeting, if a primitive does not fit this robot.
        self.gate.validate_registry()
        self.safety = SafetyGateService(self.gate, self.bus, self.clock, self.tracer)

        # ------------------------------------------------------------ motion
        self.backend = backend or SimulatedBackend(
            clock=self.clock,
            failure_rate=self.config.motion_failure_rate,
            slow_rate=self.config.motion_slow_rate,
            seed=self.config.seed,
            sink=sink if self.config.verbose else None,
        )
        self.executor = MotionExecutor(
            self.backend, self.gate, self.primitives, self.clock, self.tracer
        )
        self.motion = MotionService(self.executor, self.bus, self.tracer)

        # ---------------------------------------------------------- behavior
        self.policy = self._build_policy()
        self.arbiter = DeadlineArbiter(
            primary=self.policy,
            fallback=RulePolicy(),
            clock=self.clock,
            tracer=self.tracer,
            deadline_s=self.config.decision_deadline_s,
        )
        self.behavior = BehaviorService(
            bus=self.bus,
            clock=self.clock,
            tracer=self.tracer,
            registry=behaviors or register_builtins(),
            arbiter=self.arbiter,
        )

        # ------------------------------------------------ sensing front end
        self.presence = PresenceFilterService(
            bus=self.bus, clock=self.clock, tracer=self.tracer, config=self.config.presence
        )
        self.perception = PerceptionService(
            detector=detector,
            bus=self.bus,
            clock=self.clock,
            tracer=self.tracer,
            fps=self.config.fps,
            frame_hz=self.config.frame_hz,
        )

        self._console_task: asyncio.Task[None] | None = None

    def _build_policy(self) -> Policy:
        """Pick the deliberative policy, and never let that choice stop the robot.

        A missing credential is a deployment mistake, not a reason for a robot
        to refuse to boot. It degrades to the simulated policy and says so, in
        exactly the same way a slow or wrong model degrades to `RulePolicy` at
        runtime: the interesting policy is always optional.
        """
        if not self.config.use_vlm:
            return RulePolicy()

        if self.config.vlm_policy == "real":
            try:
                vlm_config = VLMConfig.from_env(timeout_s=self.config.vlm_timeout_s)
                policy = RealVLMPolicy(vlm_config, tracer=self.tracer)
            except (VLMConfigError, NotImplementedError) as exc:
                self._sink(f"\n[vlm] {exc}")
                self._sink("[vlm] Falling back to the simulated policy for this run.\n")
            else:
                self._sink(
                    f"\n[vlm] Using {vlm_config.provider}/{vlm_config.model} "
                    f"(deadline {self.config.decision_deadline_s:g}s, "
                    f"timeout {vlm_config.timeout_s:g}s)\n"
                )
                return policy

        return MockVLMPolicy(
            clock=self.clock,
            latency_s=self.config.vlm_latency_s,
            invalid_rate=self.config.vlm_invalid_rate,
            error_rate=self.config.vlm_error_rate,
            seed=self.config.seed,
        )

    @property
    def state(self) -> RobotState:
        return self.behavior.state

    async def start(self) -> None:
        if self.config.verbose:
            self._console_task = asyncio.create_task(self._console(), name="console")
        # Downstream first, so no stage is ever publishing into a dead end.
        await self.motion.start()
        await self.safety.start()
        await self.behavior.start()
        await self.presence.start()
        await self.perception.start()

    async def stop(self) -> None:
        await self.perception.stop()
        await self.presence.stop()
        await self.behavior.stop()
        await self.safety.stop()
        await self.motion.stop()
        if self._console_task is not None:
            self._console_task.cancel()
            await asyncio.gather(self._console_task, return_exceptions=True)
            self._console_task = None

    async def _console(self) -> None:
        subscription = self.bus.subscribe(
            PerceptionEvent, StateChanged, BehaviorSelected, MotionResult, MotionDenied, name="console"
        )
        while True:
            event = await subscription.get()
            self._sink(_render(event))


def _render(event: object) -> str:
    stamp = getattr(event, "timestamp", 0.0)
    head = f"[{stamp:6.2f}s]"
    if isinstance(event, StateChanged):
        return f"{head} STATE  {event.old.value} -> {event.new.value}   ({event.reason})"
    if isinstance(event, BehaviorSelected):
        note = f" fallback: {event.fallback_reason}" if event.fallback_reason else ""
        return (
            f"{head} DECIDE {event.behavior}  via {event.policy} "
            f"in {event.latency_ms:.0f}ms  considered={list(event.considered)}{note}"
        )
    if isinstance(event, MotionDenied):
        return f"{head} SAFETY denied {event.primitive}: {event.reason.value}  ({event.detail})"
    if isinstance(event, MotionResult):
        chain = " -> ".join(event.attempts) if len(event.attempts) > 1 else event.primitive
        return f"{head} MOTION {chain}: {event.status.value}  ({event.detail})"
    if isinstance(event, PerceptionEvent):
        return f"{head} SENSE  {type(event).__name__}  trace={event.trace_id}"
    return f"{head} {event!r}"
