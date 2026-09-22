"""The safety gate: the last thing between a decision and an actuator.

It answers two different kinds of question, and the distinction matters.

*Static* questions are properties of the command alone -- does this primitive
exist, do its goal angles fall inside the joint limits, does the goal pose put
the robot inside itself. They can be answered the moment the command appears,
which is why `SafetyGateService` runs them as a pipeline stage and a bad
command never reaches the motion layer at all.

*Dynamic* questions are properties of the moment -- is this limb already
moving, and does the newcomer outrank whatever owns it. The answer changes
between asking and acting, so it must be decided atomically with starting the
motion. `acquire` is therefore called by the executor rather than by the
pipeline stage, but the ownership table and the arbitration policy still live
here, so the gate remains the single source of truth.

The gate holds no robot state beyond limb ownership and does no I/O, which
makes every rule in it testable without an event loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..contracts.enums import DenialReason
from ..contracts.primitives import PrimitiveRegistry
from ..contracts.protocols import Tracer
from ..contracts.safety import Acquisition, Denial, Screened
from .collision import CollisionCheck, approximate_self_collision
from .model import RobotModel, RobotModelError


@dataclass
class _Owner:
    command_id: str
    priority: int
    primitive: str


@dataclass
class GateStats:
    approved: int = 0
    denied: dict[DenialReason, int] = field(default_factory=dict)

    def note_denial(self, reason: DenialReason) -> None:
        self.denied[reason] = self.denied.get(reason, 0) + 1


class SafetyGate:
    def __init__(
        self,
        model: RobotModel,
        registry: PrimitiveRegistry,
        tracer: Tracer,
        collision_check: CollisionCheck = approximate_self_collision,
    ) -> None:
        self._model = model
        self._registry = registry
        self._tracer = tracer
        self._collision_check = collision_check
        self._owners: dict[str, _Owner] = {}
        self.stats = GateStats()

    @property
    def model(self) -> RobotModel:
        return self._model

    # ---------------------------------------------------------------- static
    def screen(
        self, primitive_name: str, params: Mapping[str, Any], trace_id: str = "-"
    ) -> Screened | Denial:
        primitive = self._registry.get(primitive_name)
        if primitive is None:
            return self._deny(DenialReason.UNKNOWN_PRIMITIVE, f"no primitive {primitive_name!r}", trace_id)

        resolved = primitive.with_defaults(params)
        try:
            limbs = primitive.limbs_for(resolved)
            targets = primitive.joint_targets(resolved)
        except KeyError as exc:
            return self._deny(
                DenialReason.UNKNOWN_PRIMITIVE,
                f"{primitive_name}: missing parameter {exc.args[0]!r}",
                trace_id,
            )

        unknown_limbs = [limb for limb in limbs if limb not in self._model.limbs]
        if unknown_limbs:
            return self._deny(
                DenialReason.UNKNOWN_PRIMITIVE,
                f"{primitive_name}: limb(s) {unknown_limbs} are not in the robot model",
                trace_id,
            )

        violations = self._model.limit_violations(targets)
        if violations:
            return self._deny(DenialReason.JOINT_LIMIT, "; ".join(violations), trace_id)

        collision = self._collision_check(targets, self._model)
        if collision is not None:
            return self._deny(DenialReason.SELF_COLLISION, collision, trace_id)

        return Screened(primitive=primitive, params=resolved, limbs=limbs, joint_targets=targets)

    # --------------------------------------------------------------- dynamic
    def acquire(
        self, command_id: str, priority: int, limbs: tuple[str, ...], primitive: str = "", trace_id: str = "-"
    ) -> Acquisition | Denial:
        """All-or-nothing: a motion that needs both arms takes both or neither.

        Strictly greater priority preempts. Equal priority is rejected, because
        two behaviors of the same rank fighting over a limb is contention to be
        resolved upstream, not a race for whoever published last.
        """
        blocking = [
            owner
            for limb in limbs
            if (owner := self._owners.get(limb)) is not None and owner.command_id != command_id
        ]
        outranked = [owner for owner in blocking if priority > owner.priority]
        if len(outranked) != len(blocking):
            held_by = ", ".join(
                f"{owner.primitive}@{owner.priority}" for owner in blocking if owner not in outranked
            )
            return self._deny(
                DenialReason.LIMB_BUSY,
                f"{list(limbs)} held by {held_by}; {primitive or 'command'}@{priority} does not outrank it",
                trace_id,
            )

        preempted = tuple(dict.fromkeys(owner.command_id for owner in outranked))
        for limb in limbs:
            self._owners[limb] = _Owner(command_id=command_id, priority=priority, primitive=primitive)
        self.stats.approved += 1
        if preempted:
            self._tracer.record("safety.preempt", trace_id, primitive=primitive, preempted=list(preempted))
        return Acquisition(limbs=limbs, preempted=preempted)

    def release(self, command_id: str) -> None:
        """Only release what this command still owns: a preempted motion must
        not free a limb that its preemptor has already taken."""
        for limb, owner in list(self._owners.items()):
            if owner.command_id == command_id:
                del self._owners[limb]

    def owner_of(self, limb: str) -> str | None:
        owner = self._owners.get(limb)
        return owner.command_id if owner else None

    def is_busy(self, limb: str) -> bool:
        return limb in self._owners

    def occupancy(self) -> dict[str, str]:
        return {limb: owner.primitive for limb, owner in self._owners.items()}

    # ------------------------------------------------------------ validation
    def validate_registry(self) -> None:
        """Startup check that every registered primitive is expressible on this
        robot at its default parameters. A typo in a joint name should stop the
        robot from booting, not surface as a denial mid-greeting."""
        for name in self._registry.names():
            verdict = self.screen(name, {}, trace_id="startup")
            if not verdict.allowed and verdict.reason is not DenialReason.LIMB_BUSY:
                raise RobotModelError(
                    f"primitive {name!r} is invalid for robot {self._model.name!r}: {verdict.detail}"
                )

    # ---------------------------------------------------------------- helper
    def _deny(self, reason: DenialReason, detail: str, trace_id: str) -> Denial:
        self.stats.note_denial(reason)
        self._tracer.record("safety.denied", trace_id, reason=reason.value, detail=detail)
        return Denial(reason=reason, detail=detail)
