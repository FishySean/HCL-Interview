"""The motion primitive catalogue: the extension point for "more motions".

This lives in `contracts` rather than in `motion` because three layers need it
and for different reasons: behavior *names* primitives, safety *validates*
them, and motion *executes* them. It is shared vocabulary, so putting it
anywhere else would force two layers to import a third.

A primitive is a name plus parameters -- `wave(hand, amplitude, speed)` -- and
declares which limbs it occupies, how long it takes, when to give up, what to
do instead if it fails, and the goal joint angles its parameters map to. It
does not contain a trajectory. The coordination pattern between joints can be
learned offline from demonstration data, but it is frozen here as a
parameterisation rather than generated at run time: a parameterised primitive
can be range-checked against the robot model before a single servo moves, and a
generated trajectory cannot.

`fallbacks` is the graceful-degradation ladder. A robot that cannot wave should
still nod. Failing to move is not the same as failing to interact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

MIN_SPEED = 0.1
MAX_SPEECH_S = 6.0


@dataclass(frozen=True)
class JointTarget:
    """A goal angle for one joint, optionally scaled by a primitive parameter.

    The joint name may be templated -- `{hand}_elbow` becomes `right_elbow`
    when `wave` is called with `hand="right"` -- which is what lets one
    primitive serve both arms without duplicating its definition.
    """

    joint: str
    angle_rad: float
    scales_with: str | None = None

    def resolve(self, params: Mapping[str, Any]) -> tuple[str, float]:
        name = self.joint.format(**params) if "{" in self.joint else self.joint
        scale = 1.0
        if self.scales_with is not None:
            scale = float(params.get(self.scales_with, 1.0))
        return name, self.angle_rad * scale


@dataclass(frozen=True)
class Primitive:
    name: str
    limbs: tuple[str, ...]
    duration_s: float
    timeout_s: float
    description: str = ""
    fallbacks: tuple[str, ...] = ()
    targets: tuple[JointTarget, ...] = ()
    defaults: Mapping[str, Any] = field(default_factory=dict)

    def with_defaults(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return {**self.defaults, **params}

    def limbs_for(self, params: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(limb.format(**params) if "{" in limb else limb for limb in self.limbs)

    def joint_targets(self, params: Mapping[str, Any]) -> dict[str, float]:
        return dict(target.resolve(params) for target in self.targets)

    def duration_for(self, params: Mapping[str, Any]) -> float:
        if self.name == "say":
            base = min(MAX_SPEECH_S, 0.5 + 0.045 * len(str(params.get("text", ""))))
        else:
            base = float(params.get("duration_s", self.duration_s))
        speed = float(params.get("speed", 1.0))
        return base / max(MIN_SPEED, speed)


class PrimitiveRegistry:
    def __init__(self) -> None:
        self._primitives: dict[str, Primitive] = {}

    def register(self, primitive: Primitive) -> Primitive:
        if primitive.name in self._primitives:
            raise ValueError(f"primitive {primitive.name!r} already registered")
        self._primitives[primitive.name] = primitive
        return primitive

    def get(self, name: str) -> Primitive | None:
        return self._primitives.get(name)

    def require(self, name: str) -> Primitive:
        primitive = self._primitives.get(name)
        if primitive is None:
            raise KeyError(f"unknown primitive {name!r}")
        return primitive

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._primitives))

    def validate_fallbacks(self) -> None:
        """Catches a dangling fallback at startup rather than mid-interaction."""
        for primitive in self._primitives.values():
            for fallback in primitive.fallbacks:
                if fallback not in self._primitives:
                    raise ValueError(f"{primitive.name!r} falls back to unknown {fallback!r}")


def default_registry() -> PrimitiveRegistry:
    registry = PrimitiveRegistry()
    registry.register(
        Primitive(
            name="say",
            limbs=("voice",),
            duration_s=1.0,
            timeout_s=8.0,
            description="Speak a line of text.",
            defaults={"text": "..."},
        )
    )
    registry.register(
        Primitive(
            name="wake_up",
            limbs=("head",),
            duration_s=0.6,
            timeout_s=2.0,
            description="Lift the head and open the eyes.",
            targets=(JointTarget("head_pitch", -0.20), JointTarget("head_yaw", 0.0)),
        )
    )
    registry.register(
        Primitive(
            name="nod",
            limbs=("head",),
            duration_s=0.6,
            timeout_s=2.5,
            description="Nod once.",
            targets=(JointTarget("head_pitch", 0.45, scales_with="amplitude"),),
            defaults={"amplitude": 1.0},
        )
    )
    registry.register(
        Primitive(
            name="wave",
            limbs=("{hand}_arm",),
            duration_s=1.8,
            timeout_s=3.0,
            description="Raise an arm and wave.",
            fallbacks=("nod",),
            targets=(
                JointTarget("{hand}_shoulder_pitch", 1.40, scales_with="amplitude"),
                JointTarget("{hand}_elbow", 0.90, scales_with="amplitude"),
            ),
            defaults={"hand": "right", "amplitude": 0.8, "speed": 1.0, "cycles": 3},
        )
    )
    registry.register(
        Primitive(
            name="lower_arms",
            limbs=("left_arm", "right_arm"),
            duration_s=0.5,
            timeout_s=2.0,
            description="Bring both arms down; preempts a gesture in progress.",
            targets=(
                JointTarget("left_shoulder_pitch", 0.0),
                JointTarget("left_elbow", 0.0),
                JointTarget("right_shoulder_pitch", 0.0),
                JointTarget("right_elbow", 0.0),
            ),
        )
    )
    registry.register(
        Primitive(
            name="idle_breathe",
            limbs=("torso",),
            duration_s=2.0,
            timeout_s=4.0,
            description="Low-amplitude breathing motion so the robot never looks dead.",
            targets=(JointTarget("torso_pitch", 0.08),),
        )
    )
    registry.register(
        Primitive(
            name="relax",
            limbs=("torso", "left_arm", "right_arm"),
            duration_s=1.0,
            timeout_s=3.0,
            description="Return to the neutral resting pose.",
            targets=(
                JointTarget("torso_pitch", 0.0),
                JointTarget("left_shoulder_pitch", 0.0),
                JointTarget("left_elbow", 0.0),
                JointTarget("right_shoulder_pitch", 0.0),
                JointTarget("right_elbow", 0.0),
            ),
        )
    )
    registry.validate_fallbacks()
    return registry
