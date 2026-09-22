"""The robot's kinematic description, loaded from configuration.

Deliberately data rather than code: joint limits and limb membership are
properties of the machine, not of the program, and the moment a second robot
exists they have to come from a file anyway. `robot.yaml` stands in for a URDF
here; `from_urdf` is the seam where a real one would be parsed instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "robot.yaml"


class RobotModelError(ValueError):
    """Raised at startup for a malformed model. Never at motion time."""


@dataclass(frozen=True)
class Joint:
    name: str
    min_rad: float
    max_rad: float
    limb: str

    def violation(self, angle: float) -> str | None:
        if angle < self.min_rad or angle > self.max_rad:
            return (
                f"{self.name}={angle:.3f} outside [{self.min_rad:.3f}, {self.max_rad:.3f}]"
            )
        return None


@dataclass(frozen=True)
class ForbiddenBox:
    name: str
    bounds: Mapping[str, tuple[float, float]]

    def contains(self, targets: Mapping[str, float]) -> bool:
        for joint, (low, high) in self.bounds.items():
            if joint not in targets:
                return False
            if not (low <= targets[joint] <= high):
                return False
        return True


class RobotModel:
    def __init__(
        self,
        name: str,
        joints: Mapping[str, Joint],
        limbs: Mapping[str, tuple[str, ...]],
        forbidden_boxes: tuple[ForbiddenBox, ...] = (),
    ) -> None:
        self.name = name
        self.joints = dict(joints)
        self.limbs = dict(limbs)
        self.forbidden_boxes = forbidden_boxes
        self._validate()

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | Path | None = None) -> RobotModel:
        try:
            import yaml  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RobotModelError(
                "The robot model is YAML. Install it with: pip install -r requirements.txt"
            ) from exc

        config_path = Path(path) if path is not None else DEFAULT_CONFIG
        with open(config_path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.from_mapping(raw, source=str(config_path))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], source: str = "<memory>") -> RobotModel:
        if not isinstance(raw, Mapping):
            raise RobotModelError(f"{source}: expected a mapping at the top level")

        limb_config = raw.get("limbs") or {}
        limbs: dict[str, tuple[str, ...]] = {}
        joint_to_limb: dict[str, str] = {}
        for limb_name, body in limb_config.items():
            joint_names = tuple((body or {}).get("joints") or ())
            limbs[limb_name] = joint_names
            for joint_name in joint_names:
                if joint_name in joint_to_limb:
                    raise RobotModelError(
                        f"{source}: joint {joint_name!r} belongs to both "
                        f"{joint_to_limb[joint_name]!r} and {limb_name!r}"
                    )
                joint_to_limb[joint_name] = limb_name

        joints: dict[str, Joint] = {}
        for joint_name, limits in (raw.get("joints") or {}).items():
            if joint_name not in joint_to_limb:
                raise RobotModelError(f"{source}: joint {joint_name!r} belongs to no limb")
            joints[joint_name] = Joint(
                name=joint_name,
                min_rad=float(limits["min"]),
                max_rad=float(limits["max"]),
                limb=joint_to_limb[joint_name],
            )

        boxes = tuple(
            ForbiddenBox(
                name=entry.get("name", f"box_{index}"),
                bounds={
                    joint: (float(bounds[0]), float(bounds[1]))
                    for joint, bounds in (entry.get("joints") or {}).items()
                },
            )
            for index, entry in enumerate((raw.get("self_collision") or {}).get("forbidden_boxes") or ())
        )

        return cls(
            name=str(raw.get("robot", "unnamed")),
            joints=joints,
            limbs=limbs,
            forbidden_boxes=boxes,
        )

    @classmethod
    def from_urdf(cls, path: str | Path) -> RobotModel:  # pragma: no cover
        """The seam a real deployment uses. Left unimplemented on purpose rather
        than faked, so nobody mistakes the YAML stand-in for URDF support."""
        raise NotImplementedError(
            "URDF parsing is not implemented; see README 'What I would improve'."
        )

    # ----------------------------------------------------------------- query
    def limb_of(self, joint: str) -> str | None:
        found = self.joints.get(joint)
        return found.limb if found else None

    def limbs_for(self, joints: Mapping[str, float]) -> tuple[str, ...]:
        found = {self.joints[name].limb for name in joints if name in self.joints}
        return tuple(sorted(found))

    def limit_violations(self, targets: Mapping[str, float]) -> list[str]:
        problems: list[str] = []
        for joint_name, angle in targets.items():
            joint = self.joints.get(joint_name)
            if joint is None:
                problems.append(f"unknown joint {joint_name!r}")
                continue
            violation = joint.violation(angle)
            if violation is not None:
                problems.append(violation)
        return problems

    # ------------------------------------------------------------ validation
    def _validate(self) -> None:
        """Startup-time checks. A model error should stop the robot from booting,
        never surface as a surprise halfway through a greeting."""
        for limb, joint_names in self.limbs.items():
            for joint_name in joint_names:
                if joint_name not in self.joints:
                    raise RobotModelError(f"limb {limb!r} lists undeclared joint {joint_name!r}")
        for joint in self.joints.values():
            if joint.min_rad >= joint.max_rad:
                raise RobotModelError(f"joint {joint.name!r} has an empty range")
        for box in self.forbidden_boxes:
            for joint_name in box.bounds:
                if joint_name not in self.joints:
                    raise RobotModelError(
                        f"self-collision box {box.name!r} names undeclared joint {joint_name!r}"
                    )
