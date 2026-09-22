"""Self-collision checking -- a placeholder, and labelled as one.

What this does: looks up the commanded goal pose in a hand-written table of
joint-space boxes that are known to put the robot into itself.

What it does not do, and what a real implementation must:

* reason about *geometry* rather than joint values, by sweeping the collision
  meshes declared in the URDF (`<collision>`) through forward kinematics and
  testing pairs with FCL, Bullet, or MoveIt's `PlanningScene`;
* check the whole *trajectory*, not just the endpoint -- a start and a goal can
  both be safe while the path between them is not;
* include the environment and the person, not just the robot's own links;
* respect an allowed-collisions matrix for links that are always in contact.

It is separated behind a single function so that substitution is a one-line
change in the composition root, and so that the thing being faked is obvious to
a reviewer instead of buried inside the gate.
"""

from __future__ import annotations

from typing import Mapping, Protocol

from .model import RobotModel


class CollisionCheck(Protocol):
    def __call__(self, targets: Mapping[str, float], model: RobotModel) -> str | None:
        """Return a description of the collision, or None if the pose is clear."""


def approximate_self_collision(targets: Mapping[str, float], model: RobotModel) -> str | None:
    for box in model.forbidden_boxes:
        if box.contains(targets):
            joints = ", ".join(f"{name}={targets[name]:.2f}" for name in sorted(box.bounds))
            return f"{box.name} ({joints})"
    return None


def never_collides(targets: Mapping[str, float], model: RobotModel) -> str | None:
    """For tests that are about arbitration rather than geometry."""
    return None
