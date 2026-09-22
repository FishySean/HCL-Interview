"""Safety verdicts.

These live in `contracts` for the same reason the primitive catalogue does: two
layers need them. The safety layer produces them and the motion layer consumes
them, so if they lived in either one the other would have to import it, and the
layer boundary would exist only in the documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .enums import DenialReason
from .primitives import Primitive


@dataclass(frozen=True)
class Denial:
    reason: DenialReason
    detail: str

    @property
    def allowed(self) -> bool:
        return False


@dataclass(frozen=True)
class Screened:
    """A command that passed the static checks, with the work already done so
    the motion layer does not recompute joint targets it was handed."""

    primitive: Primitive
    params: Mapping[str, Any]
    limbs: tuple[str, ...]
    joint_targets: Mapping[str, float]

    @property
    def allowed(self) -> bool:
        return True


@dataclass(frozen=True)
class Acquisition:
    limbs: tuple[str, ...]
    preempted: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        return True
