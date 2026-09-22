"""Seams. Every one of these exists so a simulated implementation can be swapped
for a real one without touching the layer that uses it."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .events import PersonObservation
from .safety import Acquisition, Denial, Screened


@runtime_checkable
class Clock(Protocol):
    """Injected so tests can run on virtual time instead of wall-clock time."""

    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


@runtime_checkable
class PersonDetector(Protocol):
    """The fast perception tier. One call, one frame, no state across layers."""

    name: str

    async def detect(self) -> PersonObservation: ...

    async def close(self) -> None: ...


@runtime_checkable
class MotionBackend(Protocol):
    """The hardware seam. `run` must either return normally on success or raise
    MotionFailure; it must be cancellable at any await point."""

    name: str

    async def run(
        self,
        primitive: str,
        params: Mapping[str, Any],
        duration_s: float,
    ) -> str: ...


@runtime_checkable
class Tracer(Protocol):
    def record(self, kind: str, trace_id: str, **fields: Any) -> None: ...


class Arbiter(Protocol):
    """What the motion layer is allowed to know about safety.

    The executor depends on this protocol rather than on the safety package, so
    the dependency runs motion -> contracts <- safety and never motion -> safety.
    """

    def screen(
        self, primitive_name: str, params: Mapping[str, Any], trace_id: str = "-"
    ) -> Screened | Denial: ...

    def acquire(
        self,
        command_id: str,
        priority: int,
        limbs: tuple[str, ...],
        primitive: str = "",
        trace_id: str = "-",
    ) -> Acquisition | Denial: ...

    def release(self, command_id: str) -> None: ...


class Policy(Protocol):
    """Arbitrates between behaviors whose triggers all fired.

    The contract that makes a language model safe to plug in here: `decide` may
    only return a name drawn from `candidates`. Anything else is rejected by the
    caller and the deterministic fallback runs instead.
    """

    name: str

    async def decide(
        self,
        candidates: Sequence[str],
        context: Mapping[str, Any],
    ) -> str: ...
