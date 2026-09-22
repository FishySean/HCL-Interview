"""Turns a noisy per-frame detector into a stable presence state.

This is a layer of its own, sitting between perception and behavior, because it
belongs to neither. It is not perception -- it holds no model and looks at no
pixels. It is not behavior -- it decides nothing about what the robot does. It
is the contract boundary where *noisy* becomes *actionable*, and giving it its
own module means the rule "the behavior layer never sees a raw frame" is
enforced by the wiring rather than by discipline.

It is also the most safety-relevant component in the sensing path, and it is
not a neural network. A detector flickers: a person half-occluded for 200ms
produces absent->present, and a naive system greets them a second time. Presence
is not a per-frame boolean, it is a debounced state over time.

The hysteresis is deliberately asymmetric:

  * enter after a few consecutive confident frames (~100ms) so the robot feels
    instant to a human walking up to it;
  * leave only after a much longer gap (~1.5s) so a hand across the lens, a
    turn of the body, or one dropped frame does not end the interaction.

Everything here is pure and synchronous: no clock, no I/O, no bus. That makes
the flapping edge cases cheap to test exhaustively.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.events import PerceptionEvent, PersonAppeared, PersonLeft, PersonObservation, new_id


@dataclass(frozen=True)
class PresenceConfig:
    min_confidence: float = 0.5
    enter_frames: int = 3
    exit_frames: int = 15

    def __post_init__(self) -> None:
        if self.enter_frames < 1 or self.exit_frames < 1:
            raise ValueError("frame thresholds must be >= 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")


class PresenceDebouncer:
    def __init__(self, config: PresenceConfig | None = None) -> None:
        self.config = config or PresenceConfig()
        self._present = False
        self._hits = 0
        self._misses = 0
        self._track_id: int | None = None
        self._absent_since: float | None = None

    @property
    def present(self) -> bool:
        return self._present

    def update(
        self, observation: PersonObservation, now: float, trace_id: str | None = None
    ) -> list[PerceptionEvent]:
        """Feed one frame. Returns the (usually empty) list of events it caused.

        `trace_id` lets the caller carry the id of the frame that tipped the
        decision, so a greeting can be traced back to the exact observation
        that confirmed the person rather than starting a fresh chain here.
        """
        confident = observation.present and observation.confidence >= self.config.min_confidence

        if confident:
            self._hits += 1
            self._misses = 0
            self._absent_since = None
            if observation.track_id is not None:
                self._track_id = observation.track_id
        else:
            self._misses += 1
            self._hits = 0
            if self._absent_since is None:
                self._absent_since = now

        if not self._present and confident and self._hits >= self.config.enter_frames:
            self._present = True
            return [
                PersonAppeared(
                    trace_id=trace_id or new_id(),
                    timestamp=now,
                    track_id=self._track_id,
                    confidence=observation.confidence,
                )
            ]

        if self._present and not confident and self._misses >= self.config.exit_frames:
            self._present = False
            absent_for = now - (self._absent_since if self._absent_since is not None else now)
            track_id, self._track_id = self._track_id, None
            return [
                PersonLeft(
                    trace_id=trace_id or new_id(),
                    timestamp=now,
                    track_id=track_id,
                    absent_for_s=absent_for,
                )
            ]

        return []
