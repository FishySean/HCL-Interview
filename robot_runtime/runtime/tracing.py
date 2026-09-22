"""Structured tracing.

A `trace_id` is born with the perception event that caused a reaction and is
carried, unchanged, through the behavior decision, every motion command, and
every motion result. One id therefore reconstructs a full causal chain:

    grep a1b2c3d4 trace.jsonl
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, TextIO


@dataclass
class TraceRecord:
    kind: str
    trace_id: str
    at: float
    fields: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> str:
        return json.dumps({"kind": self.kind, "trace": self.trace_id, "t": round(self.at, 4), **self.fields})


class Tracer:
    """Keeps records in memory (tests assert on them) and optionally streams JSONL.

    The interface is deliberately one method wide so it can be re-pointed at
    OpenTelemetry later without touching a single call site.
    """

    def __init__(self, clock: Any, stream: TextIO | None = None, keep: int = 2000) -> None:
        self._clock = clock
        self._stream = stream
        self._keep = keep
        self.records: list[TraceRecord] = []

    def record(self, kind: str, trace_id: str, **fields: Any) -> None:
        entry = TraceRecord(kind=kind, trace_id=trace_id, at=self._clock.now(), fields=fields)
        self.records.append(entry)
        if len(self.records) > self._keep:
            del self.records[: len(self.records) - self._keep]
        if self._stream is not None:
            self._stream.write(entry.as_json() + "\n")
            self._stream.flush()

    def of_kind(self, kind: str) -> list[TraceRecord]:
        return [r for r in self.records if r.kind == kind]

    def chain(self, trace_id: str) -> list[TraceRecord]:
        return [r for r in self.records if r.trace_id == trace_id]


class NullTracer:
    def record(self, kind: str, trace_id: str, **fields: Any) -> None:  # noqa: D102
        return None


def stdout_tracer(clock: Any) -> Tracer:
    return Tracer(clock, stream=sys.stdout)
