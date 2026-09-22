"""The debouncer is the component most likely to produce a visibly broken robot,
so it gets the most adversarial tests: flicker, occlusion, and low confidence.

The last group covers the stage as wired into the pipeline, where the claim is
structural rather than numerical: raw frames go in, only actionable events come
out, and nothing downstream is given the chance to react to a single frame."""

from __future__ import annotations

import asyncio

import pytest
from conftest import RecordingTracer, run

from robot_runtime.contracts.events import (
    PerceptionEvent,
    PersonAppeared,
    PersonLeft,
    PersonObservation,
    PersonObserved,
)
from robot_runtime.presence_filter.filter import PresenceConfig, PresenceDebouncer
from robot_runtime.presence_filter.service import PresenceFilterService
from robot_runtime.runtime.bus import EventBus
from robot_runtime.runtime.clock import FakeClock


def seen(confidence: float = 0.9) -> PersonObservation:
    return PersonObservation(present=True, confidence=confidence, track_id=1, source="test")


def empty() -> PersonObservation:
    return PersonObservation(present=False, confidence=0.0, source="test")


def feed(debouncer: PresenceDebouncer, observations, start: float = 0.0, dt: float = 0.05):
    events = []
    now = start
    for observation in observations:
        events.extend(debouncer.update(observation, now))
        now += dt
    return events


def test_appears_only_after_enough_confident_frames():
    debouncer = PresenceDebouncer(PresenceConfig(enter_frames=3, exit_frames=10))

    assert feed(debouncer, [seen(), seen()]) == []
    assert debouncer.present is False

    events = feed(debouncer, [seen()], start=0.1)
    assert len(events) == 1
    assert isinstance(events[0], PersonAppeared)
    assert debouncer.present is True


def test_low_confidence_frames_do_not_count():
    debouncer = PresenceDebouncer(PresenceConfig(min_confidence=0.6, enter_frames=2))
    assert feed(debouncer, [seen(0.3), seen(0.4), seen(0.55)]) == []
    assert debouncer.present is False


def test_brief_occlusion_does_not_end_the_visit():
    """The regression this whole component exists to prevent: a hand across the
    lens must not produce PersonLeft followed by a second greeting."""
    debouncer = PresenceDebouncer(PresenceConfig(enter_frames=2, exit_frames=15))
    feed(debouncer, [seen()] * 5)
    assert debouncer.present is True

    events = feed(debouncer, [empty()] * 10 + [seen()] * 5, start=0.25)
    assert events == []
    assert debouncer.present is True


def test_real_departure_emits_person_left_once():
    debouncer = PresenceDebouncer(PresenceConfig(enter_frames=2, exit_frames=5))
    feed(debouncer, [seen()] * 4)

    events = feed(debouncer, [empty()] * 20, start=0.2)
    assert len(events) == 1
    assert isinstance(events[0], PersonLeft)
    assert debouncer.present is False


def test_departure_reports_how_long_the_person_was_absent():
    debouncer = PresenceDebouncer(PresenceConfig(enter_frames=1, exit_frames=4))
    feed(debouncer, [seen()])
    events = feed(debouncer, [empty()] * 4, start=1.0, dt=0.1)
    assert isinstance(events[0], PersonLeft)
    assert events[0].absent_for_s == pytest.approx(0.3)


def test_alternating_frames_never_settle_into_present():
    debouncer = PresenceDebouncer(PresenceConfig(enter_frames=3, exit_frames=5))
    events = feed(debouncer, [seen(), empty()] * 30)
    assert events == []
    assert debouncer.present is False


def test_each_appearance_produces_exactly_one_event():
    debouncer = PresenceDebouncer(PresenceConfig(enter_frames=2, exit_frames=3))
    events = feed(debouncer, [seen()] * 6 + [empty()] * 6 + [seen()] * 6)
    assert [type(e).__name__ for e in events] == ["PersonAppeared", "PersonLeft", "PersonAppeared"]


def test_config_rejects_nonsense():
    with pytest.raises(ValueError):
        PresenceConfig(enter_frames=0)
    with pytest.raises(ValueError):
        PresenceConfig(min_confidence=1.5)


# --------------------------------------------------------------------------- #
# The stage, as wired into the pipeline
# --------------------------------------------------------------------------- #


def drive(frames, config):
    """Publish raw observations through the real service and collect what the
    rest of the system would be able to see."""

    async def main():
        clock = FakeClock()
        bus = EventBus()
        service = PresenceFilterService(bus, clock, RecordingTracer(), config)
        downstream = bus.subscribe(PerceptionEvent, name="test-downstream")
        await service.start()

        for index, observation in enumerate(frames):
            bus.publish(
                PersonObserved(trace_id=f"frame-{index}", timestamp=index * 0.05, observation=observation)
            )
        await asyncio.sleep(0)
        await clock.advance(0.0)

        emitted = []
        while not downstream.queue.empty():
            emitted.append(downstream.queue.get_nowait())
        await service.stop()
        return emitted, service

    return run(main)


def test_the_stage_emits_only_debounced_events():
    emitted, service = drive([seen()] * 5 + [empty()] * 20, PresenceConfig(enter_frames=2, exit_frames=5))

    assert [type(e).__name__ for e in emitted] == ["PersonAppeared", "PersonLeft"]
    assert service.observations == 25
    assert all(not isinstance(e, PersonObserved) for e in emitted)


def test_the_stage_swallows_a_flickering_stream_entirely():
    """Twenty raw frames in, zero actionable events out: the behavior layer is
    never given the chance to greet a flicker."""
    emitted, service = drive([seen(), empty()] * 10, PresenceConfig(enter_frames=3, exit_frames=5))

    assert emitted == []
    assert service.observations == 20


def test_an_emitted_event_carries_the_trace_id_of_the_frame_that_confirmed_it():
    emitted, _ = drive([seen()] * 3, PresenceConfig(enter_frames=3, exit_frames=5))

    assert len(emitted) == 1
    assert emitted[0].trace_id == "frame-2"
