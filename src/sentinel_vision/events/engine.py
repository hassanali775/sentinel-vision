"""Hysteresis event engine for the deterministic pipeline (PR-009).

``EventEngine`` is the first layer allowed to apply thresholds, accumulate
over time, and produce judgments (ADR-0008's facts-not-violations boundary).
Each rule registered with the engine is given its own ``sustain_frames`` and
``clear_frames`` budgets, directly mirroring how ``PersistentEntityTracker``
budgets consecutive unobserved frames (ADR-0006): a condition must hold for
``sustain_frames`` consecutive frames to open an event, and must then be
absent for ``clear_frames`` consecutive frames to close it.

The engine is deterministic by construction: rules are evaluated in
registration order and per-rule keys are iterated in sorted order, so a given
sequence of ``SpatialFrameObservation`` values always produces the same
sequence of events.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from sentinel_vision.events.event import Event, EventStatus, EventType
from sentinel_vision.events.rules import BaseEventRule
from sentinel_vision.spatial.workspace import SpatialFrameObservation


@dataclass
class _RuleState:
    """Per-rule accumulation state kept between ``update`` calls."""

    rule: BaseEventRule[Any]
    sustain_frames: int
    clear_frames: int
    true_counts: dict[Any, int] = field(default_factory=dict)
    false_counts: dict[Any, int] = field(default_factory=dict)
    open_events: dict[Any, Event] = field(default_factory=dict)


class EventEngine:
    """Converts per-frame spatial facts into sustained, hysteresis-bounded events.

    Rules are passed to the constructor as ``(rule, sustain_frames,
    clear_frames)`` triples. Each rule's keys are tracked independently, so a
    rule may have several concurrent events (e.g. several entity pairs in
    proximity) at different points in their sustain/clear cycles.
    """

    def __init__(self, rules: list[tuple[BaseEventRule[Any], int, int]]) -> None:
        if not rules:
            raise ValueError("at least one rule must be registered")
        states: list[_RuleState] = []
        for rule, sustain_frames, clear_frames in rules:
            if sustain_frames <= 0:
                raise ValueError(
                    f"sustain_frames ({sustain_frames}) must be > 0 for "
                    f"rule {type(rule).__name__}"
                )
            if clear_frames <= 0:
                raise ValueError(
                    f"clear_frames ({clear_frames}) must be > 0 for "
                    f"rule {type(rule).__name__}"
                )
            states.append(
                _RuleState(
                    rule=rule,
                    sustain_frames=sustain_frames,
                    clear_frames=clear_frames,
                )
            )
        self._rule_states = states

    def update(
        self, frame_id: int, spatial: SpatialFrameObservation
    ) -> list[Event]:
        """Advance the engine to ``frame_id`` with this frame's spatial facts.

        Returns only the events that opened or closed on this frame, in
        deterministic order: rules in registration order, then keys in sorted
        order. A frame where nothing changes returns ``[]``.

        Per rule, for each of its keys, in order:
        1. Drop accumulated true-frame counts for keys whose condition no
           longer holds.
        2. Reset the false-frame count of any open event whose condition is
           true again — an event that never reached ``clear_frames`` of
           consecutive absence stays open (flicker guarantee).
        3. Increment true-frame counts and open an event for any key whose
           count reaches ``sustain_frames``.
        4. Increment false-frame counts for open events whose condition is
           absent, and close any event whose count reaches ``clear_frames``.
        """
        if frame_id < 0:
            raise ValueError(f"frame_id ({frame_id}) must be non-negative")

        events: list[Event] = []
        for state in self._rule_states:
            current_true = state.rule.condition_holds(spatial)

            for key in sorted(state.true_counts):
                if key not in current_true:
                    del state.true_counts[key]

            for key in list(state.open_events):
                if key in current_true:
                    state.false_counts.pop(key, None)

            for key in sorted(current_true):
                state.true_counts[key] = state.true_counts.get(key, 0) + 1
                if (
                    key not in state.open_events
                    and state.true_counts[key] >= state.sustain_frames
                ):
                    opened = self._make_event(state.rule, key, frame_id)
                    state.open_events[key] = opened
                    state.false_counts[key] = 0
                    events.append(opened)

            for key in sorted(state.open_events):
                if key in current_true:
                    continue
                state.false_counts[key] = state.false_counts.get(key, 0) + 1
                if state.false_counts[key] >= state.clear_frames:
                    open_event = state.open_events.pop(key)
                    del state.false_counts[key]
                    events.append(
                        replace(
                            open_event,
                            status=EventStatus.CLOSED,
                            closed_frame_id=frame_id,
                        )
                    )

        return events

    def _make_event(
        self, rule: BaseEventRule[Any], key: Any, frame_id: int
    ) -> Event:
        """Build an ``OPEN`` event for ``key`` from ``rule``'s event metadata."""
        entity_ids: tuple[int, ...]
        if rule.event_type is EventType.PROXIMITY_HAZARD:
            entity_ids = (key[0], key[1])
        else:
            entity_ids = (key,)
        return Event(
            event_type=rule.event_type,
            entity_ids=entity_ids,
            status=EventStatus.OPEN,
            opened_frame_id=frame_id,
            closed_frame_id=None,
            zone_name=rule.zone_name,
        )
