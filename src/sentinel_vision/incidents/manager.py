"""Deterministic incident state machine (ADR-0012).

``IncidentManager`` sits directly on top of ``EventEngine``'s output: it takes
the events of a frame and advances security incidents through their lifecycle,
one incident per entity at a time. It applies no thresholds of its own — the
event engine already judged that something is happening — and its only judgment
is the *response* judgment: given that an unauthorized entity triggered an
event, has the guard been told, have they answered, and is this now loud enough
to escalate?

Two kinds of transition, deliberately separated:

- **Automatic, frame-driven** (``update``): timeout budgets count frames since
  the last transition and escalate an unanswered incident by itself.
- **External, human-driven** (``record_response``, ``resolve``): a guard acting
  on the incident. Out-of-band, because this PR has no notification delivery
  mechanism to call them automatically.

Every timeout budget is counted in frames, not wall-clock time, for the same
reason every earlier layer counts frames (ADR-0006, ADR-0009): determinism.
Replay the same frames and the same transitions happen on the same frame ids.
Converting budgets to seconds requires the stream's frame rate at runtime, which
is deferred until notification delivery lands (see ADR-0012).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from sentinel_vision.events.event import Event, EventStatus
from sentinel_vision.incidents.incident import Incident, IncidentState


@dataclass
class _IncidentRecord:
    """Per-entity incident bookkeeping kept between ``update`` calls."""

    incident: Incident
    frames_in_state: int = 0


class IncidentManager:
    """Turns events into incidents and drives them through their lifecycle.

    Args:
        guard_response_budget: Frames a guard-notified incident waits for a
            response before the alarm activates. Must be > 0.
        escalation_budget: Frames an alarm-activated incident waits before it
            escalates. Must be > 0.
        is_authorized: Identity resolution function taking an ``entity_id`` and
            returning whether it is a known/whitelisted identity. The whitelist
            itself is not implemented here (ADR-0012 defers it); the manager only
            consumes the boolean.

    One active incident per ``entity_id`` at a time: a second unauthorized event
    for an entity that already has an active incident is folded into it rather
    than opening a parallel incident (ADR-0012's stated simplification).
    """

    def __init__(
        self,
        guard_response_budget: int,
        escalation_budget: int,
        is_authorized: Callable[[int], bool],
    ) -> None:
        if guard_response_budget <= 0:
            raise ValueError(
                f"guard_response_budget ({guard_response_budget}) must be > 0"
            )
        if escalation_budget <= 0:
            raise ValueError(f"escalation_budget ({escalation_budget}) must be > 0")
        self._guard_response_budget = guard_response_budget
        self._escalation_budget = escalation_budget
        self._is_authorized = is_authorized
        self._records: dict[int, _IncidentRecord] = {}
        self._next_incident_id = 0

    def update(self, frame_id: int, events: list[Event]) -> list[Incident]:
        """Advance every active incident to ``frame_id`` with this frame's events.

        Returns only the incidents whose state changed on this frame, in
        deterministic order: timeout-driven transitions first, in sorted
        ``entity_id`` order, then newly opened incidents in event order and then
        ``entity_ids`` order. A frame where nothing changes returns ``[]`` — the
        same only-emit-deltas convention ``EventEngine`` uses.

        Timeout transitions are evaluated before new incidents open, so an
        incident is never charged for its own opening frame:

        1. ``GUARD_NOTIFIED`` -> ``ALARM_ACTIVATED`` once ``frames_in_state``
           reaches ``guard_response_budget``.
        2. ``ALARM_ACTIVATED`` -> ``ESCALATED`` once ``frames_in_state``
           reaches ``escalation_budget``.

        Only these two states accumulate a frame counter. An incident a guard
        has answered (``RESPONDED``), or one that has escalated
        (``ESCALATED``), waits for an external ``resolve`` and does not progress
        on its own. An incident is deliberately *not* auto-resolved when its
        triggering event closes: the event ending says the hazard geometry
        ended, not that anyone handled the person.

        Opening rule: an ``OPEN`` event opens an incident for each of its
        entities that ``is_authorized`` rejects and that has no active incident
        yet. An event whose every entity is authorized is not a security
        incident and opens nothing. ``CLOSED`` events are ignored here.
        """
        if frame_id < 0:
            raise ValueError(f"frame_id ({frame_id}) must be non-negative")

        changed: list[Incident] = []

        for entity_id in sorted(self._records):
            record = self._records[entity_id]
            if record.incident.state is IncidentState.GUARD_NOTIFIED:
                record.frames_in_state += 1
                if record.frames_in_state >= self._guard_response_budget:
                    self._transition(
                        record,
                        state=IncidentState.ALARM_ACTIVATED,
                        alarm_activated_frame_id=frame_id,
                        changed=changed,
                    )
            elif record.incident.state is IncidentState.ALARM_ACTIVATED:
                record.frames_in_state += 1
                if record.frames_in_state >= self._escalation_budget:
                    self._transition(
                        record,
                        state=IncidentState.ESCALATED,
                        escalated_frame_id=frame_id,
                        changed=changed,
                    )

        for event in events:
            if event.status is not EventStatus.OPEN:
                continue
            for entity_id in event.entity_ids:
                if entity_id in self._records or self._is_authorized(entity_id):
                    continue
                record = _IncidentRecord(
                    incident=Incident(
                        incident_id=self._next_incident_id,
                        entity_id=entity_id,
                        triggering_event_type=event.event_type,
                        state=IncidentState.OPEN,
                        opened_frame_id=frame_id,
                        guard_notified_frame_id=None,
                        responded_frame_id=None,
                        alarm_activated_frame_id=None,
                        escalated_frame_id=None,
                        resolved_frame_id=None,
                    ),
                    frames_in_state=0,
                )
                self._records[entity_id] = record
                self._next_incident_id += 1
                # Notification is synchronous in this model: the guard is
                # notified on the opening frame and the incident moves straight
                # to GUARD_NOTIFIED, so OPEN exists only as the incident's
                # initial state and never appears in the returned list. ADR-0012
                # records that no delivery delay is modeled yet.
                self._transition(
                    record,
                    state=IncidentState.GUARD_NOTIFIED,
                    guard_notified_frame_id=frame_id,
                    changed=changed,
                )

        return changed

    def record_response(self, entity_id: int, frame_id: int) -> Incident:
        """Record that a guard responded to ``entity_id``'s active incident.

        Transitions ``GUARD_NOTIFIED`` -> ``RESPONDED``, which stops the response
        timer: a guard who answered before the budget expires gets no alarm.

        Raises:
            LookupError: If the entity has no active incident, or its active
                incident is not in ``GUARD_NOTIFIED``. Silently doing nothing
                would hide a guard action that never registered.
        """
        if frame_id < 0:
            raise ValueError(f"frame_id ({frame_id}) must be non-negative")
        record = self._require_active(entity_id)
        self._require_state(record, (IncidentState.GUARD_NOTIFIED,))
        return self._transition(
            record,
            state=IncidentState.RESPONDED,
            responded_frame_id=frame_id,
        )

    def resolve(self, entity_id: int, frame_id: int) -> Incident:
        """Resolve ``entity_id``'s active incident, freeing the entity for a new one.

        Transitions ``RESPONDED`` -> ``RESOLVED`` (a guard handled it) or
        ``ESCALATED`` -> ``RESOLVED`` (escalation handled it). An incident still
        in ``ALARM_ACTIVATED`` cannot be resolved: doing so would assert a human
        took it over during the window where the alarm is sounding but nobody
        has yet been told to go look.

        Raises:
            LookupError: If the entity has no active incident, or its active
                incident is in neither ``RESPONDED`` nor ``ESCALATED``.
        """
        if frame_id < 0:
            raise ValueError(f"frame_id ({frame_id}) must be non-negative")
        record = self._require_active(entity_id)
        self._require_state(
            record, (IncidentState.RESPONDED, IncidentState.ESCALATED)
        )
        resolved = self._transition(
            record,
            state=IncidentState.RESOLVED,
            resolved_frame_id=frame_id,
        )
        del self._records[entity_id]
        return resolved

    @staticmethod
    def _transition(
        record: _IncidentRecord,
        *,
        state: IncidentState,
        changed: list[Incident] | None = None,
        **frame_fields: Any,
    ) -> Incident:
        """Apply a state change to ``record``, resetting its frame counter.

        ``frame_fields`` carries exactly the one frame id recording the frame
        the incident entered ``state`` on (``responded_frame_id``, and so on);
        every other frame id is carried over unchanged, which is what keeps an
        incident's branch marker intact as it moves to ``RESOLVED``.

        Every transition re-enters a state, so the counter restarts at 0 here —
        that is what makes each budget "frames since the last transition" rather
        than "frames since the incident opened".

        ``changed`` is the list ``update`` returns; the external hooks omit it
        because they return the incident directly.
        """
        incident = replace(record.incident, state=state, **frame_fields)
        record.incident = incident
        record.frames_in_state = 0
        if changed is not None:
            changed.append(incident)
        return incident

    def _require_active(self, entity_id: int) -> _IncidentRecord:
        record = self._records.get(entity_id)
        if record is None:
            raise LookupError(
                f"no active incident for entity_id {entity_id} "
                f"(active entity_ids: {sorted(self._records)})"
            )
        return record

    @staticmethod
    def _require_state(
        record: _IncidentRecord, expected: tuple[IncidentState, ...]
    ) -> None:
        incident = record.incident
        if incident.state not in expected:
            expected_names = " or ".join(state.name for state in expected)
            raise LookupError(
                f"incident {incident.incident_id} for entity_id "
                f"{incident.entity_id} is in state {incident.state.name}, "
                f"expected {expected_names}"
            )