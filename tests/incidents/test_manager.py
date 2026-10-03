"""Behavior tests for IncidentManager (ADR-0012).

The budget boundaries are the point of this file. Each timeout test drives the
manager frame by frame and asserts the exact frame on which the transition
fires, hand-derived from the rule the manager implements: a budget counts
*frames since the last transition*, and an incident is never charged for its own
opening frame. An incident opened on frame F with ``guard_response_budget = B``
activates its alarm on frame ``F + B``, not ``F + B - 1`` and not ``F + B + 1``.
"""

import pytest

from sentinel_vision.events.event import Event, EventStatus, EventType
from sentinel_vision.incidents.incident import IncidentState
from sentinel_vision.incidents.manager import IncidentManager


def _manager(
    guard_response_budget: int = 3,
    escalation_budget: int = 2,
    authorized: set[int] | None = None,
) -> IncidentManager:
    allowed = authorized or set()
    return IncidentManager(
        guard_response_budget=guard_response_budget,
        escalation_budget=escalation_budget,
        is_authorized=lambda entity_id: entity_id in allowed,
    )


def _intrusion(
    entity_id: int, opened_frame_id: int, *, status: EventStatus = EventStatus.OPEN
) -> Event:
    return Event(
        event_type=EventType.ZONE_INTRUSION,
        entity_ids=(entity_id,),
        status=status,
        opened_frame_id=opened_frame_id,
        closed_frame_id=(
            None if status is EventStatus.OPEN else opened_frame_id + 1
        ),
        zone_name="lane_a",
    )


def _proximity(
    first: int, second: int, opened_frame_id: int, *, status: EventStatus = EventStatus.OPEN
) -> Event:
    return Event(
        event_type=EventType.PROXIMITY_HAZARD,
        entity_ids=(first, second),
        status=status,
        opened_frame_id=opened_frame_id,
        closed_frame_id=(
            None if status is EventStatus.OPEN else opened_frame_id + 1
        ),
        zone_name=None,
    )


class TestOpening:
    def test_unauthorized_event_opens_and_notifies_on_the_same_frame(self) -> None:
        manager = _manager()
        changed = manager.update(4, [_intrusion(0, 4)])

        assert len(changed) == 1
        incident = changed[0]
        assert incident.incident_id == 0
        assert incident.entity_id == 0
        assert incident.state is IncidentState.GUARD_NOTIFIED
        assert incident.triggering_event_type is EventType.ZONE_INTRUSION
        assert incident.opened_frame_id == 4
        assert incident.guard_notified_frame_id == 4
        assert incident.responded_frame_id is None
        assert incident.alarm_activated_frame_id is None
        assert incident.escalated_frame_id is None
        assert incident.resolved_frame_id is None

    def test_authorized_entity_never_creates_an_incident_even_sustained(self) -> None:
        manager = _manager(authorized={0})
        for frame_id in range(30):
            assert manager.update(frame_id, [_intrusion(0, frame_id)]) == []

    def test_only_the_unauthorized_entity_of_a_pair_opens_an_incident(self) -> None:
        manager = _manager(authorized={0})
        changed = manager.update(0, [_proximity(0, 1, 0)])

        assert [i.entity_id for i in changed] == [1]

    def test_fully_authorized_pair_opens_nothing(self) -> None:
        manager = _manager(authorized={0, 1})
        assert manager.update(0, [_proximity(0, 1, 0)]) == []

    def test_two_unauthorized_entities_in_one_event_get_independent_incidents(self) -> None:
        manager = _manager()
        changed = manager.update(0, [_proximity(0, 1, 0)])

        assert [(i.entity_id, i.incident_id) for i in changed] == [(0, 0), (1, 1)]

    def test_closed_events_open_nothing(self) -> None:
        manager = _manager()
        closed = _intrusion(0, 0, status=EventStatus.CLOSED)
        assert manager.update(5, [closed]) == []

    def test_second_event_for_the_same_entity_does_not_open_a_second_incident(self) -> None:
        manager = _manager(guard_response_budget=5)
        first = manager.update(0, [_intrusion(0, 0)])

        # Frames 1 through 4 re-trigger the same entity: no new incident, and
        # the repeat triggers do not restart the budget, so the alarm still
        # lands on frame 5.
        for frame_id in range(1, 5):
            assert manager.update(frame_id, [_intrusion(0, frame_id)]) == []

        activated = manager.update(5, [])
        assert len(activated) == 1
        assert activated[0].incident_id == first[0].incident_id
        assert activated[0].opened_frame_id == 0
        assert activated[0].alarm_activated_frame_id == 5


class TestGuardResponseBudget:
    def test_no_alarm_one_frame_before_the_budget(self) -> None:
        manager = _manager(guard_response_budget=3)
        manager.update(0, [_intrusion(0, 0)])

        # Frame 0 notified the guard. Frames 1 and 2 are the 1st and 2nd frames
        # since: 1 < 3 and 2 < 3, so the alarm has not fired.
        assert manager.update(1, []) == []
        assert manager.update(2, []) == []

    def test_alarm_activates_exactly_at_the_guard_response_budget(self) -> None:
        manager = _manager(guard_response_budget=3)
        manager.update(0, [_intrusion(0, 0)])
        assert manager.update(1, []) == []
        assert manager.update(2, []) == []

        # Frame 3 is the 3rd frame since notification (0 + budget 3).
        activated = manager.update(3, [])
        assert len(activated) == 1
        incident = activated[0]
        assert incident.state is IncidentState.ALARM_ACTIVATED
        assert incident.alarm_activated_frame_id == 3
        assert incident.guard_notified_frame_id == 0
        assert incident.responded_frame_id is None
        assert incident.escalated_frame_id is None
        assert incident.resolved_frame_id is None

    def test_single_frame_budget_alarms_on_the_very_next_frame(self) -> None:
        manager = _manager(guard_response_budget=1)
        manager.update(0, [_intrusion(0, 0)])

        activated = manager.update(1, [])
        assert len(activated) == 1
        assert activated[0].state is IncidentState.ALARM_ACTIVATED
        assert activated[0].alarm_activated_frame_id == 1

    def test_response_before_the_budget_stops_the_alarm(self) -> None:
        manager = _manager(guard_response_budget=3)
        manager.update(0, [_intrusion(0, 0)])

        responded = manager.record_response(0, 2)
        assert responded.state is IncidentState.RESPONDED
        assert responded.responded_frame_id == 2
        assert responded.alarm_activated_frame_id is None

        # Frame 3 is where the alarm would have fired. It never does, and
        # nothing is reported for any later frame either.
        for frame_id in range(3, 25):
            assert manager.update(frame_id, []) == []

    def test_incident_is_not_auto_resolved_when_its_event_closes(self) -> None:
        """An event ending says the geometry ended, not that anyone handled it."""
        manager = _manager(guard_response_budget=10)
        manager.update(0, [_intrusion(0, 0)])
        closed = _intrusion(0, 0, status=EventStatus.CLOSED)

        assert manager.update(5, [closed]) == []
        assert manager.record_response(0, 6).state is IncidentState.RESPONDED


class TestEscalationBudget:
    def test_escalates_exactly_at_the_escalation_budget(self) -> None:
        manager = _manager(guard_response_budget=2, escalation_budget=3)
        manager.update(0, [_intrusion(0, 0)])

        # Guard notified on 0, so the alarm fires on frame 2 (0 + 2).
        assert manager.update(1, []) == []
        activated = manager.update(2, [])
        assert activated[0].state is IncidentState.ALARM_ACTIVATED
        assert activated[0].alarm_activated_frame_id == 2

        # The escalation budget counts from the alarm frame: frames 3 and 4 are
        # the 1st and 2nd frames since, frame 5 is the 3rd (2 + 3).
        assert manager.update(3, []) == []
        assert manager.update(4, []) == []
        escalated = manager.update(5, [])
        assert len(escalated) == 1
        incident = escalated[0]
        assert incident.state is IncidentState.ESCALATED
        assert incident.escalated_frame_id == 5
        assert incident.alarm_activated_frame_id == 2
        assert incident.responded_frame_id is None
        assert incident.resolved_frame_id is None

        # ESCALATED is the end of the automatic line: no further transitions.
        assert manager.update(6, []) == []
        assert manager.update(7, []) == []

    def test_answered_incident_never_escalates(self) -> None:
        manager = _manager(guard_response_budget=10, escalation_budget=1)
        manager.update(0, [_intrusion(0, 0)])
        manager.record_response(0, 1)

        for frame_id in range(2, 20):
            assert manager.update(frame_id, []) == []


class TestExternalHooks:
    def test_resolve_after_a_response_reaches_resolved(self) -> None:
        manager = _manager(guard_response_budget=3)
        manager.update(0, [_intrusion(0, 0)])
        manager.record_response(0, 2)

        resolved = manager.resolve(0, 6)
        assert resolved.state is IncidentState.RESOLVED
        assert resolved.resolved_frame_id == 6
        assert resolved.responded_frame_id == 2
        assert resolved.alarm_activated_frame_id is None
        assert resolved.escalated_frame_id is None

    def test_resolve_after_escalation_reaches_resolved(self) -> None:
        manager = _manager(guard_response_budget=1, escalation_budget=1)
        manager.update(0, [_intrusion(0, 0)])
        manager.update(1, [])  # alarm at 1
        assert manager.update(2, [])  # escalated at 2

        resolved = manager.resolve(0, 4)
        assert resolved.state is IncidentState.RESOLVED
        assert resolved.resolved_frame_id == 4
        assert resolved.escalated_frame_id == 2
        assert resolved.alarm_activated_frame_id == 1
        assert resolved.responded_frame_id is None

    def test_resolved_entity_can_open_a_new_incident(self) -> None:
        manager = _manager()
        manager.update(0, [_intrusion(0, 0)])
        manager.record_response(0, 1)
        manager.resolve(0, 2)

        reopened = manager.update(3, [_intrusion(0, 3)])
        assert len(reopened) == 1
        assert reopened[0].incident_id == 1
        assert reopened[0].opened_frame_id == 3
        assert reopened[0].guard_notified_frame_id == 3
        assert reopened[0].state is IncidentState.GUARD_NOTIFIED

    def test_record_response_without_an_incident_raises(self) -> None:
        manager = _manager()
        with pytest.raises(
            LookupError, match="no active incident for entity_id 3"
        ):
            manager.record_response(3, 0)

    def test_record_response_twice_raises_instead_of_no_op(self) -> None:
        manager = _manager()
        manager.update(0, [_intrusion(0, 0)])
        manager.record_response(0, 1)

        with pytest.raises(
            LookupError,
            match="is in state RESPONDED, expected GUARD_NOTIFIED",
        ):
            manager.record_response(0, 2)

    def test_record_response_after_escalation_raises(self) -> None:
        manager = _manager(guard_response_budget=1)
        manager.update(0, [_intrusion(0, 0)])
        manager.update(1, [])

        with pytest.raises(
            LookupError, match="is in state ALARM_ACTIVATED, expected GUARD_NOTIFIED"
        ):
            manager.record_response(0, 2)

    def test_resolve_without_an_incident_raises(self) -> None:
        manager = _manager()
        with pytest.raises(LookupError, match="no active incident for entity_id 0"):
            manager.resolve(0, 0)

    def test_resolve_before_a_response_raises(self) -> None:
        manager = _manager()
        manager.update(0, [_intrusion(0, 0)])

        with pytest.raises(
            LookupError,
            match="is in state GUARD_NOTIFIED, expected RESPONDED or ESCALATED",
        ):
            manager.resolve(0, 1)

    def test_resolve_while_the_alarm_is_sounding_raises(self) -> None:
        manager = _manager(guard_response_budget=1)
        manager.update(0, [_intrusion(0, 0)])
        manager.update(1, [])

        with pytest.raises(
            LookupError, match="is in state ALARM_ACTIVATED, expected RESPONDED"
        ):
            manager.resolve(0, 2)

    def test_resolve_twice_raises_instead_of_no_op(self) -> None:
        manager = _manager()
        manager.update(0, [_intrusion(0, 0)])
        manager.record_response(0, 1)
        manager.resolve(0, 2)

        with pytest.raises(LookupError, match="no active incident for entity_id 0"):
            manager.resolve(0, 3)


class TestIndependentEntities:
    def test_two_entities_keep_independent_clocks(self) -> None:
        """Interleaved transitions prove each budget counts from its own transition.

        Budgets: guard 3, escalation 2. Entity 0 is notified on frame 0 and
        entity 1 on frame 4, so their transitions must land 4 frames apart:
        entity 0 alarms at 0 + 3 = 3 and escalates at 3 + 2 = 5; entity 1 alarms
        at 4 + 3 = 7 and escalates at 7 + 2 = 9.
        """
        manager = _manager(guard_response_budget=3, escalation_budget=2)
        log: list[tuple[int, int, str]] = []

        for frame_id in range(10):
            events: list[Event] = []
            if frame_id == 0:
                events.append(_intrusion(0, 0))
            if frame_id == 4:
                events.append(_intrusion(1, 4))
            for incident in manager.update(frame_id, events):
                log.append((frame_id, incident.entity_id, incident.state.name))

        assert log == [
            (0, 0, "GUARD_NOTIFIED"),
            (3, 0, "ALARM_ACTIVATED"),
            (4, 1, "GUARD_NOTIFIED"),
            (5, 0, "ESCALATED"),
            (7, 1, "ALARM_ACTIVATED"),
            (9, 1, "ESCALATED"),
        ]

    def test_answering_one_entity_does_not_disturb_another(self) -> None:
        manager = _manager(guard_response_budget=3, escalation_budget=2)
        manager.update(0, [_proximity(0, 1, 0)])

        # Entity 0 answers immediately; entity 1 is left to the alarm.
        assert manager.record_response(0, 1).state is IncidentState.RESPONDED
        assert manager.update(1, []) == []
        assert manager.update(2, []) == []

        # Entity 0 was notified on frame 0 and would have alarmed on frame 3,
        # but answering stopped its clock: only entity 1 alarms, on its own
        # schedule.
        activated = manager.update(3, [])
        assert [(i.entity_id, i.state) for i in activated] == [
            (1, IncidentState.ALARM_ACTIVATED)
        ]

        # Entity 0's incident stays RESPONDED and is never reported again:
        # every later report is entity 1's alone.
        assert manager.update(4, []) == []
        escalated = manager.update(5, [])
        assert [(i.entity_id, i.state) for i in escalated] == [
            (1, IncidentState.ESCALATED)
        ]
        for frame_id in range(6, 15):
            assert manager.update(frame_id, []) == []


class TestDeltaEmission:
    def test_update_returns_only_incidents_that_changed_state(self) -> None:
        manager = _manager(guard_response_budget=3, escalation_budget=2)
        assert len(manager.update(0, [_intrusion(0, 0)])) == 1

        # Frames 1 and 2: the incident sits in GUARD_NOTIFIED unchanged and must
        # not be re-emitted â€” same only-emit-deltas rule EventEngine follows.
        assert manager.update(1, []) == []
        assert manager.update(2, []) == []

        assert [i.state for i in manager.update(3, [])] == [
            IncidentState.ALARM_ACTIVATED
        ]
        assert manager.update(4, []) == []
        assert [i.state for i in manager.update(5, [])] == [IncidentState.ESCALATED]

        # ESCALATED has no counter, so it is never re-emitted either.
        for frame_id in range(6, 20):
            assert manager.update(frame_id, []) == []

    def test_responded_incident_is_never_re_emitted(self) -> None:
        manager = _manager(guard_response_budget=3)
        manager.update(0, [_intrusion(0, 0)])
        manager.record_response(0, 2)

        for frame_id in range(3, 20):
            assert manager.update(frame_id, []) == []

    def test_timeout_transitions_are_reported_before_new_incidents(self) -> None:
        manager = _manager(guard_response_budget=2, escalation_budget=1)
        manager.update(0, [_intrusion(0, 0)])
        manager.update(1, [_intrusion(1, 1)])

        # Frame 2 both escalates entity 0's timeout and opens entity 2's
        # incident; the timeout comes first, and entity 1 (notified on 1, budget
        # 2) is not yet due.
        changed = manager.update(2, [_intrusion(2, 2)])
        assert [(i.entity_id, i.state) for i in changed] == [
            (0, IncidentState.ALARM_ACTIVATED),
            (2, IncidentState.GUARD_NOTIFIED),
        ]

    def test_identical_input_produces_identical_output(self) -> None:
        def run() -> list[tuple[int, int, str]]:
            manager = _manager(guard_response_budget=2, escalation_budget=2)
            log: list[tuple[int, int, str]] = []
            for frame_id in range(12):
                events = [_intrusion(frame_id % 2, frame_id)] if frame_id < 4 else []
                for incident in manager.update(frame_id, events):
                    log.append((frame_id, incident.entity_id, incident.state.name))
            return log

        assert run() == run()


class TestValidation:
    @pytest.mark.parametrize("bad_budget", [0, -1])
    def test_constructor_rejects_non_positive_guard_response_budget(
        self, bad_budget: int
    ) -> None:
        with pytest.raises(ValueError, match="guard_response_budget"):
            IncidentManager(bad_budget, 2, lambda entity_id: False)

    @pytest.mark.parametrize("bad_budget", [0, -1])
    def test_constructor_rejects_non_positive_escalation_budget(
        self, bad_budget: int
    ) -> None:
        with pytest.raises(ValueError, match="escalation_budget"):
            IncidentManager(2, bad_budget, lambda entity_id: False)

    def test_update_rejects_a_repeated_frame_id(self) -> None:
        manager = _manager()
        manager.update(0, [_intrusion(0, 0)])
        with pytest.raises(
            ValueError, match=r"frame_id \(0\) must be > last updated frame_id \(0\)"
        ):
            manager.update(0, [])

    def test_update_rejects_a_decreasing_frame_id(self) -> None:
        manager = _manager()
        manager.update(5, [_intrusion(0, 5)])
        with pytest.raises(
            ValueError, match=r"frame_id \(3\) must be > last updated frame_id \(5\)"
        ):
            manager.update(3, [])

    def test_hooks_accept_frame_ids_the_ordering_invariant_allows(self) -> None:
        """Hooks are out-of-band: they are exempt from update's monotonicity.

        A guard action is timestamped by when it was reported, not by the
        pipeline's frame cursor, so it may carry a frame at or before the last
        ``update`` — only ``Incident``'s own ordering invariant polices it.
        """
        manager = _manager(guard_response_budget=3)
        manager.update(5, [_intrusion(0, 5)])
        manager.update(6, [])
        manager.update(9, [])

        responded = manager.record_response(0, 7)
        assert responded.state is IncidentState.RESPONDED
        assert responded.responded_frame_id == 7

        resolved = manager.resolve(0, 7)
        assert resolved.state is IncidentState.RESOLVED
        assert resolved.resolved_frame_id == 7

    def test_hook_frame_before_the_notification_is_rejected(self) -> None:
        manager = _manager(guard_response_budget=10)
        manager.update(5, [_intrusion(0, 5)])

        with pytest.raises(
            ValueError,
            match=r"responded_frame_id \(3\) must be >= guard_notified_frame_id \(5\)",
        ):
            manager.record_response(0, 3)

    def test_update_rejects_negative_frame_id(self) -> None:
        manager = _manager()
        with pytest.raises(ValueError, match="frame_id"):
            manager.update(-1, [])

    def test_record_response_rejects_negative_frame_id(self) -> None:
        manager = _manager()
        manager.update(0, [_intrusion(0, 0)])
        with pytest.raises(ValueError, match="frame_id"):
            manager.record_response(0, -1)

    def test_resolve_rejects_negative_frame_id(self) -> None:
        manager = _manager()
        manager.update(0, [_intrusion(0, 0)])
        manager.record_response(0, 1)
        with pytest.raises(ValueError, match="frame_id"):
            manager.resolve(0, -1)
