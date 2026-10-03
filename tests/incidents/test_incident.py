"""Contract tests for the Incident dataclass (ADR-0012).

Every invariant in ``Incident.__post_init__`` is exercised here: non-negative
ids and the opened frame id, the state -> required-frame-fields table from the
``Incident`` docstring (each field tested *both* ways: present when its state
has not been reached, and absent when it has), and the ``>= opened_frame_id``
frame ordering. A malformed incident is a data-quality bug and must be rejected
at construction.
"""

import pytest

from sentinel_vision.events.event import EventType
from sentinel_vision.incidents.incident import Incident, IncidentState

_FRAME_FIELDS: tuple[str, ...] = (
    "opened_frame_id",
    "guard_notified_frame_id",
    "responded_frame_id",
    "alarm_activated_frame_id",
    "escalated_frame_id",
    "resolved_frame_id",
)

# The state -> required-frame-fields table, restated here from the contract's
# docstring. One entry per legal incident shape; RESOLVED appears twice because
# the responded path and the escalation path leave different fields set.
_SHAPES: tuple[tuple[str, IncidentState, dict[str, int]], ...] = (
    ("open", IncidentState.OPEN, {"opened_frame_id": 10}),
    (
        "guard-notified",
        IncidentState.GUARD_NOTIFIED,
        {"opened_frame_id": 10, "guard_notified_frame_id": 10},
    ),
    (
        "responded",
        IncidentState.RESPONDED,
        {
            "opened_frame_id": 10,
            "guard_notified_frame_id": 10,
            "responded_frame_id": 14,
        },
    ),
    (
        "alarm",
        IncidentState.ALARM_ACTIVATED,
        {"opened_frame_id": 10, "guard_notified_frame_id": 10, "alarm_activated_frame_id": 13},
    ),
    (
        "escalated",
        IncidentState.ESCALATED,
        {
            "opened_frame_id": 10,
            "guard_notified_frame_id": 10,
            "alarm_activated_frame_id": 13,
            "escalated_frame_id": 15,
        },
    ),
    (
        "resolved-after-response",
        IncidentState.RESOLVED,
        {
            "opened_frame_id": 10,
            "guard_notified_frame_id": 10,
            "responded_frame_id": 14,
            "resolved_frame_id": 20,
        },
    ),
    (
        "resolved-after-escalation",
        IncidentState.RESOLVED,
        {
            "opened_frame_id": 10,
            "guard_notified_frame_id": 10,
            "alarm_activated_frame_id": 13,
            "escalated_frame_id": 15,
            "resolved_frame_id": 20,
        },
    ),
)


_BRANCH_FIELDS: tuple[str, ...] = ("responded_frame_id", "alarm_activated_frame_id")

# (shape label, field) pairs whose violation has a branch-specific message
# rather than the generic presence message, and is asserted on its own below.
_COVERED_BY_DEDICATED_TEST: frozenset[tuple[str, str]] = frozenset(
    {
        # A RESOLVED incident whose branch marker was dropped has no marker left
        # at all, which is a different mistake than a missing field on a known
        # branch — both cases are asserted by the dedicated test below.
        ("resolved-after-response", "responded_frame_id"),
        ("resolved-after-escalation", "alarm_activated_frame_id"),
    }
)


def _sets_both_branches(fields: dict[str, int], name: str) -> bool:
    """True if adding ``name`` to ``fields`` would leave both branches recorded."""
    return name in _BRANCH_FIELDS and any(
        branch in fields for branch in _BRANCH_FIELDS
    )


def _incident(
    state: IncidentState,
    *,
    incident_id: int = 0,
    entity_id: int = 0,
    **frame_fields: int | None,
) -> Incident:
    """Build an incident whose frame fields are exactly ``frame_fields``.

    Frame fields not named default to ``None`` except ``opened_frame_id``
    (10), so a test names only the fields it is about.
    """
    fields: dict[str, int | None] = {
        "opened_frame_id": 10,
        "guard_notified_frame_id": None,
        "responded_frame_id": None,
        "alarm_activated_frame_id": None,
        "escalated_frame_id": None,
        "resolved_frame_id": None,
    }
    fields.update(frame_fields)
    return Incident(
        incident_id=incident_id,
        entity_id=entity_id,
        triggering_event_type=EventType.ZONE_INTRUSION,
        state=state,
        opened_frame_id=fields["opened_frame_id"],
        guard_notified_frame_id=fields["guard_notified_frame_id"],
        responded_frame_id=fields["responded_frame_id"],
        alarm_activated_frame_id=fields["alarm_activated_frame_id"],
        escalated_frame_id=fields["escalated_frame_id"],
        resolved_frame_id=fields["resolved_frame_id"],
    )


class TestValidIncidents:
    @pytest.mark.parametrize(
        ("state", "fields"), [(state, fields) for _, state, fields in _SHAPES]
    )
    def test_every_state_shape_constructs(
        self, state: IncidentState, fields: dict[str, int]
    ) -> None:
        incident = _incident(state, **fields)
        assert incident.state is state
        assert incident.triggering_event_type is EventType.ZONE_INTRUSION

    def test_same_frame_guard_notification_is_legal(self) -> None:
        """``>= opened_frame_id``, not ``>``: notification happens on the opening frame."""
        incident = _incident(
            IncidentState.GUARD_NOTIFIED,
            opened_frame_id=10,
            guard_notified_frame_id=10,
        )
        assert incident.opened_frame_id == incident.guard_notified_frame_id

    def test_proximity_event_type_is_carried_through(self) -> None:
        incident = Incident(
            incident_id=7,
            entity_id=2,
            triggering_event_type=EventType.PROXIMITY_HAZARD,
            state=IncidentState.OPEN,
            opened_frame_id=0,
            guard_notified_frame_id=None,
            responded_frame_id=None,
            alarm_activated_frame_id=None,
            escalated_frame_id=None,
            resolved_frame_id=None,
        )
        assert incident.incident_id == 7
        assert incident.triggering_event_type is EventType.PROXIMITY_HAZARD


class TestFieldPresenceBothWays:
    @pytest.mark.parametrize(
        ("state", "fields", "name"),
        [
            pytest.param(state, fields, name, id=f"{label}-{name}")
            for label, state, fields in _SHAPES
            for name in fields
            if name != "opened_frame_id"
            and (label, name) not in _COVERED_BY_DEDICATED_TEST
        ],
    )
    def test_reached_state_missing_frame_field_is_rejected(
        self, state: IncidentState, fields: dict[str, int], name: str
    ) -> None:
        overrides: dict[str, int | None] = dict(fields)
        overrides[name] = None
        with pytest.raises(
            ValueError, match=f"{name} cannot be None when state is {state.name}"
        ):
            _incident(state, **overrides)

    @pytest.mark.parametrize(
        ("state", "fields", "name"),
        [
            pytest.param(state, fields, name, id=f"{label}-{name}")
            for label, state, fields in _SHAPES
            for name in _FRAME_FIELDS
            if name not in fields
            and not _sets_both_branches(fields, name)
            and (label, name) not in _COVERED_BY_DEDICATED_TEST
        ],
    )
    def test_unreached_state_carrying_frame_field_is_rejected(
        self, state: IncidentState, fields: dict[str, int], name: str
    ) -> None:
        overrides: dict[str, int | None] = dict(fields)
        overrides[name] = 20
        with pytest.raises(
            ValueError, match=f"{name} must be None when state is {state.name}"
        ):
            _incident(state, **overrides)


class TestFrameOrdering:
    @pytest.mark.parametrize(
        ("state", "fields", "name"),
        [
            pytest.param(state, fields, name, id=f"{label}-{name}")
            for label, state, fields in _SHAPES
            for name in fields
            if name != "opened_frame_id"
        ],
    )
    def test_frame_field_before_its_path_predecessor_is_rejected(
        self, state: IncidentState, fields: dict[str, int], name: str
    ) -> None:
        """Each shape's fields are written in path order, so the predecessor is the prior key."""
        predecessor = list(fields)[list(fields).index(name) - 1]
        predecessor_frame_id = fields[predecessor]
        overrides = dict(fields)
        overrides[name] = predecessor_frame_id - 1
        with pytest.raises(
            ValueError,
            match=(
                f"{name} \\({predecessor_frame_id - 1}\\) must be >= "
                f"{predecessor} \\({predecessor_frame_id}\\)"
            ),
        ):
            _incident(state, **overrides)

    def test_resolve_before_respond_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="resolved_frame_id \\(8\\) must be >= responded_frame_id \\(10\\)",
        ):
            _incident(
                IncidentState.RESOLVED,
                opened_frame_id=10,
                guard_notified_frame_id=10,
                responded_frame_id=10,
                resolved_frame_id=8,
            )

    def test_escalate_before_alarm_is_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="escalated_frame_id \\(11\\) must be >= alarm_activated_frame_id \\(13\\)",
        ):
            _incident(
                IncidentState.ESCALATED,
                opened_frame_id=10,
                guard_notified_frame_id=10,
                alarm_activated_frame_id=13,
                escalated_frame_id=11,
            )

    def test_equal_frame_ids_are_accepted_on_the_responded_path(self) -> None:
        """Opened, notified, responded and resolved all on one frame is legal."""
        incident = _incident(
            IncidentState.RESOLVED,
            opened_frame_id=10,
            guard_notified_frame_id=10,
            responded_frame_id=10,
            resolved_frame_id=10,
        )
        assert incident.opened_frame_id == 10
        assert incident.responded_frame_id == incident.resolved_frame_id == 10

    def test_equal_frame_ids_are_accepted_on_the_escalation_path(self) -> None:
        incident = _incident(
            IncidentState.RESOLVED,
            opened_frame_id=4,
            guard_notified_frame_id=4,
            alarm_activated_frame_id=4,
            escalated_frame_id=4,
            resolved_frame_id=4,
        )
        assert incident.alarm_activated_frame_id == incident.escalated_frame_id == 4

    def test_respond_on_the_notification_frame_is_accepted(self) -> None:
        incident = _incident(
            IncidentState.RESPONDED,
            opened_frame_id=10,
            guard_notified_frame_id=10,
            responded_frame_id=10,
        )
        assert incident.responded_frame_id == incident.guard_notified_frame_id

    def test_responded_and_alarm_frames_are_mutually_exclusive(self) -> None:
        """A guard response and an alarm escalation are the two branches, not both."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            _incident(
                IncidentState.ESCALATED,
                opened_frame_id=10,
                guard_notified_frame_id=10,
                responded_frame_id=14,
                alarm_activated_frame_id=13,
                escalated_frame_id=15,
            )

    @pytest.mark.parametrize(
        ("state", "fields"),
        [
            pytest.param(state, fields, id=label)
            for label, state, fields in _SHAPES
        ],
    )
    def test_both_branches_recorded_is_rejected_in_every_state(
        self, state: IncidentState, fields: dict[str, int]
    ) -> None:
        overrides: dict[str, int | None] = dict(fields)
        for branch in _BRANCH_FIELDS:
            overrides[branch] = 20
        with pytest.raises(ValueError, match="mutually exclusive"):
            _incident(state, **overrides)

    def test_resolved_incident_without_a_branch_marker_is_rejected(self) -> None:
        """RESOLVED claims which branch it took; with neither marker it claims nothing."""
        with pytest.raises(ValueError, match="must record which branch it took"):
            _incident(
                IncidentState.RESOLVED,
                opened_frame_id=10,
                guard_notified_frame_id=10,
                resolved_frame_id=20,
            )


class TestIdValidation:
    def test_rejects_negative_incident_id(self) -> None:
        with pytest.raises(ValueError, match="incident_id"):
            _incident(
                IncidentState.OPEN,
                incident_id=-1,
                opened_frame_id=10,
                guard_notified_frame_id=None,
            )

    def test_rejects_negative_entity_id(self) -> None:
        with pytest.raises(ValueError, match="entity_id"):
            _incident(
                IncidentState.OPEN,
                entity_id=-1,
                opened_frame_id=10,
                guard_notified_frame_id=None,
            )

    def test_rejects_negative_opened_frame_id(self) -> None:
        with pytest.raises(ValueError, match="opened_frame_id"):
            _incident(
                IncidentState.OPEN,
                opened_frame_id=-1,
                guard_notified_frame_id=None,
            )