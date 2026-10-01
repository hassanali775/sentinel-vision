"""Incident contracts for the deterministic incident state machine (ADR-0012).

This module defines the ``IncidentState`` lifecycle enum and the frozen
``Incident`` contract representing a security incident's state at one point in
its lifecycle. An incident is one step above an ``Event`` (PR-009): the event
engine judges that *something* is happening, this contract records what the
system is *doing about it* — notifying the guard, activating an alarm,
escalating, resolving.

``EventType`` is deliberately imported from ``sentinel_vision.events.event``
rather than redefined: the incident layer's only input vocabulary is the event
layer's, and a second ``EventType`` enum would let an incident name a triggering
condition the event engine cannot produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sentinel_vision.events.event import EventType


class IncidentState(Enum):
    """Lifecycle states of a security incident, in escalation order.

    Two paths run from ``GUARD_NOTIFIED`` to ``RESOLVED``: a guard answers
    (``RESPONDED``) and the incident is resolved, or the guard does not answer
    and the system escalates by itself (``ALARM_ACTIVATED`` then ``ESCALATED``).
    """

    OPEN = "OPEN"
    GUARD_NOTIFIED = "GUARD_NOTIFIED"
    RESPONDED = "RESPONDED"
    ALARM_ACTIVATED = "ALARM_ACTIVATED"
    ESCALATED = "ESCALATED"
    RESOLVED = "RESOLVED"


# The two legal paths through the lifecycle. ``OPEN`` and ``GUARD_NOTIFIED`` are
# common to both; the branches diverge at the third state and rejoin at
# ``RESOLVED``. A transition path is either one of these two tuples, which is
# what makes "which frame ids must be present" a finite check rather than a
# pile of per-state special cases.
_RESPONDED_PATH: tuple[IncidentState, ...] = (
    IncidentState.OPEN,
    IncidentState.GUARD_NOTIFIED,
    IncidentState.RESPONDED,
    IncidentState.RESOLVED,
)

_ESCALATION_PATH: tuple[IncidentState, ...] = (
    IncidentState.OPEN,
    IncidentState.GUARD_NOTIFIED,
    IncidentState.ALARM_ACTIVATED,
    IncidentState.ESCALATED,
    IncidentState.RESOLVED,
)

# The frame id that records entering each state. ``opened_frame_id`` is separate
# because it is present from the very first state.
_FRAME_FIELD_BY_STATE: dict[IncidentState, str] = {
    IncidentState.OPEN: "",
    IncidentState.GUARD_NOTIFIED: "guard_notified_frame_id",
    IncidentState.RESPONDED: "responded_frame_id",
    IncidentState.ALARM_ACTIVATED: "alarm_activated_frame_id",
    IncidentState.ESCALATED: "escalated_frame_id",
    IncidentState.RESOLVED: "resolved_frame_id",
}

_ALL_FRAME_FIELDS: tuple[str, ...] = (
    "opened_frame_id",
    "guard_notified_frame_id",
    "responded_frame_id",
    "alarm_activated_frame_id",
    "escalated_frame_id",
    "resolved_frame_id",
)


@dataclass(frozen=True)
class Incident:
    """A security incident's state, from the event that triggered it to resolution.

    Attributes:
        incident_id: Monotonic id, minted from 0 in open order. Non-negative.
        entity_id: The entity this incident is about. Non-negative.
        triggering_event_type: The kind of event that opened the incident.
            Reuses the event engine's ``EventType``.
        state: The current lifecycle state.
        opened_frame_id: The frame the incident opened on. Non-negative.
        guard_notified_frame_id: The frame the guard was notified on.
        responded_frame_id: The frame a guard was recorded as responding on.
        alarm_activated_frame_id: The frame the alarm was activated on.
        escalated_frame_id: The frame the incident was escalated on.
        resolved_frame_id: The frame the incident was resolved on.

    Field-presence rules — ``state`` determines exactly which frame ids may be
    non-``None``. Every frame id for a state already reached must be non-``None``
    and ``>= opened_frame_id``; every frame id for a state not yet reached must
    be ``None``. Both directions are enforced, mirroring ``EntityObservation``'s
    LOST/box invariant (ADR-0006): an incident that carries a response frame
    without being in ``RESPONDED`` is claiming a transition that never happened.

    =================  ====================================================
    State              Required frame fields (all others must be ``None``)
    =================  ====================================================
    OPEN               ``opened_frame_id``
    GUARD_NOTIFIED     ``opened_frame_id``, ``guard_notified_frame_id``
    RESPONDED          ``opened_frame_id``, ``guard_notified_frame_id``,
                       ``responded_frame_id``
    ALARM_ACTIVATED    ``opened_frame_id``, ``guard_notified_frame_id``,
                       ``alarm_activated_frame_id``
    ESCALATED          ``opened_frame_id``, ``guard_notified_frame_id``,
                       ``alarm_activated_frame_id``, ``escalated_frame_id``
    RESOLVED           either the responded path — ``opened_frame_id``,
                       ``guard_notified_frame_id``, ``responded_frame_id``,
                       ``resolved_frame_id`` — or the escalation path —
                       ``opened_frame_id``, ``guard_notified_frame_id``,
                       ``alarm_activated_frame_id``, ``escalated_frame_id``,
                       ``resolved_frame_id``. ``responded_frame_id`` and
                       ``alarm_activated_frame_id`` are mutually exclusive:
                       exactly one branch may have been taken.
    =================  ====================================================

    Other invariants validated in ``__post_init__``:
    - ``incident_id``, ``entity_id``, and ``opened_frame_id`` must be non-negative.
    - ``responded_frame_id`` and ``alarm_activated_frame_id`` are mutually
      exclusive — an incident is either handled by a guard or escalated, never
      both — and a ``RESOLVED`` incident must record which of the two happened.
    - Every non-``None`` frame id must be ``>= opened_frame_id``. The bound is
      ``>=`` and not ``>`` because guard notification happens on the same frame
      the incident opens (ADR-0012's synchronous-notification simplification),
      so ``guard_notified_frame_id == opened_frame_id`` is legal and expected.
    """

    incident_id: int
    entity_id: int
    triggering_event_type: EventType
    state: IncidentState
    opened_frame_id: int
    guard_notified_frame_id: int | None
    responded_frame_id: int | None
    alarm_activated_frame_id: int | None
    escalated_frame_id: int | None
    resolved_frame_id: int | None

    def __post_init__(self) -> None:
        if self.incident_id < 0:
            raise ValueError(
                f"incident_id ({self.incident_id}) must be non-negative"
            )
        if self.entity_id < 0:
            raise ValueError(f"entity_id ({self.entity_id}) must be non-negative")
        if self.opened_frame_id < 0:
            raise ValueError(
                f"opened_frame_id ({self.opened_frame_id}) must be non-negative"
            )

        if (
            self.responded_frame_id is not None
            and self.alarm_activated_frame_id is not None
        ):
            raise ValueError(
                "responded_frame_id and alarm_activated_frame_id are mutually "
                "exclusive: an incident is either handled by a guard or escalated, "
                "never both"
            )
        if (
            self.state is IncidentState.RESOLVED
            and self.responded_frame_id is None
            and self.alarm_activated_frame_id is None
        ):
            raise ValueError(
                "a RESOLVED incident must record which branch it took: "
                "responded_frame_id (a guard handled it) or alarm_activated_frame_id "
                "with escalated_frame_id (escalation handled it)"
            )

        path = self._transition_path()
        required = self._required_frame_fields(path)
        present = {
            name for name in _ALL_FRAME_FIELDS if getattr(self, name) is not None
        }

        for name in sorted(required - present):
            raise ValueError(
                f"{name} cannot be None when state is {self.state.name}"
            )
        for name in sorted(present - required):
            raise ValueError(
                f"{name} must be None when state is {self.state.name}"
            )

        for name in sorted(required):
            frame_id: int = getattr(self, name)
            if frame_id < self.opened_frame_id:
                raise ValueError(
                    f"{name} ({frame_id}) must be >= opened_frame_id "
                    f"({self.opened_frame_id})"
                )

    def _transition_path(self) -> tuple[IncidentState, ...]:
        """Return the transition path this incident is on.

        The branch is decided by which terminal-branch frame ids are set, not by
        ``state``: an incident that has been answered is on the responded path
        even after it resolves, and an incident whose state and branch markers
        disagree is a contract violation that the field-presence checks below
        must report against the branch its state belongs to.
        """
        if self.responded_frame_id is not None or self.state is IncidentState.RESPONDED:
            return _RESPONDED_PATH
        return _ESCALATION_PATH

    def _required_frame_fields(self, path: tuple[IncidentState, ...]) -> set[str]:
        """Return the frame field names required for the current state on ``path``."""
        if self.state not in path:
            # State belongs to the other branch (e.g. ESCALATED on the
            # responded path): report against the branch the state is on, so the
            # error names the frame id that is actually missing.
            path = _ESCALATION_PATH if self.state in _ESCALATION_PATH else _RESPONDED_PATH
        reached = path[: path.index(self.state) + 1]
        required = {"opened_frame_id"}
        for reached_state in reached:
            field_name = _FRAME_FIELD_BY_STATE[reached_state]
            if field_name:
                required.add(field_name)
        return required