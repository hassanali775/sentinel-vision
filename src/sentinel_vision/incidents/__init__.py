"""Deterministic incident state machine package (PR-012).

One step above the event engine: ``IncidentManager`` turns the events of a frame
into security incidents for unauthorized entities and drives each incident
through a deterministic lifecycle (notify guard, respond, alarm, escalate,
resolve) bounded by frame-counted timeout budgets. See
docs/adr/0012-deterministic-incident-state-machine.md.
"""

__all__ = [
    "Incident",
    "IncidentManager",
    "IncidentState",
]

from sentinel_vision.incidents.incident import Incident, IncidentState
from sentinel_vision.incidents.manager import IncidentManager