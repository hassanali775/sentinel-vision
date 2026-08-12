"""Persistent entity state management package (PR-006).

Provides entity lifecycle management (``EntityState``, ``EntityObservation``)
and ``PersistentEntityTracker`` to wrap raw frame-level tracking into persistent
entity observations across 5 lifecycle states (VISIBLE, OCCLUDED, PREDICTED,
LOST, RETIRED).
"""

__all__ = [
    "EntityObservation",
    "EntityState",
    "PersistentEntityTracker",
]

from sentinel_vision.state.entity import EntityObservation, EntityState
from sentinel_vision.state.tracker import PersistentEntityTracker
