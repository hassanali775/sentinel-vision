# ADR-0006: Persistent Entity State and Occlusion Management

## Status

Accepted

## Context

PR-005 provided raw frame-to-frame data association (`TrackedDetection`). Raw frame-level association track IDs, however, are transient frame-association tags. Downstream event reasoning (PR-009) and spatial workspace modeling (PR-008) require a stable concept of an **entity** with explicit lifecycle states (e.g. knowing whether an object is currently visible, temporarily occluded, estimated by motion extrapolation, lost, or retired).

PR-006 introduces `PersistentEntityTracker` and the frozen `EntityObservation` contract to manage persistent entity state through a 5-state lifecycle: `VISIBLE`, `OCCLUDED`, `PREDICTED`, `LOST`, and `RETIRED`.

## Decision

### 1. 5-State Entity Lifecycle Contract

`PersistentEntityTracker` consumes `TrackedDetection` objects and emits `EntityObservation` objects classified into five lifecycle states:

- **`VISIBLE`**: Matched with an incoming detection. `bounding_box` is the detected box.
- **`OCCLUDED`**: Unobserved for $k \le \text{occlusion\_budget}$ frames. `bounding_box` is held frozen at the last known observed box.
- **`PREDICTED`**: Unobserved for $\text{occlusion\_budget} < k \le \text{prediction\_budget}$ frames. `bounding_box` is linearly extrapolated using per-frame velocity $(box_{t-1} - box_{t-2})$.
  - *Single-observation fallback*: If only one observed position exists before occlusion (no velocity history), extrapolation falls back to holding the last known box.
- **`LOST`**: Unobserved for $\text{prediction\_budget} < k \le \text{retirement\_budget}$ frames. `bounding_box` is strictly `None`.
- **`RETIRED`**: Terminal state when unobserved frames $k > \text{retirement\_budget}$. Emits a final `RETIRED` observation (with last known box) once, then the entity is purged from internal memory.

### 2. Tracker-Composition Contract and Upstream `max_age` Bound

`PersistentEntityTracker` delegates frame-to-frame association to an upstream `BaseTracker` (e.g. `GreedyIoUTracker`).

- **Assumption**: The upstream `BaseTracker`'s `max_age` must be configured $\ge$ `PersistentEntityTracker`'s total budget (`retirement_budget`).
- **Limitation**: If the upstream tracker drops a track and reissues a new `track_id` earlier than `retirement_budget`, `PersistentEntityTracker` receives an unrecognized `track_id` and creates a new entity. The original entity continues toward `LOST`/`RETIRED` independently. Reconnecting different `track_id`s across temporal gaps is explicitly the responsibility of re-identification (PR-007). This limitation is tested and documented.

### 3. Linear Finite-Difference Extrapolation over Kalman Filtering

`PREDICTED` uses linear finite-difference extrapolation based on the last two observed bounding boxes rather than a Kalman filter.

- **Rationale**: Keeps the baseline minimal and deterministic. On synthetic and simple industrial movement, linear extrapolation isolates lifecycle state evaluation. Motion modeling with process/measurement noise is deferred until non-synthetic dynamics demand it.

### 4. Purging RETIRED Entities vs. Unlimited Retention

RETIRED entities are purged from memory after emitting a single `RETIRED` observation.

- **Rationale**: Prevents unbounded memory growth in continuous streams. Re-identification (PR-007) will maintain its own separate bounded window of recently retired entities for appearance feature matching.

## Consequences

- Event reasoning (PR-009) can rely on typed `EntityObservation` states to distinguish actual sensor visibility from estimated occlusion or lost objects.
- Invariant enforcement ensures `bounding_box is None` if and only if `state == EntityState.LOST`.
- Upstream tracker configuration (`max_age`) must be aligned with downstream entity budgets.

## Alternatives Considered

- **Kalman Filter for PREDICTED state**: Rejected for PR-006 to avoid introducing tuning parameters and matrix math dependencies before real motion dynamics require them.
- **Retaining RETIRED entities indefinitely**: Rejected due to memory bounds for long-running monitoring.
