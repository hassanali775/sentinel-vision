# ADR-0008: Spatial Workspace Model (Facts, Not Violations)

## Status

Accepted

## Context

PR-006 introduced persistent entities with explicit lifecycle states (`VISIBLE`, `OCCLUDED`, `PREDICTED`, `LOST`, `RETIRED`), and PR-007 added spatial re-identification for reappearing entities. Before temporal/threshold-based event reasoning lands (PR-009), the pipeline needs a per-frame **spatial model**: where entities are relative to user-defined zones and to each other.

The load-bearing boundary for PR-008 is that it computes **descriptive geometric facts only** — zone membership and pairwise distances — and nothing else. No thresholds, no temporal accumulation, no "violation" concept. A zone becoming occupied, or two entities getting close, is recorded as a fact; whether that fact *matters* is a judgment that PR-009 makes.

## Decision

### 1. Pixel-Space Limitation: Not Real-World Calibrated

All zone geometry and all distances are expressed in **raw image-pixel coordinates**. A box center 40 pixels from a zone boundary, or two entities 50 pixels apart, are pixel-space quantities; they carry no physical unit.

- A "proximity" or "intrusion" conclusion built directly on these numbers is **not yet physically calibrated**. Nothing in this PR converts pixels to meters, inches, or any real-world metric.
- Real-world-metric proximity requires camera calibration (intrinsics/extrinsics, distortion) or a planar homography to ground plane coordinates. That work is a **future camera-calibration/homography PR**, explicitly out of scope here. Until then, pixel-space facts are a faithful, deterministic intermediate representation that PR-009 may combine with calibration-derived scale factors.

### 2. Center-Point Zone Membership vs. Full Polygon Overlap

Zone membership is decided by a single point: the entity's bounding-box center, tested with `Zone.contains_point`. It is **not** decided by the box's full footprint or polygon-polygon overlap with the zone.

- **Known simplification**: a box that partially overlaps a zone boundary without its center crossing in or out is not detected as entering or leaving the zone. A large object straddling a fence-line zone while its center sits outside will report *not in zone*.
- This is a deliberate simplicity trade-off, not an oversight: center-point membership is one ray-cast per entity per zone, has a single well-defined on/off transition per zone, and no consumer yet needs overlap fractions. Polygon-polygon overlap and partial-membership semantics are deferred until a PR that actually reasons about them demands it.

### 3. LOST Entities Excluded from All Spatial Facts

`evaluate` filters to entities whose observation state is `VISIBLE`, `OCCLUDED`, or `PREDICTED` — the only states that carry an active bounding box (ADR-0006 enforces `bounding_box is None` if and only if `LOST`).

- `LOST` entities contribute **no** zone membership and **no** pairwise distance: with no bounding box there is no geometry to reason about, and inventing one would smuggle an estimate into a facts layer that must stay descriptive.
- `RETIRED` entities are excluded too: although they carry a final box, they are terminal bookkeeping records purged from the tracker, not active spatial entities. This is a documented decision, not an omission.

### 4. Facts-Not-Violations Boundary with PR-009

This PR emits no temporal judgment, no thresholds, and no violation concept whatsoever. `SpatialFrameObservation` contains exactly two fact kinds — `zone_memberships` and `pairwise_distances` — both per frame, both derived only from that frame's eligible observations.

PR-009 will be the first layer allowed to:
- apply thresholds (e.g. "distance < N pixels"),
- accumulate over time (e.g. "inside zone for M consecutive frames"),
- and produce judgments (e.g. "intrusion", "proximity hazard").

Nothing in PR-008's data model anticipates, permits, or half-implements those concepts; keeping the facts layer pure is what makes the judgment layer independently testable against known geometry.

## Consequences

- Downstream layers (PR-009 event reasoning, PR-010 audit logging, forensic visualization) consume deterministic, per-frame geometric facts with an explicit pixel-space caveat.
- Membership is deterministic and cheap: one even-odd ray-cast per (entity, zone) pair, pure Python, no geometry dependency.
- Pairwise distance lookup is order-independent by construction: keys are always `(min(entity_id_a, entity_id_b), max(entity_id_a, entity_id_b))`.
- The facts/violations boundary is architectural: any "violation" that ever appears in the codebase must be traced to PR-009, not silently added here.

## Alternatives Considered

- **Polygon-polygon overlap for membership**: Rejected for PR-008 because center-point membership satisfies no consumer yet and avoids ambiguity in entry/exit semantics. Deferred with the partial-membership caveat documented above.
- **Shapely / geometry libraries**: Rejected. The pipeline's dependency ceiling is NumPy plus the standard library (ADR-0002, ADR-0003), and an even-odd ray cast is ~10 lines, deterministic, and trivially unit-testable.
- **Emitting violations in PR-008**: Rejected on principle — it would blur the facts/violations boundary and make PR-009's judgments untestable against a clean geometric substrate.
- **Real-world units now**: Rejected. Without camera calibration the numbers would be fabricated; pixel-space facts are the honest intermediate representation.
