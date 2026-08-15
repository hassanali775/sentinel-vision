# ADR-0009: Deterministic Event Engine (First Judgment Layer)

## Status

Accepted

## Context

PR-006 introduced persistent entities with lifecycle budgets, PR-007 spatial re-identification, and PR-008 a per-frame spatial workspace model of pure geometric facts (`zone_memberships`, `pairwise_distances`). Nothing yet converts those facts into a *judgment* — a statement that something worth paying attention to happened, when it started, and when it ended.

This PR is the boundary ADR-0008 reserved: the first layer allowed to apply thresholds, accumulate over time, and produce events. It is also the last deterministic-core PR before PR-010 (audit trail) and the future VLM layer, so the judgment semantics it establishes are the ones downstream consumers will be built against.

## Decision

### 1. Events, Not Violations, via Hysteresis

`EventEngine` converts per-frame spatial facts into events through two consecutive-frame budgets per rule:

- **`sustain_frames`**: a condition must hold for that many *consecutive* frames before an event opens.
- **`clear_frames`**: once open, a condition must be absent for that many *consecutive* frames before the event closes.

This is a direct reapplication of the budget pattern already established in PR-006: `PersistentEntityTracker` decides OCCLUDED/PREDICTED/LOST/RETIRED by counting consecutive unobserved frames, and PR-009 decides event open/close by counting consecutive true/absent frames. The same "N consecutive frames" reasoning, applied to judgments. A single noisy frame neither opens an event (needs `sustain_frames`) nor closes one (needs `clear_frames`); a brief false blip inside an open event resets its clear streak (the flicker guarantee), so the event stays open.

Each rule is registered as `(rule, sustain_frames, clear_frames)`, giving per-rule tuning. Per-rule keys are tracked independently: a proximity rule may hold several concurrent pair events, each at its own point in its sustain/clear cycle.

### 2. Stateless Rules, Stateful Engine

Rules are stateless per-frame predicates: `condition_holds(spatial) -> set[key]`, where a proximity key is an order-normalized `(min_id, max_id)` pair and an intrusion key is an entity id. All temporal memory lives in the engine's per-rule counter state. This mirrors the facts layer's discipline — a rule answers "does the condition hold this frame?", never "has it been holding?", and is therefore trivially unit-testable against known geometry.

The engine is deterministic by construction: rules evaluate in registration order and keys iterate in sorted order, so identical input produces identical output every run (ADR-0001).

### 3. Two Event Types Only: Proximity Hazard and Zone Intrusion

- **`PROXIMITY_HAZARD`**: an unordered pair of entities whose center distance is strictly below a pixel-space threshold. Strict `<` (not `<=`) pins the open/close boundary.
- **`ZONE_INTRUSION`**: an entity whose bounding-box center is inside a named zone.

Both inherit PR-008's center-point and pixel-space conventions unchanged. The `Event` contract is frozen and fully validated: entity-count and `zone_name` shape per type, non-negative ids and frame ids, and the open/closed frame ordering.

### 4. Pixel-Space Limitation Inherited and Restated

This PR is where a pixel-space quantity first becomes a *safety judgment* ("hazard"). ADR-0008's caveat therefore needs restating here, because the stakes are higher:

- A "proximity hazard" means *closer than N pixels*, not closer than N meters. Nothing in this PR converts pixels to physical units; real-world-metric judgments require camera calibration, which remains a future homography/calibration PR.
- A zone boundary is a line in image coordinates, and intrusion is decided by a box center, not full footprint overlap (ADR-0008 §2).
- Until calibration lands, event thresholds are operational constants to be tuned per deployment, not physical guarantees.

### 5. Deterministic Engine Is the Source of Truth; VLM Is Advisory

Per ADR-0001, the deterministic event engine is authoritative and every event is traceable to frames, detections, and explicit logic. The future VLM layer (PR-012 and PR-013) is strictly advisory and asynchronous: it may suggest events or narratives, but a VLM-sourced claim must be labeled as such and, where possible, verified against deterministic state. No PR before PR-012 may introduce a model call as a substitute for this engine.

### 6. Deferred Deliberately

- **Severity / prioritization**: an event is opened/closed, not "urgent" or "low". Ranking is deferred until a consumer needs it.
- **Compound events**: e.g. "intrusion while already in proximity" or event aggregation. The engine emits one event per (rule, key) occurrence.
- **Acknowledgment / dismissal**: who acted on an event and when. This belongs to the audit-trail PR (PR-010), not here.

## Consequences

- Events carry exact open/close frame ids, so duration is a simple difference and everything is auditable against the frames that produced it.
- Hysteresis parameters are per-rule operational constants; picking them is a tuning concern, not an architecture one.
- The facts/judgment boundary is preserved: PR-008 still emits no judgment, PR-009 still emits no per-pixel or per-track detail.
- Downstream (PR-010 audit trail, alerting, VLM narrative) consumes a single validated `Event` contract.
- Because events are deterministic, PR-010 can log them with the same reproducibility guarantees as every earlier layer.

## Alternatives Considered

- **Thresholds without hysteresis (immediate open/close)**: Rejected. Single-frame jitter would open and close events constantly, making every downstream consumer (audit, alerting) noisy and hard to tune. The consecutive-frame budget is the same idiom PR-006 already validated.
- **Stateful rules (rules remember history)**: Rejected. It splits temporal logic across rules and the engine, complicating unit tests and the audit story; keeping rules stateless makes "what changed this frame" fully explained by the engine's one code path.
- **Floating-point severity scores**: Rejected. ADR-0001 wants traceable, reproducible judgments; a score requires thresholds to interpret anyway, and the project has no consumer for graded events yet.
- **`<=` threshold comparison**: Rejected. A pair exactly at the threshold must deterministically be either "hazard" or "not hazard"; strict `<` pins that boundary and the integration test pins the exact frame numbers it produces.
- **VLM-based event judgment**: Rejected. It would violate ADR-0001's deterministic-state-as-source-of-truth decision and could not guarantee reproducibility or auditability.
