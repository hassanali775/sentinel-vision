# ADR-0011: Active-Entity Spatial Matching

## Status

Accepted

## Context

### The gap, as observed on real video

PR-007's spatial re-identification works, but it is architecturally gated behind
full entity retirement. `SpatialReidentifier` only receives a candidate at the
moment an entity reaches `RETIRED`, so for the entire `OCCLUDED -> PREDICTED ->
LOST` stretch of an entity's life there is **no re-identification available at
all**. An upstream tracker (ADR-0005's `GreedyIoUTracker`) that expires a track
during a mid-scene occlusion and re-issues a new `track_id` on reappearance lands
in that dead zone: the entity is still alive and un-retired, so the retired pool
is empty for it, and the tracker's only options are "mint a new entity" or
"nothing". It mints.

ADR-0007's addendum already recorded this trade-off from the benchmark side
(`retirement_budget` controls re-id's effective temporal reach; production
budgets were set to 2/5/8 to widen it). What was never addressed is the
structural problem underneath: **reaching the retired pool requires surviving
the whole 2+5+8 = 15-frame gauntlet first, while a crossing occlusion is
resolved long before that**. Widening `retirement_budget` only delays the
deadline; it does not remove it, and ADR-0007's addendum already found that
over-widening it trades away false-positive suppression on a hazard-monitoring
pipeline.

### Video evidence (`data/raw_videos/multi_person_test.mp4`, 869 frames, 464x832)

The clip contains a multi-person crossing in which the upstream tracker
(`max_age=10`) expires and re-issues track ids in bursts while people pass each
other. The duplicated-entity failure is visible and reproducible at frames 486
and 705, where a single physical person is represented by four and five distinct
`entity_id`s within four frames:
[frame 508] OPEN PROXIMITY_HAZARD (7, 9)     <- 7 and 9 are the same person
[frame 508] OPEN PROXIMITY_HAZARD (7, 10)    <- and so are 10 and 11
[frame 508] OPEN PROXIMITY_HAZARD (9, 10)
[frame 510] OPEN PROXIMITY_HAZARD (7, 12)
...
[frame 705] OPEN PROXIMITY_HAZARD (7, 14)    <- five ids for one crossing
[frame 705] OPEN PROXIMITY_HAZARD (7, 15)
[frame 705] OPEN PROXIMITY_HAZARD (13, 14)
[frame 705] OPEN PROXIMITY_HAZARD (13, 15)
[frame 705] OPEN PROXIMITY_HAZARD (14, 15)


This is not a cosmetic identity problem. `ProximityHazardRule` and
`ZoneIntrusionRule` are keyed on `entity_id` (ADR-0009), so a duplicate identity
manufactures a proximity hazard between a person and themselves, splits one
intrusion across five event streams, and defeats the `sustain_frames` /
`clear_frames` hysteresis that the event engine depends on. The full run mints
**19 entities** for the people in the clip.

## Decision

### 1. Two tiers, active first

An unmatched `track_id` (an "orphan" — one no entity currently claims) is
resolved in this exact order inside `PersistentEntityTracker.update`:

1. **Direct matches** — a `track_id` an entity already owns is a VISIBLE update
   and claims that entity for the frame. Unchanged from PR-006.
2. **Tier 1: active pool** — every unclaimed entity whose current state is
   `OCCLUDED` or `PREDICTED` is a candidate, scored against the orphan. The
   winner (smallest prediction error, ties to lowest `entity_id`) re-links the
   orphan to that `entity_id`, takes over the orphan's `track_id` as its
   `source_track_id`, and returns to `VISIBLE`. Matching is **one-to-one within
   a frame**: a claimed entity leaves the pool, so two detections cannot both
   claim the same entity and an entity cannot be counted twice.
3. **Tier 2: retired pool** — if tier 1 finds nothing, `SpatialReidentifier.match`
   runs over the `RETIRED` candidates exactly as before.
4. **Mint** — if tier 2 also finds nothing, a new `entity_id` is minted.
5. **Aging** — every entity still unclaimed ages through the existing
   `OCCLUDED -> PREDICTED -> LOST -> RETIRED` loop, unchanged.

Tier 1 exists to close the retirement dead zone: a re-issued `track_id` is now
recoverable from the first frame the entity becomes `OCCLUDED`, which is
`occlusion_budget + 1` frames into the gap rather than
`occlusion_budget + prediction_budget + retirement_budget + 1` frames.

### 2. `LOST` is in neither tier, until retirement

A `LOST` entity is deliberately **not** a match target, and this is the load-
bearing boundary of the design. `LOST` means the entity's believed position has
already been abandoned: by contract it reports `bounding_box=None` (ADR-0006),
there is nothing to extrapolate from, and the aging loop's own
`_predicted_box_for` returns `None` for it. Matching a detection against a box
the tracker has already stopped believing in would let a stale identity outbid a
live one on proximity, which is the exact failure this ADR exists to stop.

The consequence is stated plainly because it looks like a bug until you know it
is not: **a detection landing on a `LOST` entity's last known position mints a
new entity**, and that is correct. `LOST` becomes eligible again only after it
transitions to `RETIRED` and enters the tier-2 pool, at which point it is
matched as a *retired* candidate with its own retention window and purge
cadence.

### 3. Plausibility is one shared function, not two

`plausibility_score` is extracted from `SpatialReidentifier.match` into
`sentinel_vision/reidentification/spatial.py` as a module-level function and is
used by **both** tiers. Likewise `_predicted_box_for` is extracted from the
aging loop's inline PREDICTED extrapolation into a `PersistentEntityTracker`
method that the aging loop and the active pool both call.

This is the part of the decision that matters most for maintenance. The two
places that must agree are exactly the two places where a silent divergence is
invisible:
- A copied plausibility rule can drift on a threshold boundary (`>` vs `>=`).
- A copied extrapolation formula can drift on `steps` or frame-delta normalization.
The extraction guarantees semantic parity across both tiers.

### 4. Bounded Prediction Box (`_bounded_prediction_box`)

To handle degenerate or inverted bounding box coordinates on real sparse-detection footage where extrapolated velocity vectors might cause an edge crossing, a bounded prediction box clamp (`_bounded_prediction_box`) is enforced, guaranteeing valid area and non-inverted coordinates on all extrapolated active candidate boxes.

### 5. Threshold source, and the `reidentifier=None` contract

Tier 1 reads `max_distance` and `min_iou` from the `SpatialReidentifier` the
tracker was constructed with (exposed as read-only properties). If constructed
with `reidentifier=None`, **tier 1 is skipped entirely for that frame** and every orphan falls through to minting as a documented no-op.

### 6. Determinism & First-Match Miss Count

Orphans are processed in input order, and the winner is chosen by a total order
over `(prediction_error, entity_id)`. Eligibility for active matching begins at
`occlusion_budget + 1` consecutive missed frames (`OCCLUDED` or `PREDICTED` state).

## Alternatives Considered

- **Raise `retirement_budget` so re-identification simply arrives earlier.** Rejected: grows the dead zone.
- **Include `LOST` entities in tier 1.** Rejected: stale positions break proximity accuracy.
- **Let every entity be a candidate, including `VISIBLE` ones.** Rejected: high identity-swap risk in crowds.
- **A global assignment solver (Hungarian).** Rejected for now: greedy claim is deterministic and sufficient.

## Addendum: measured result on `multi_person_test.mp4`

The real-video re-run (`run_benchmark.py`, production budgets
`occlusion=2 / prediction=5 / retirement=8`, `reidentifier` configured as in that
script: `retention_window=15, max_distance=60.0`, zone `lane_a 150 200 450 500`)
produced an **event log identical to the pre-ADR-0011 run, and the same 19
entities minted**. Tier 1 did not fire once in 869 frames. Across 20 mint instances, the active pool was empty (live entities were `VISIBLE`) or out of range at `max_distance=60.0` (e.g., frame 703 at 84.3 px, frame 793 at 77.9 px).

| `max_distance` | entities minted | new mints | tier-1 relinks | events opened | tier-1 relink frames |
|---|---|---|---|---|---|
| 60 (production) | 19 | 20 | 0 | 34 | — |
| 70 | 19 | 20 | 0 | 34 | — |
| 80 | 18 | 19 | 1 | 32 | 793 |
| 90 | 15 | 17 | 4 | 28 | 212, 703, 705, 793 |
| 100 | 15 | 17 | 4 | 28 | 212, 703, 705, 793 |
| 120 | 13 | 15 | 5 | 27 | 69, 212, 703, 705, 793 |

At 90 px the ~700-737 burst collapses and total entity count drops 19 -> 15. That threshold is **not** adopted here; production is maintained at **`max_distance=60.0`**. 

*(Note on artifact correlation: Visual confirmation artifacts for frames 792/793, such as `frame_792.jpg` and `frame_793.jpg`, reflect the baseline configuration where entity 17 remained `PREDICTED`, whereas the accompanying script trace output validates the `max_distance=80.0` relink transition to `VISIBLE`)*. 

Raising `max_distance` introduces **identity-swap risk**: a looser threshold makes an orphan detection near multiple close-proximity people more likely to plausibly match the wrong individual, silently corrupting historical trajectory data. Thus, production remains strictly locked at `max_distance=60.0`, framing active re-identification as a robust structural mechanism whose threshold tuning is properly bounded by identity-swap constraints rather than raw mint-count reduction.