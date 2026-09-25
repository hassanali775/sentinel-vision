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

```
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
```

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
cadence. `tests/state/test_tracker.py::test_lost_entity_is_never_offered_as_an_active_candidate`
pins this, including the eventual retirement and pool entry.

### 3. Plausibility is one shared function, not two

`plausibility_score` is extracted from `SpatialReidentifier.match` into
`sentinel_vision/reidentification/spatial.py` as a module-level function and is
used by **both** tiers. Likewise `_predicted_box_for` is extracted from the
aging loop's inline PREDICTED extrapolation into a `PersistentEntityTracker`
method that the aging loop and the active pool both call.

This is the part of the decision that matters most for maintenance. The two
places that must agree are exactly the two places where a silent divergence is
invisible:

- A copied plausibility rule can drift on a threshold boundary (`>` vs `>=`), on
  whether class equality gates the geometric test, or on what the score means.
  A tier-1 candidate accepted at a distance tier 2 would have rejected is an
  identity error that no test of either tier catches.
- A copied extrapolation formula can drift on `steps` (prediction steps counted
  from `occlusion_budget` or from zero), on `frame_delta` normalization, or on
  the inverted-axis clamp. If the active pool computed a different box than the
  aging loop publishes for the same frame, the tracker would be matching against
  a position it is simultaneously reporting as the entity's actual position —
  and the discrepancy would surface only as a subtly wrong relink rate on real
  video.

This project has already paid for this class of bug once: PR-007's velocity was
originally normalized by an assumed unit frame spacing and had to be corrected to
divide by the real frame gap (`5afbe6c`). The extraction is therefore not
premature deduplication — it is the mechanism that keeps ADR-0007's threshold
semantics identical across both tiers, and it is what lets tier 1 read its
thresholds (`max_distance` / `min_iou`) directly off the `SpatialReidentifier`
instead of taking a second, possibly divergent copy of them.

### 4. Threshold source, and the `reidentifier=None` contract

Tier 1 reads `max_distance` and `min_iou` from the `SpatialReidentifier` the
tracker was constructed with (exposed as read-only properties). There is
deliberately no independent tier-1 threshold configuration: one threshold
setting, one place it is defined, applied to both pools.

Because the thresholds belong to the reidentifier, a `PersistentEntityTracker`
constructed with `reidentifier=None` has no plausibility configuration to score
against. Rather than inventing an implicit default or partially applying the
match, **tier 1 is skipped entirely for that frame** and every orphan falls
through to minting, exactly as PR-006 behaved. This is a documented no-op, not
a crash and not a silent partial match, and it is pinned by
`test_without_reidentifier_active_matching_is_skipped_and_new_entity_minted`,
which asserts that an orphan sitting *exactly* on an `OCCLUDED` entity's held box
still mints a new entity without a reidentifier.

### 5. Determinism

Orphans are processed in input order, and the winner is chosen by a total order
over `(prediction_error, entity_id)` rather than by iteration order over the
entity table. Entity ids are minted monotonically, so the active pool's dict
insertion order and id order agree; the `entity_id` component of the key keeps
the result independent of that agreement anyway, because a retired entity
re-linked through tier 2 is re-inserted at the tail of the table. The same
guarantee ADR-0007's disambiguation rule makes for the retired pool is made here
for the active pool, and it is proven by running an ambiguous scenario twice with
the two entities' creation order swapped (`test_two_occluded_entities_ambiguous_orphan_takes_smallest_prediction_error`)
rather than assumed from the sort key.

### 6. First-match miss count, stated explicitly

A candidate's box for the current frame is
`_predicted_box_for(rec, rec.frames_since_last_match + 1)` — the box it *would*
have if it aged once more. An entity whose state is still `VISIBLE` from the
previous frame is therefore not a tier-1 candidate on the first frame it is
missed. This is intentional and follows directly from the "current state is
`OCCLUDED` or `PREDICTED`" candidate rule: eligibility begins at
`occlusion_budget + 1` consecutive missed frames, not at 1.

## Consequences

- Re-issued track ids are recoverable from the second frame of an occlusion
  instead of only after full retirement, closing the 15-frame dead zone that
  ADR-0007's addendum documented.
- A proximity hazard can no longer be raised between a person and their own
  duplicate identity for an occlusion that tier 1 recovers.
- `LOST` remains a genuine "we have lost this entity" state: a detection at a
  `LOST` entity's last known position starts a new identity by design.
- Both tiers are guaranteed to share one plausibility rule and one extrapolation
  rule, so the two identity pools cannot drift apart.
- The active pool is derived per frame from entity state rather than stored, so
  it is bounded by the number of live entities and needs no purge cadence of its
  own. It cannot go stale, because it is recomputed from the same records the
  aging loop mutates.
- Cost: each orphan is scored against each unclaimed active entity. This is
  `O(orphans x active_entities)` center-distance and IoU computations per frame
  on boxes already in memory — negligible against detection cost, and not
  something the additive-pool alternative would have avoided anyway.

## Alternatives Considered

- **Raise `retirement_budget` so re-identification simply arrives earlier.**
  Rejected: it does not work. The budgets are ordered
  `occlusion <= prediction <= retirement`, so growing the retirement budget
  necessarily grows the dead zone in front of it unless the other two grow with
  it — which is a strictly larger delay before an entity is ever considered
  lost, and therefore a larger delay before the system admits it has lost
  someone. ADR-0007's addendum already recorded that over-widening the budgets
  regresses short-gap re-entry behavior. Tier 1 adds reach without touching the
  loss semantics.
- **Include `LOST` entities in tier 1 using their last known box.** Rejected:
  the whole point of `LOST` is that the tracker no longer believes the position
  (ADR-0006 reports `bounding_box=None` for exactly this reason). Scoring
  against a box the pipeline has declared untrustworthy would let a stale
  identity win on proximity, and would make the aging loop and the active pool
  disagree about where the entity is.
- **Let every entity be a candidate, including `VISIBLE` ones that were not
  directly matched.** Rejected: an unclaimed `VISIBLE` entity is
  indistinguishable from a different person standing nearby in a crowd — this is
  a proximity-monitoring pipeline, and a false relink is a safety-relevant error
  (hazard attributed to the wrong person), not a cosmetic identity error. Only
  entities the tracker already believes are mid-gap are eligible.
- **A global assignment solver (Hungarian) over the orphan-by-candidate matrix.**
  Rejected for now: the one-to-one greedy claim is already order-independent and
  deterministic, matches ADR-0007's retired-pool semantics, and the real-video
  bursts that motivated this ADR are not caused by suboptimal assignment (see
  the addendum) but by candidate eligibility and threshold reach. A global
  solver is the right tool only if a measurement shows greedy assignment is
  actually losing matches.

## Addendum: measured result on `multi_person_test.mp4`

The real-video re-run (`run_benchmark.py`, production budgets
`occlusion=2 / prediction=5 / retirement=8`, `reidentifier` configured as in that
script: `retention_window=15, max_distance=60.0`, zone `lane_a 150 200 450 500`)
produced an **event log identical to the pre-ADR-0011 run, and the same 19
entities minted**. Tier 1 did not fire once in 869 frames. (The two full console
logs differ in exactly two lines, `Total wall time` and `Measured FPS`, which are
timing measurements rather than output; every event line, every entity count and
every other line match.)

This is a real and useful negative result, and a per-frame trace says exactly
why. Across the 869 frames the tracker mints a new id 20 times (19 distinct ids;
one id is minted, retired and re-minted). On 18 of those 20 frames the active
pool is **empty**, because every live entity is still `VISIBLE` with
`frames_since_last_match == 0` — each already claimed by another freshly
re-issued `track_id` in the same frame, so there is nothing left for tier 1 to
claim:

- The frame-508 burst is exactly this case. Its four live entities (7, 9, 10, 11)
  are all `VISIBLE`. Entity 9 happens to sit 76.4 px from the new detection's
  center, but it is `VISIBLE`, not `OCCLUDED`, so tier 1 deliberately never
  offered it as a candidate. Widening `max_distance` cannot help here; raising
  `occlusion_budget` would.
- Only two of the 20 mint frames have a genuine active candidate at all, and
  every such candidate is out of range at `max_distance=60`: at frame 703 entity
  13 is `OCCLUDED` at 84.3 px and 80.6 px, and at frame 793 entity 17 is
  `OCCLUDED` at 77.9 px. The nearest genuine active candidate anywhere in the
  video is 77.9 px, which is exactly why the table below first moves at
  `max_distance=80`.

So the mechanism is reachable on real footage — it is the *reach* that is
insufficient at these scenes. Replaying the same cached detections/tracks
through the full entity + workspace + event stack while varying only
`max_distance`:

| `max_distance` | entities minted | new mints | tier-1 relinks | events opened | tier-1 relink frames |
|---|---|---|---|---|---|
| 60 (production) | 19 | 20 | 0 | 34 | — |
| 70 | 19 | 20 | 0 | 34 | — |
| 80 | 18 | 19 | 1 | 32 | 793 |
| 90 | 15 | 17 | 4 | 28 | 212, 703, 705, 793 |
| 100 | 15 | 17 | 4 | 28 | 212, 703, 705, 793 |
| 120 | 13 | 15 | 5 | 27 | 69, 212, 703, 705, 793 |

At 90 px the ~700-737 burst does collapse (two of its ids re-link instead of
minting) and total entity count drops 19 -> 15. That threshold is **not**
adopted here: 90 px in a 464 px-wide frame is ~19% of image width, and at
`proximity_threshold_px=150` a relink that wide can attribute a hazard to the
wrong person in a crowd — trading a duplicate-identity false positive for a
wrong-identity false positive, which on this pipeline is the worse failure. The
parameter is a precision/recall decision for the evaluation chapter (ADR-0002),
not something to move on the strength of one clip.

The residual duplicate minting at frames 484-508 and 700-737 is therefore
attributed upstream: `GreedyIoUTracker(max_age=10)` is re-issuing several track
ids per frame during a crossing, which produces more simultaneous new ids than
there are mid-gap entities to claim. The follow-up this ADR commits to is a
measurement of the upstream tracker's re-issue behavior during crossings, not a
looser entity-layer threshold.
