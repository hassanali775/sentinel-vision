# ADR-0007: Spatial Re-identification and Retention Pool

## Status

Accepted

## Context

PR-006 introduced `PersistentEntityTracker` to manage persistent entity lifecycles across five states (`VISIBLE`, `OCCLUDED`, `PREDICTED`, `LOST`, `RETIRED`). However, ADR-0006 explicitly documented and tested a key architectural limitation: when an upstream tracker (e.g. `GreedyIoUTracker`) drops a track ID earlier than `PersistentEntityTracker`'s total budget due to extended occlusion or tight `max_age`, the upstream tracker reissues a new `track_id` upon object reappearance.

Without re-identification, `PersistentEntityTracker` treats the unrecognized `track_id` as an unobserved new object, minting a brand-new `entity_id` while the original entity continues toward `LOST` and `RETIRED` independently. Reconnecting separate `track_id` occurrences of the same physical object across temporal gaps was explicitly deferred to PR-007.

PR-007 introduces spatial re-identification (`ReidentificationCandidate` and `SpatialReidentifier`) and multi-object synthetic stream support (`MultiObjectSyntheticFrameStream`).

## Decision

### 1. Spatial/Motion-Only Re-identification over Appearance Modeling

Re-identification is strictly spatial and motion-based using finite-difference velocity extrapolation. DeepSORT and deep neural appearance feature extractors are explicitly rejected at this stage.

- **Rationale**: The synthetic objects are featureless white boxes rendered on a black background, possessing zero visual texture or distinctiveness. Deep appearance extractors would introduce heavy neural network dependencies (PyTorch/torchvision/scipy) for zero predictive benefit. Spatial trajectory extrapolation isolates the re-identification lifecycle logic and keeps the pipeline NumPy/standard-library-only. Appearance modeling is deferred until real (non-synthetic) footage with actual visual distinctiveness demands it.

### 2. Retention Pool Semantics and Expiry

When an entity transitions to `RETIRED` in `PersistentEntityTracker`, a `ReidentificationCandidate` is constructed holding:
- `entity_id`: the retired entity's original ID.
- `last_known_box`: the last observed bounding box.
- `velocity`: per-frame finite-difference velocity vector $(v_{x,\min}, v_{y,\min}, v_{x,\max}, v_{y,\max})$.
- `retired_frame_id`: the frame index at which the entity retired.
- `last_observed_frame_id`: the frame index of its last observed box.
- `class_label`: the entity's category.

A retention pool (`SpatialReidentifier`), bounded by a configurable `retention_window` (number of frames), holds these candidates.
- **Matching**: When an unmatched track arrives at `frame_id`, the pool predicts each candidate's expected box using linear extrapolation: $\text{box}_{\text{pred}} = \text{last\_known\_box} + ( \text{frame\_id} - \text{last\_observed\_frame\_id} ) \times \text{velocity}$.
- **Thresholding**: If the incoming detection's position matches the candidate's predicted position within a configurable distance/IoU threshold, the detection re-links to the candidate's original `entity_id` instead of minting a new one. The matched candidate is removed from the retention pool upon successful re-link.
- **Purging cadence**: `purge_expired` is invoked unconditionally at the top of every `PersistentEntityTracker.update(frame_id, ...)` call — on every frame, including frames where every detection is matched and no unmatched track reaches the re-identification path. It is also a public method callable independently. This is what makes the pool's memory strictly bounded by `retention_window` on continuous streams rather than only when a match attempt happens.
- **Velocity normalization**: The `velocity` stored on retirement is computed from the two most recent observed boxes divided by the exact frame gap: $\text{frame\_delta} = \text{last\_observed\_frame\_id} - \text{second\_to\_last\_observed\_frame\_id}$ and $\text{velocity} = (\text{last\_box} - \text{second\_to\_last\_box}) / \text{frame\_delta}$. If the entity was only ever observed once (no second-to-last observation), velocity falls back to zero (static box).
- **Purging**: Candidates that remain unclaimed past `retired_frame_id + retention_window` are permanently purged from the retention pool and become unrecoverable for future re-identification.

### 3. Explicit Disambiguation Tie-Break Rule

When a new detection is spatially plausible against **more than one** retained candidate:
- **Primary Rule**: The match is awarded to the candidate with the **smallest prediction error** (closest Euclidean distance between the detection bounding box center and the candidate's predicted bounding box center).
- **Tie-Break Precedent**: In the event of an exact tie in prediction error, the match resolves to the candidate with the lowest `entity_id` (creation order precedent), mirroring `GreedyIoUTracker`'s confidence-ordering tie-break precedent from PR-005.

### 4. Multi-Object Synthetic Stream Extension

`MultiObjectSyntheticFrameStream` (and extended `SyntheticFrameStream`) supports `num_objects >= 1`, where each object moves along an independent linear trajectory with configurable or auto-spaced starting positions and velocities. `SyntheticBoxDetector` uses connected components analysis to extract disjoint foreground boxes as separate detections, enabling multi-object testing across the full pipeline.

### 5. Amendment to ADR-0004: Connected Components without Heavy Dependencies

ADR-0004 explicitly deferred multi-instance detection ("connected components and non-maximum suppression are deferred") and documented `SyntheticBoxDetector` as enclosing all foreground pixels in a single box. PR-007 amends that decision for the new `multi_instance=True` mode: distinct foreground connected components are now extracted as separate bounding boxes. The single-box behavior remains the default (`multi_instance=False`) for backward compatibility with PR-004's contract.

The connected components extraction is deliberately implemented in pure NumPy and the Python standard library — **no `scipy`, no OpenCV, and no other external CV library**:

- **Threshold mask**: `np.asarray((image > 0).any(axis=2), dtype=bool)` produces the foreground mask with pure NumPy, mirroring the threshold-based decision in ADR-0004.
- **Connected components**: an iterative flood-fill over an **8-connected neighborhood** (all `(dr, dc)` in `{-1, 0, 1}`) using a Python `list` as the queue and a NumPy boolean `visited` array. No `scipy.ndimage.label` or equivalent.
- **Bounding boxes**: each component's min/max row and column are accumulated during the flood-fill and emitted as an exclusive-max `BoundingBox`, sorted by `(y_min, x_min)` for deterministic output order.

- **Rationale**: ADR-0002/ADR-0003 establish NumPy + standard library as the ceiling for pipeline dependencies; the synthetic objects are axis-aligned filled rectangles, so a hand-rolled flood-fill is a few dozen lines, deterministic, and trivially testable. Pulling in `scipy.ndimage` or OpenCV for this would add a heavy dependency for a problem the standard library already solves. NMS remains explicitly deferred exactly as ADR-0004 records, since components of a filled white rectangle on a black background never overlap.

## Consequences

- Re-identification resolves the exact limitation documented and tested in ADR-0006: reappearing tracks after upstream track drops re-link to their original `entity_id`.
- The pipeline remains zero-heavy-dependency, running entirely on NumPy and Python standard library.
- Disambiguation is deterministic and order-independent: spatial prediction error decides matches regardless of candidate insertion order in the pool.
- Retained candidate memory is strictly bounded by `retention_window`.

## Alternatives Considered

- **DeepSORT / Neural Re-ID Feature Embeddings**: Rejected for PR-007 because synthetic boxes have no visual texture, and adding heavy dependencies (PyTorch/OpenCV) violates the lightweight deterministic testing goal.
- **Unlimited Candidate Retention**: Rejected because retaining retired entities indefinitely causes unbounded memory growth and increases spatial ambiguity over long streams.
