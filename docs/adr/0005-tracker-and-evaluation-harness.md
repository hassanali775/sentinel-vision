# ADR-0005: Tracker Abstraction and Evaluation Harness

## Status

Accepted

## Context

PR-004 turned a single frame into detections. A frame's detections are not
yet knowledge about the world: industrial monitoring needs to know that
"the object" seen in frame 12 is the same object seen in frame 13 — that
is, identity over time. PR-005 introduces the tracking boundary and the
metrics that measure whether tracking works:

1. **Shape of the tracker interface** — what do consumers (entity state
   PR-006, event engine PR-009) need from tracking?
2. **The baseline algorithm** — what is the simplest tracker that makes
   tracking evaluation meaningful?
3. **The metrics** — how do we score a tracker (MOTA / IDF1), and which
   edge cases are documented conventions rather than errors?

## Decision

### 1. `BaseTracker` is a single-method, stateful interface

`BaseTracker` has exactly one abstract method:

```python
def track(self, frame: FrameData, detections: list[Detection]) -> list[TrackedDetection]:
```

A tracker consumes the detections of one frame, associates them with its
internal tracks, and returns each detection tagged with a stable
`track_id`. Unlike `BaseDetector` (a pure function of one frame, ADR-0004),
tracking is inherently stateful: the association at frame `t` depends on
what was seen at frame `t-1`. The contract therefore requires callers to
invoke `track` once per frame, in stream order, and states that the tracker
holds association state between calls.

### 2. Greedy IoU is the baseline, not Kalman / DeepSORT

`GreedyIoUTracker` associates frame-to-frame by maximum IoU: each detection
(sorted by confidence descending) takes the still-available track whose
last box overlaps it most, provided the overlap meets `iou_threshold`.
Matched detections inherit the track's id; unmatched detections start a new
monotonically increasing track id; `max_age` bounds how many consecutive
unmatched frames a track survives before it is dropped.

This is deliberately the simplest baseline that makes tracking evaluation
meaningful:

- **No motion model (no Kalman filter / SORT).** The synthetic object is a
  filled box on a black background following a fully-known linear
  trajectory. A Kalman motion model predicts where a box will be next; on a
  deterministic trajectory it adds nothing that the previous frame's box
  does not already provide, and it introduces tuning parameters (process
  and measurement noise) whose only effect here is to obscure the
  evaluation. When the pipeline acquires objects with real dynamics, a
  motion-model tracker can be introduced behind the same `BaseTracker`
  contract.
- **No appearance model (no DeepSORT / re-identification).** Appearance
  features require deep feature extraction and per-object appearance
  memory. Re-identification is explicitly deferred to PR-007, alongside
  occlusion handling (PR-006), because the synthetic object has no
  distinguishing appearance to model.

### 3. MOTA and IDF1 are the tracking metrics

Two complementary measures, both computed per frame by greedy IoU matching
at `iou_threshold` (default 0.5) with PR-002's class-aware matching
convention.

**MOTA (multiple object tracking accuracy)** is a per-frame
detection-level score:

```
MOTA = 1 - (FN + FP + IDSW) / GT
```

`GT` is the total number of ground-truth detections across all frames, `FN`
the misses, `FP` the false positives, and `IDSW` the identity switches: a
ground-truth track reassigned to a different predicted track id than the
one it last matched. Unmatched ground-truth tracks carry their previous
assignment forward across gaps, so an object that reappears under a new
predicted id counts as an identity switch.

**IDF1 (identification F1)** is a track-level identity score. Each frame's
matches are counted per (ground-truth track, predicted track) pair, then a
global one-to-one assignment between ground-truth tracks and predicted
tracks maximizes the total matched detections (`IDTP`). With `IDFN` the
matched-deficient ground-truth detections and `IDFP` the spurious predicted
detections:

```
IDF1 = 2 * IDTP / (2 * IDTP + IDFN + IDFP)
```

**Documented edge cases (never a division by zero):**

- both ground truth and predictions empty: `1.0` for both metrics — an
  evaluation over nothing reports perfect agreement, mirroring the
  both-empty convention PR-002 established.
- ground truth empty but predictions present: `0.0` for both — every
  prediction is spurious.
- predictions empty but ground truth present: `0.0` for both — every
  ground-truth object is missed.
- MOTA is unbounded below: with `FN + FP + IDSW > GT` the score is
  negative (e.g. a single missed-and-spurious frame), which is a valid
  signal of a badly degraded tracker, not an error.

### 4. Occlusion state management is deferred to PR-006

`GreedyIoUTracker.max_age` handles short gaps by keeping a track alive, but
the tracker performs **no occlusion reasoning**: it does not model objects
hiding behind one another, does not decide when a disappearance is an
occlusion versus a genuine exit, and does not attempt re-identification.
Persistent entity state (PR-006) is the layer that owns the question "is
this object still in the scene, and is this new observation the same
entity?" — this ADR deliberately keeps that out of the tracker so the
baseline stays minimal and the evaluation stays interpretable.

## Consequences

- Consumers (entity state, event engine, evaluation harness) consume
  `TrackedDetection` objects — frozen, with a validated non-negative
  `track_id` — so identity is typed and immutable like every other fact in
  the pipeline.
- A better tracker (motion model, appearance, occlusion) can be
  substituted for `GreedyIoUTracker` behind the same `BaseTracker` contract
  without touching consumers or the evaluation harness.
- MOTA and IDF1 give two different views of the same run: MOTA's per-frame
  FP/FN catches detection-level regressions, IDF1's track-level assignment
  catches identity regressions. Both must be reported, matching PR-002's
  staging of detection metrics first and tracking metrics now.
- The harness stays NumPy/standard-library-only: the global IDF1
  assignment is a pure-Python Hungarian implementation, so no scipy
  dependency is introduced for PR-005.

## Alternatives Considered

**Kalman-filter tracker (SORT-style) as the baseline.** Rejected: on the
deterministic synthetic trajectory a motion model adds no predictive value,
costs tuning parameters, and would make the first tracking scores harder to
attribute. Greedy IoU isolates the association logic being evaluated.

**DeepSORT / appearance-based re-identification as the baseline.**
Rejected: requires deep feature extraction, which conflicts with the
NumPy-only constraint of this stage (ADR-0004), and has nothing to model on
a featureless synthetic box. Deferred to PR-007.

**IDF1 without a global assignment (per-frame greedy identity counting).**
Rejected: per-frame greedy identity counting under-counts identity
switches and distorts the score when track ids swap; the global optimal
assignment is the MOTChallenge-standard formulation and is what makes IDF1
a reliable identity score.

**Skip tracking metrics until the tracker is "real".** Rejected for the
same reason PR-002 shipped detection metrics before a real detector
existed: a metric with a real (if simple) subject is verifiable, and the
evaluation harness (PR-011) will need a trustworthy score to evaluate the
tracker on.
