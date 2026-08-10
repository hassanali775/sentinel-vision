# ADR-0002: Data and Evaluation Strategy

## Status

Accepted

## Context

PR-001 established the deterministic pipeline as the authoritative source
of truth (ADR-0001). To build that pipeline credibly, we need a way to
measure whether the components we add are actually correct. This ADR fixes
what "evaluation" means now (PR-002), and — just as importantly — what it
explicitly does not mean yet.

Evaluation can happen at three increasingly integrated levels:

1. **Detection-level**: is each predicted box, frame by frame, a correct
   detection of a ground-truth object? Measured by box overlap (IoU) and
   thresholded precision/recall.
2. **Tracking-level**: across a sequence, did the system keep each object
   under a single consistent identity? Measured by MOTA, IDF1, and identity
   switches.
3. **Event-level**: did the system correctly conclude that a workspace
   event (e.g. "part present", "handoff completed") occurred? Requires the
   deterministic event engine's outputs to compare against.

Each level depends on machinery that does not exist yet, so building the
metrics before their subject is scaffolding with nothing to measure.

## Decision

PR-002 ships **detection-level evaluation only**: IoU-based
precision/recall over single frames, computed against hand-labeled ground
truth, using an IoU threshold of **0.5**.

- **0.5 is the field-standard default** (Pascal VOC, COCO's single-threshold
  variant, and most detection literature use it). It is used as the default
  parameter value, not a hardcoded constant, and is revisitable per-class
  in a later PR if class-specific evidence warrants a different threshold
  (e.g. differently shaped objects where 0.5 over- or under-penalizes).
- Ground truth is expressed in the frozen contracts in
  `src/sentinel_vision/data/contracts.py` (`GroundTruthAnnotation`,
  `BoundingBox`, `Detection`, `FrameRef`). Frozen, because these represent
  facts about a specific frame and must not be silently mutated.
- **Empty-frame edge cases are a documented convention, never a division by
  zero.** When predictions and/or ground truth are empty (a common state in
  industrial monitoring, where idle scenes are valid data), precision and
  recall are both defined as 0.0. In particular, precision with zero
  predictions is 0.0 — not 1.0 — to penalize inactivity: an idle detector
  that emits no boxes is not rewarded for "avoiding false positives"; in
  this domain missing a frame's objects is a real cost. The convention is
   enforced and explained in
   `src/sentinel_vision/evaluation/detection_metrics.py`.

Detection matching requires exact, case-sensitive class-label equality between
predicted detections and ground-truth annotations. A class mismatch is scored
strictly as a missed detection (false negative) plus a spurious detection
(false positive), rather than partial credit, because in an industrial
workspace monitoring domain, conflating object classes (e.g., person vs.
machinery) carries equal operational risk to failing to detect the object
entirely.

**Tracking-level metrics (MOTA, IDF1, ID switches) are explicitly deferred
to PR-005 and PR-006**, because they require track continuity across
frames, which doesn't exist until the tracker (PR-005) and persistent
entity state (PR-006) do.

**Event-level evaluation is deferred to PR-011 (Evaluation Harness)**,
because it evaluates the deterministic event engine (PR-009), which must
exist before its outputs can be scored.

## Consequences

- PR-002's metrics only answer "did this frame's detections match this
  frame's labels?" — they do not and must not attempt sequence-level or
  event-level claims.
- No metric in PR-002 may depend on identity continuity, on multiple
  frames, or on event-engine output; a metric that does is out of scope
  here by definition.
- Dataset curation (ground-truth annotations per frame) is contractually
  pinned to the frozen contracts; `data/README.md` documents the expected
  layout that PR-003 and later curation follow.
- Later PRs that introduce tracking or event evaluation must record any
  new metric semantics in their own ADR entries rather than amending this
  one, keeping the "what evaluation means at which stage" decision legible
  over time.

## Alternatives Considered

**Ship tracking and event metrics in the same PR-002.** Rejected: their
inputs (track continuity, persistent entity state, event-engine output) do
not exist yet, so the metrics could not be tested against real output and
would be dead code with invented semantics.

**Define precision/recall only for non-empty evaluations.** Rejected: empty
frames are common in industrial monitoring (idle scenes are valid data),
so the metrics must define and document a convention for zero predictions
and/or zero ground truth instead of dividing by zero or erroring.
