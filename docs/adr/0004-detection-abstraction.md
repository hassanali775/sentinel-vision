# ADR-0004: Detection Abstraction

## Status

Accepted

## Context

PR-003 established the ingestion boundary: `BaseFrameProvider` turns a
stream of images into immutable `FrameData` units. PR-004 introduces the
next stage of the deterministic pipeline — turning one of those frames
into a set of detections. Before writing any code, two questions had to be
fixed, because their answers shape every later PR:

1. **Shape of the interface** — what does a detector look like to the rest
   of the pipeline, so that tracking (PR-005), entity state (PR-006), and
   the evaluation harness (PR-011) can consume it without knowing which
   detector produced the boxes?
2. **What the first detector actually does** — if the detector for the
   synthetic stream merely reads back the stream's known object position,
   the integration test scores a tautology. What makes a detector worth
   evaluating?

## Decision

### 1. `BaseDetector` is a single-method interface

`BaseDetector` is an ABC with exactly one abstract method:

```python
def detect(self, frame: FrameData) -> list[Detection]: ...
```

Detection is a **pure function of one frame**: it consumes a `FrameData`,
returns the `Detection` objects found in it, and holds no stream state and
owns no resources. Because there is no state to manage and nothing to
clean up, the iterator and context-manager machinery that `BaseFrameProvider`
needs would be dead weight here — the minimalism principle that shaped
PR-003's contract (ADR-0003) applies in its strongest form: an interface
carries only the operations its consumers actually call.

### 2. The synthetic detector is threshold-based, not a trajectory replay

`SyntheticBoxDetector` performs **genuine pixel-based detection**: it finds
every pixel with any channel above zero (`np.argwhere`) and encloses that
foreground set in one axis-aligned box. It deliberately does **not** read
back `SyntheticFrameStream`'s internal rendering formula or object
trajectory.

This is the load-bearing decision for PR-004's integration test: the test
recomputes each frame's expected box independently (from the stream's
public metadata) and scores the detector's pixel-derived boxes against it
with PR-002's `precision_recall_at_iou`. A replay-based mock would always
agree with the ground truth by construction, so a precision/recall score
of 1.0 would validate nothing. Only because the detector actually processes
pixel data can the score mean anything — and can a regression in the
stream or the detector be caught by a score below 1.0.

## Consequences

- Every later consumer (tracker, entity state, evaluation harness) is
  written against `BaseDetector`; a real ML model that lands later
  implements the same contract, so consumer code does not change.
- Detectors are stateless by contract: a detector that needs internal state
  must be explicit about it (as the tracker in PR-005 will be), not hide it
  behind the interface.
- Detections are expressed in PR-002's frozen contracts, so evaluation
  (PR-002) and dataset curation already interoperate with the detector.
- The integration test now closes the loop PR-002 set up: ground truth
  (independently computed) + predictions (pixel-derived) + metrics
  (precision/recall) all agree — the first end-to-end measurement of the
  deterministic pipeline.
- `SyntheticBoxDetector` inherits the synthetic stream's determinism: same
  stream parameters, same frames, same boxes, same scores, every run.

## Explicitly Deferred

The following are **deferred, not foreclosed** — nothing in this ADR
prevents a future PR from adding them:

- **Real ML model integration** — any specific library or framework (and
  its model weights, preprocessing, and confidence calibration) is out of
  scope. `BaseDetector` is designed so a model detector implements the same
  one-method contract.
- **Multi-instance detection** — `SyntheticBoxDetector` encloses all
  foreground pixels in a single box even if they were disjoint. Connected
  components and non-maximum suppression are deferred; the synthetic
  stream renders a single object, so a single box is sufficient to evaluate
  detection today.
- **NMS and duplicate suppression** — deferred with multi-instance
  detection; a single-box detector cannot produce duplicate detections.
- **GPU / batched inference** — deferred; nothing in the contract assumes
  CPU-only, single-frame processing, but the synthetic detector does not
  need batching or accelerators to be evaluated.

## Alternatives Considered

**Replay the stream's known object position.** Rejected: a detector that
answers "the object is where the stream said it is" without looking at the
pixels would make PR-004's integration test a tautology. Its precision/
recall score of 1.0 would be meaningless, and a broken rendering or a
drifted stream would go undetected. Threshold-based detection is
deliberately naive but real: it is a genuine signal-processing claim that
the pipeline can be measured against.

**Ship the detection abstraction with multiple abstract methods (e.g.
`detect`, `warmup`, `reset`).** Rejected: every method beyond `detect`
presumes lifecycle concerns a pure-function detector does not have. The
single-method interface mirrors the minimalism established in PR-003 and
keeps later stages (which do need lifecycle — a tracker accumulates state)
free to add their own contracts in their own PRs.
