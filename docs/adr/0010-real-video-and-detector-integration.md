# ADR-0010: Real Video and Detector Integration (First Heavy-Dependency Exception)

## Status

Accepted

## Context

PR-001 through PR-009 built a fully deterministic core on NumPy plus the standard
library, validated end to end by hand-computed assertions: exact frame ids, exact
bounding boxes, exact event open/close frame numbers. Everything about the pipeline
was *provably* correct — but nothing in it had ever looked at a real video frame or
a real model's output. The perception layer, the actual point of the system, was
still synthetic: `SyntheticFrameStream` and `SyntheticBoxDetector` exercise the
contracts, not reality.

PR-010 is where the pipeline first touches reality. That raises three questions that
had to be fixed before any code:

1. **Roadmap order.** The roadmap reserved PR-010 for the Audit Trail — an inner-loop
   PR that serializes an already-proven deterministic core. Real perception is the
   load-bearing input of everything downstream of it: the audit trail logs events
   regardless of which detector produced them, the evaluation harness scores real
   detections, and the VLM worker verifies real detections. Building those against a
   synthetic-only pipeline would be scaffolding with nothing to measure (the same
   reasoning that staged PR-002's metrics behind their subjects). The roadmap must be
   amended so real perception lands first.
2. **Dependencies.** Real video decoding (H.264/AAC containers) and real object
   detection (a pretrained convnet) cannot honestly be written against NumPy plus the
   standard library. This PR is the correct point to end the zero-heavy-dependency
   stance.
3. **Validation.** For the first time, end-to-end correctness cannot be pinned by
   hand-computed assertions: a COCO model's detections on arbitrary pixels are not
   numbers a human can recompute. End-to-end correctness becomes a human visual
   review of rendered output.

## Decision

### 1. Roadmap amendment: real perception before audit trail

The roadmap table in `README.md` is amended:

| PR (before)      | PR (after)   | Scope                        |
|------------------|--------------|------------------------------|
| PR-010 Audit Trail | **PR-010** Real Video & Detector Integration |
| PR-011 Evaluation Harness | PR-011 Audit Trail |
| PR-012 Async VLM Worker | PR-012 Evaluation Harness |
| PR-013 VLM Evidence Verification | PR-013 Async VLM Worker |
| PR-014+ Advanced Research | PR-014 VLM Evidence Verification |
|                   | PR-015+ Advanced Research     |

The deterministic core (PR-001 through PR-009) is complete and proven. The next thing
the project needs is **visual proof that the perception layer works on real input** —
that a real frame becomes real detections, persistent entities, spatial facts, and
events, rendered onto an annotated output video. Audit trail (now PR-011) serializes
events whose sources are now real and verified by eye; the evaluation harness (now
PR-012) will have real detections to score; the VLM worker (now PR-013) will have
real detections to verify. Every one of those PRs only becomes meaningful after this
one, so the amendment is a reordering, not a scope change.

### 2. The dependency exception: opencv-python and ultralytics

**Why every prior PR's zero-dependency stance was correct.** Detection, tracking,
entity-state, spatial, and event logic is pure geometry and bookkeeping over the
frozen contracts. NumPy plus the standard library genuinely sufficed — a threshold
detector over synthetic frames needs no CV library (ADR-0004, ADR-0007), a ray-cast
point-in-polygon test needs no geometry library (ADR-0008), and the tracking metrics
needed only a hand-rolled Hungarian assignment (ADR-0005). Adding heavy dependencies
for those problems would have bought download size, tuning surface, and runtime
latency for zero benefit, and would have muddied exactly the components this project
needs to stay deterministic and auditable.

**Why this PR must break it.** Decoding real video files (codec containers such as
H.264) is a solved, battle-tested problem that is not something to hand-roll — that
is what `cv2.VideoCapture` provides. And real object detection means a pretrained
convolutional network: ultralytics gives us pretrained COCO weights with no training,
and a maintained pipeline (letterbox, inference, NMS) that we do not have to re-derive.
The alternative to this exception is either writing a real detector from scratch
(months, and worse than the baseline) or staying synthetic forever (forfeiting the
project's purpose). This is the boundary ADR-0004's "Explicitly Deferred" section
was pointing at: *"a real ML model integration — any specific library or framework
(and its model weights, preprocessing, and confidence calibration) is out of scope.
`BaseDetector` is designed so a model detector implements the same one-method
contract."*

The exception is deliberate and narrow:

- `opencv-python` is used only for `VideoCapture`/`VideoWriter` and drawing
  primitives.
- `ultralytics` is used only for YOLO inference and `Results` parsing.
- Both sit behind the existing `BaseFrameProvider` and `BaseDetector` contracts, so
  the deterministic core still sees the same `FrameData`/`Detection` vocabulary and
  no consumer code changes.
- The type-stub gap (cv2 and ultralytics ship no full stubs) is handled by a scoped
  `[[tool.mypy.overrides]]` for exactly those two modules — strict mode is not
  weakened project-wide.
- CI now installs torch transitively. That is the accepted cost of this exception;
  the unit tests stub `cv2.VideoCapture` and the ultralytics model so no real video
  file and no model-weight download are needed in CI.

### 3. Validation-mode shift: human visual review for end-to-end correctness

PR-001 through PR-009 pinned end-to-end correctness with hand-computed assertions —
the event-pipeline integration test asserts the exact frame on which each event opens
and closes, derived by arithmetic the test documents. That discipline remains for the
deterministic core and for every component of this PR that can still be checked by a
machine:

- `test_video.py` asserts the `FrameData` contract mapping against a stubbed capture
  (frame_id sequencing, the `frame_id * (1000.0 / fps)` timestamp formula, BGR->RGB
  conversion, file-not-found error).
- `test_yolo.py` asserts the `Detection` conversion logic against a stubbed model
  (xyxy box format, confidence, class label, class filtering, confidence
  thresholding).
- `test_render.py` asserts the immutability contract (a new array is returned, the
  input frame's image is provably unmodified).

What cannot be machine-asserted is the *meaning* of the rendered output — whether the
green/amber/blue boxes correctly surround real people, whether the zone outlines sit
where the scene's lanes are, whether the events the banner reports actually happened
in the footage. For the first time, **end-to-end correctness is confirmed by human
visual review of the rendered output video**, produced by `scripts/run_real_pipeline.py`.

This is stated as a deliberate, honest description of the project's current stage,
not as a lowering of rigor: the component-level assertions above stay, the e2e run
also reports measured wall-clock FPS as real throughput data for the FYP's evaluation
objective, and — per this PR's definition of done — the engineer running the e2e
script reports their own visual assessment of whether the boxes, zones, and events
looked correct. When the audit trail (PR-011) and evaluation harness (PR-012) land,
they will turn part of this visual review back into machine-checkable artifacts; until
then the human eye is the acceptance oracle for the perception layer.

### 4. Explicitly deferred (not foreclosed)

- **Fine-tuned industrial-object detection.** The pretrained COCO model detects
  "person" and generic COCO categories only. COCO has no industrial-specific classes
  (forklift, PPE); this detector does not claim to find them out of the box.
  Fine-tuning on industrial data is deferred, matching the FYP proposal's own PPE
  scope note.
- **Camera calibration / real-world units.** Proximity thresholds and zone geometry
  remain in pixel space, exactly as inherited from ADR-0008 and ADR-0009. Converting
  pixels to meters requires a future calibration/homography PR.
- **ONNX-Runtime-based lighter inference.** Ultralytics' PyTorch path is used to prove
  the architecture; switching inference to ONNX Runtime (lighter, faster, no torch at
  runtime) is recorded as the future optimization once this architecture is proven.

## Consequences

- The project's runtime dependencies grow from NumPy-only to opencv-python and
  ultralytics (torch transitively); CI installs them. This is the first exception to
  the ADR-0002/ADR-0003 dependency ceiling, and is scoped to exactly two modules
  behind existing contracts.
- The deterministic core is untouched by the exception: it still consumes
  `FrameData`, `Detection`, `EntityObservation`, and `Event` exactly as before.
- The pipeline standardizes on RGB for `FrameData.image`; the video provider converts
  BGR->RGB on ingest, the render layer draws RGB triples, and the CLI script converts
  RGB->BGR only at the `VideoWriter` boundary. This single color-order rule is what
  makes rendered colors correct in the output video.
- End-to-end correctness now includes a human visual review step, and the e2e script
  emits a measured wall-clock FPS figure that is real throughput data.
- Model weights are downloaded once on first use (network access required) and cached
  locally.
- Every PR that references the old numbering (e.g. audit trail as "PR-010") in
  pre-PR-010 ADRs is a historical record of the plan as it stood then; the roadmap in
  `README.md` and this ADR are the current truth.

## Alternatives Considered

- **Keep the zero-dependency stance and defer real perception.** Rejected: it would
  forfeit the project's purpose. ADR-0003 and ADR-0004 always framed their synthetic
  sources as the CI backbone standing in for real capture; the abstraction PRs were
  in service of real sources, not a replacement for them.
- **Use ONNX Runtime now instead of ultralytics.** Rejected: this PR is about proving
  the architecture end to end, and ultralytics offers pretrained weights plus a
  maintained inference pipeline with the least friction. ONNX remains the documented
  future optimization once the architecture is proven.
- **Add opencv-python but keep the synthetic detector on real video.** Rejected: a
  real pipeline needs a real detector to be meaningfully reviewed end to end —
  `SyntheticBoxDetector` only finds white boxes on black.
- **Keep BGR throughout (do not convert on ingest).** Rejected: the synthetic stream
  and the render layer already assume RGB channel meaning; one pipeline-wide order is
  simpler than per-layer color knowledge, and the VideoWriter boundary is the natural,
  single place to convert.
