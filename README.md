# Sentinel Vision

Visual world-state and event reasoning engine for industrial workspace monitoring and asset intelligence.

## Status

**PR-010: Real Video & Detector Integration.** PR-001 through PR-009 are in
place (deterministic pipeline foundation, frozen data contracts,
detection-level evaluation, frame ingestion, detection abstraction, tracking
layer, persistent entity state, spatial re-identification, spatial workspace
model, deterministic event engine). This PR opens the real-perception layer:
`VideoFileFrameProvider` ingests actual video files via OpenCV,
`YoloDetector` runs a pretrained ultralytics YOLOv8 model on real frames, and
a visualization layer renders entities, zones, and events onto annotated
output video. It is the project's first heavy-dependency exception
(opencv-python, ultralytics) and the first PR whose end-to-end correctness
is validated by human visual review of rendered output rather than
hand-computed assertions — see
`docs/adr/0010-real-video-and-detector-integration.md`.

## Roadmap

| PR | Scope |
|----|-------|
| PR-001 | Engineering Foundation |
| PR-002 | Data + Evaluation Strategy |
| PR-003 | Frame Acquisition |
| PR-004 | Detection Abstraction |
| PR-005 | Tracker Benchmark |
| PR-006 | Persistent Entity State |
| PR-007 | Occlusion & Re-identification |
| PR-008 | Spatial Workspace Model |
| PR-009 | Deterministic Event Engine |
| PR-010 | Real Video & Detector Integration |
| PR-011 | Audit Trail |
| PR-012 | Evaluation Harness |
| PR-013 | Async VLM Worker |
| PR-014 | VLM Evidence Verification |
| PR-015+ | Advanced Research |

PR-001 through PR-012 build the deterministic pipeline, which is the
system's authoritative source of truth (see ADR-0001). The VLM layer
(PR-013, PR-014) is introduced only afterward, and is strictly advisory —
it never silently becomes ground truth.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy src
pytest
```

## Architecture Decisions

- [ADR-0001: Deterministic State Is the Source of Truth](docs/adr/0001-deterministic-state-is-source-of-truth.md)
- [ADR-0002: Data and Evaluation Strategy](docs/adr/0002-data-and-evaluation-strategy.md)
- [ADR-0003: Video Ingestion and Streaming](docs/adr/0003-video-ingestion-and-streaming.md)
- [ADR-0004: Detection Abstraction](docs/adr/0004-detection-abstraction.md)
- [ADR-0005: Tracker and Evaluation Harness](docs/adr/0005-tracker-and-evaluation-harness.md)
- [ADR-0006: Persistent Entity State](docs/adr/0006-persistent-entity-state.md)
- [ADR-0007: Spatial Re-identification](docs/adr/0007-spatial-reidentification.md)
- [ADR-0008: Spatial Workspace Model](docs/adr/0008-spatial-workspace-model.md)
- [ADR-0009: Deterministic Event Engine](docs/adr/0009-deterministic-event-engine.md)
- [ADR-0010: Real Video and Detector Integration](docs/adr/0010-real-video-and-detector-integration.md)

