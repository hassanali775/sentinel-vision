# Sentinel Vision

Visual world-state and event reasoning engine for industrial workspace monitoring and asset intelligence.

## Status

**PR-005: Tracker Abstraction & Evaluation Harness.** PR-001 through PR-004
are in place (deterministic pipeline foundation, frozen data contracts,
detection-level evaluation, frame ingestion, detection abstraction). This
PR adds the tracking boundary: `BaseTracker` — a stateful single-method
interface that associates each frame's detections into tracks — plus
`GreedyIoUTracker`, a frame-to-frame greedy IoU baseline, and the tracking
evaluation harness (`calculate_mota`, `calculate_idf1`). The integration
test closes the loop stream -> detect -> track -> evaluate with perfect
MOTA/IDF1. Occlusion reasoning and re-identification are deferred — see
`docs/adr/0005-tracker-and-evaluation-harness.md`.

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
| PR-010 | Audit Trail |
| PR-011 | Evaluation Harness |
| PR-012 | Async VLM Worker |
| PR-013 | VLM Evidence Verification |
| PR-014+ | Advanced Research |

PR-001 through PR-011 build the deterministic pipeline, which is the
system's authoritative source of truth (see ADR-0001). The VLM layer
(PR-012, PR-013) is introduced only afterward, and is strictly advisory —
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
