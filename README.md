# Sentinel Vision

Visual world-state and event reasoning engine for industrial workspace monitoring and asset intelligence.

## Status

**PR-004: Detection Abstraction.** PR-001 through PR-003 are in place
(deterministic pipeline foundation, frozen data contracts, detection-level
evaluation, frame ingestion). This PR adds the detection boundary:
`BaseDetector` — a single-method interface that turns one `FrameData` into
a list of `Detection` objects — plus `SyntheticBoxDetector`, a NumPy-only
threshold detector that finds the synthetic stream's moving object by its
pixels. The integration test closes PR-002's loop: independent ground
truth + pixel-derived predictions + precision/recall all agree. No real ML
model, tracking, or GPU inference exists yet — see
`docs/adr/0004-detection-abstraction.md`.

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
