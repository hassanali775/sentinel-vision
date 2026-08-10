# Sentinel Vision

Visual world-state and event reasoning engine for industrial workspace monitoring and asset intelligence.

## Status

**PR-002: Data + Evaluation Strategy.** The engineering foundation is in
place and this PR adds the frozen data contracts (`BoundingBox`,
`Detection`, `FrameRef`, `GroundTruthAnnotation` in
`src/sentinel_vision/data/contracts.py`) and detection-level evaluation
(IoU and thresholded precision/recall in
`src/sentinel_vision/evaluation/`). Tracking-level metrics are explicitly
deferred until the tracker (PR-005) and persistent entity state (PR-006)
exist; event-level evaluation waits for PR-011 — see
`docs/adr/0002-data-and-evaluation-strategy.md`. The `data/` layout
contract for benchmark clips and hand-labeled ground truth lives in
`data/README.md`. No actual data or capture/tracking code exists yet.

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
