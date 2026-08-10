# Sentinel Vision

Visual world-state and event reasoning engine for industrial workspace monitoring and asset intelligence.

## Status

**PR-001: Engineering Foundation.** This repository currently contains
only the project skeleton — package structure, tooling configuration, CI,
and the foundational architecture decision (see `docs/adr/0001-deterministic
-state-is-source-of-truth.md`). No detection, tracking, or reasoning code
exists yet. Subpackages (`capture`, `perception`, `tracking`, etc.) are
deliberately not scaffolded in advance — each is introduced in the PR that
actually needs it, containing real code from the start rather than an
empty placeholder.

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
