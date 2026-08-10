# ADR-0001: Deterministic State Is the Source of Truth

## Status

Accepted

## Context

Sentinel Vision reasons about a physical workspace over time: what assets
are present, where they are, and what events have occurred. Two broad
approaches exist for producing that world-state:

1. A **deterministic pipeline** — frame acquisition, detection, tracking,
   persistent entity state, and a rule-based event engine — where every
   claim about the world is traceable to specific frames, detections, and
   explicit logic.
2. A **VLM-driven pipeline** — asking a vision-language model to describe
   the scene or judge whether an event occurred, directly from images.

VLMs are powerful at open-ended visual understanding but are probabilistic,
non-reproducible across runs, and cannot (today) guarantee that a claim
they make is actually grounded in the pixels they were shown — the same
class of problem this project's authors have dealt with directly in
LLM-based structured extraction, where a model can produce an output that
looks justified without actually being derivable from its source.

## Decision

Deterministic state is the **authoritative** layer. Every fact the system
reports as ground truth — an asset's position, an event's occurrence, a
timestamp — must originate from the deterministic pipeline (detection,
tracking, persistent entity state, and the deterministic event engine),
which is reproducible and auditable by construction.

The VLM layer, when introduced (PR-012 and PR-013), is strictly
**advisory and asynchronous**. It may be used to:

- surface candidate events or anomalies the deterministic layer isn't
  designed to catch (open-ended scene understanding),
- provide human-readable narrative summaries of deterministic state,
- flag low-confidence deterministic regions for review.

A VLM output is never permitted to silently become authoritative state.
Any VLM claim that is surfaced to a user or downstream system must be
labeled as VLM-sourced and, where possible, verified against deterministic
state before being trusted — mirroring the verify-before-trust principle
applied elsewhere: a claim is not trustworthy merely because a model
produced it fluently: it must be checked against ground truth before it is
treated as fact.

## Consequences

- The deterministic pipeline must be built first and treated as the
  project's core value; PR-003 through PR-011 exist entirely within this
  authoritative layer before any VLM code is introduced.
- The event engine (PR-009) must be deterministic and rule-based, not a
  model call — its output must be reproducible given the same input state.
- When the VLM layer lands (PR-012+), every VLM-originated claim needs an
  explicit provenance marker so downstream consumers can distinguish
  "the system observed this" from "the VLM suggested this."
- This constrains scope early: no PR before PR-012 may introduce a VLM
  dependency, even for a convenience shortcut, without revisiting this ADR.

## Alternatives Considered

**VLM-first, deterministic-as-fallback.** Rejected: this would make the
system's core guarantees dependent on a probabilistic, non-reproducible
component from day one, undermining the auditability the project exists
to provide.

**Hybrid with no clear authority boundary.** Rejected: without an explicit
rule for which layer wins when they disagree, the system's outputs become
unpredictable and hard to reason about — exactly the failure mode this
ADR is meant to prevent.
