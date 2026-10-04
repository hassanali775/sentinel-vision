# ADR-0012: Deterministic Incident State Machine

## Status

Accepted

## Context

PR-009 established the project's judgment layer: `EventEngine` turns per-frame spatial facts into sustained `Event` objects with hysteresis — `PROXIMITY_HAZARD` and `ZONE_INTRUSION`, each with exact open and close frame ids. An event answers *"something worth paying attention to is happening, and it stopped happening."*

An event is not a security incident. Nothing yet answers the operational questions that follow one:

- Is the person involved **authorized** — a known identity on a whitelist, or a stranger?
- Has the guard been **notified**, and have they **responded**?
- If nobody answered, is this now worth **activating an alarm** and **escalating**?
- Who **resolved** it, and at what frame?

Those are the questions a security response actually turns on. Answering them from anything other than explicit state logic would break ADR-0001 (deterministic state is the source of truth), so this PR adds the layer that owns them as plain Python state logic: no model calls, no framework, no database. It sits directly on top of PR-009's output and nothing else.

## Decision

### 1. One Incident, Two Kinds of Transition

`IncidentManager` advances each incident through a fixed lifecycle:

```
                      ┌──────────────────────────► RESPONDED ──► RESOLVED
                      │                          (external)
   OPEN ──► GUARD_NOTIFIED
  (same     (same frame as OPEN)
   frame)    │
             │ guard_response_budget frames, no response
             ▼
        ALARM_ACTIVATED
             │
             │ escalation_budget frames, unresolved
             ▼
          ESCALATED ──► RESOLVED
                       (external)
```

Transitions come from exactly two places, and the split is deliberate:

- **Automatic and frame-driven** (`update`): the two timeout budgets. `GUARD_NOTIFIED → ALARM_ACTIVATED` once `guard_response_budget` frames pass with no response; `ALARM_ACTIVATED → ESCALATED` once `escalation_budget` frames pass unresolved. A guard who answers gets no alarm: responding leaves `RESPONDED` and stops the clock.
- **External and human-driven** (`record_response`, `resolve`): a guard acting on the incident. These are out-of-band from `update()` because this PR ships no notification delivery mechanism that could call them automatically. Until one exists, only an explicit call moves an incident, and that separation is what keeps an unactionable hook from being silently wired to nothing.

`record_response` and `resolve` raise rather than no-op. A guard action that silently does nothing is worse than one that raises: the first hides a lost response, the second surfaces it.

`resolve` accepts `RESPONDED` or `ESCALATED` and nothing else. An alarm that is still sounding cannot be resolved, because resolving one would assert a human took it over during the window where nobody has yet been told to go look.

### 2. State Table and the Field-Presence Invariant

`Incident` is a frozen dataclass whose `state` fully determines which frame ids exist. The table, enforced in `__post_init__` in both directions:

| State | Required frame fields (all others must be `None`) |
| --- | --- |
| `OPEN` | `opened_frame_id` |
| `GUARD_NOTIFIED` | `opened_frame_id`, `guard_notified_frame_id` |
| `RESPONDED` | `opened_frame_id`, `guard_notified_frame_id`, `responded_frame_id` |
| `ALARM_ACTIVATED` | `opened_frame_id`, `guard_notified_frame_id`, `alarm_activated_frame_id` |
| `ESCALATED` | `opened_frame_id`, `guard_notified_frame_id`, `alarm_activated_frame_id`, `escalated_frame_id` |
| `RESOLVED` | responded path: `opened_frame_id`, `guard_notified_frame_id`, `responded_frame_id`, `resolved_frame_id` — or escalation path: `opened_frame_id`, `guard_notified_frame_id`, `alarm_activated_frame_id`, `escalated_frame_id`, `resolved_frame_id` |

Two branch rules fall out of the table and are enforced separately, because each deserves its own error message rather than being reported as a generic missing field:

- `responded_frame_id` and `alarm_activated_frame_id` are mutually exclusive. An incident is either handled by a guard or escalated, never both.
- A `RESOLVED` incident must record which of the two branches it took.

This mirrors `EntityObservation`'s LOST/`bounding_box` invariant from PR-006: field presence is a function of state, checked both ways, so an incident carrying a response frame without being in `RESPONDED` is rejected at construction rather than silently believed downstream.

Every frame id that exists must be non-decreasing along the incident's path, in path order and considering only the fields its state requires: `opened_frame_id <= guard_notified_frame_id <= (responded_frame_id | alarm_activated_frame_id) <= escalated_frame_id <= resolved_frame_id`. The bounds are `>=` and not `>` deliberately: guard notification happens on the frame the incident opens, and a guard can respond on the frame they were notified. What the invariant rules out is an incident claiming a transition happened *before* the one preceding it — an incident resolved on frame 8 by a guard who responded on frame 10 records a response that had not happened yet.

`OPEN` is therefore an internal, immediately-superseded state. It exists in the enum because it is where an incident starts, but it never appears in `update()`'s return value: the contract would reject an `OPEN` incident that already carried a notification frame. Callers observe an incident only once it is `GUARD_NOTIFIED`.

### 3. Frame Budgets, Not Wall-Clock Timeouts

Budgets count `update()` calls, not elapsed frame ids: each call advances every
active incident's counter by one, so `update()` requires a strictly increasing
`frame_id` (a repeated or rewound frame would let a timeout fire twice or not at
all). Gaps are allowed but not free: a skipped frame pauses every clock. An incident
opened on frame F with budget B alarms on the B-th later call, which is frame F+B
only when frames are consecutive. The benchmark's failure path is the concrete case
(a failed frame skips `update()`); subsampled video would require rescaling the
budgets. The external hooks are exempt: a guard action is timestamped by when it was
reported, and `Incident`'s ordering invariant polices the frame id they are handed.

### 4. One Active Incident Per `entity_id`

Per this ADR's stated simplification, `entity_id` is the incident key: at most one active incident per entity at a time. A second unauthorized event for an entity that already has an active incident is folded into it rather than opening a parallel incident.

This is the right trade for now because every transition this PR models is *per entity* — one guard to notify, one alarm to activate. It also keeps incident identity stable for the audit trail and the eventual renderer, which both want to draw one thing per entity. What it cannot express is two simultaneous *reasons* against the same person (intrusion **and** proximity); that is the compound-trigger work deferred below.

`resolve` frees the entity, so a later unauthorized event opens a fresh incident with a new `incident_id` from the same monotonic sequence.

### 5. Authorization Is Injected, Not Implemented

An unauthorized entity's event opens an incident; an event whose every entity is authorized opens nothing, and a single unauthorized entity in a two-entity `PROXIMITY_HAZARD` opens an incident for that entity alone.

The whitelist itself is **not** implemented here. `IncidentManager` takes `is_authorized: Callable[[int], bool]` — the identity resolution function — and consumes only the boolean. This keeps the identity mechanism (face recognition? badge registry? a manual roster?) a swappable decision that this PR should not make on the project's behalf, and it keeps the state machine testable without one.

### 6. Only Deltas, and No Automatic Resolution

`update()` returns only incidents whose state changed this frame — timeout transitions first in sorted `entity_id` order, then newly opened incidents in event order — mirroring `EventEngine`'s convention exactly. An incident sitting unchanged in `GUARD_NOTIFIED` is never re-emitted, so a consumer's output volume is proportional to what happened rather than to how many incidents are open. `CLOSED` events are ignored: a closing event is not a new hazard.

An incident is deliberately **not** auto-resolved when its triggering event closes. The event ending says the hazard *geometry* ended — the two people moved apart, the person left the zone. It does not say anyone was found, identified, or turned away. Tying resolution to the event would silently mark strangers as handled because they walked out of frame, which is the failure mode this layer exists to prevent. Resolution is explicit and external.

### 7. Known limitation: authorization keys on tracker-minted entity ids

`is_authorized` receives an `entity_id`, and entity ids come from the tracker.
PR-011 measured that they fragment: 19 ids were minted for the 2 people in
multi_person_test.mp4, and the benchmark opened 18 incidents. Errors run both ways:
- Fragmentation: an enrolled person re-minted under a new id looks unknown and
  triggers a false incident.
- Identity swap (ADR-0011): a stranger wrongly relinked onto an authorized entity_id
  inherits that authorization, which is a missed incident (the costliest failure).
A relink preserves the id; a mint does not. How an entity_id becomes bound to an
enrolled identity, and measuring authorization continuity, are deferred.

## Consequences

- Every incident transition carries an exact frame id, so "how long until the alarm" and "how long until escalation" are frame differences, auditable against the frames that produced them.
- An authorized-entity event is provably a no-op for incidents, so tuning the whitelist cannot create or suppress incident noise.
- Determinism is preserved end to end: the incident log is a pure function of the frame sequence and the events the engine produced.
- Downstream consumers get one validated `Incident` contract and a delta stream, the same shape PR-010's audit trail and the future renderer consume.
- No notification is actually delivered by this PR. `GUARD_NOTIFIED` currently means "the system decided the guard should be told," not "the guard was told." That gap is why `record_response` is an explicit call.

## Alternatives Considered

- **Wall-clock timeouts (seconds)**: Rejected. Non-deterministic under variable decode speed and load, and unauditable — a replay would not reproduce the transitions. Frames are the project's established unit (ADR-0006, ADR-0009); wall-clock conversion is deferred to the notification PR that has a frame rate and a latency model to convert with.
- **Immediate same-frame alarm, no response window**: Rejected. A guard who is one minute away still deserves the chance to answer; the whole point of `GUARD_NOTIFIED` as a distinct state is the window between "told" and "alarm." The budget makes that window explicit and tunable.
- **Auto-resolving an incident when its event closes**: Rejected. It would report a stranger as handled because they left the frame, and it would erase the escalation state machine's purpose. Event geometry ending is not a human action.
- **Multiple concurrent incidents per entity (compound triggers)**: Deferred, not rejected. It is the right end state — "intrusion while already in proximity" is a real and more alarming situation than either alone — but it needs an incident model that can name multiple triggering events, and that is not this PR's contract.
- **Implementing the whitelist here**: Rejected. Whether authorization means face recognition, a badge registry, or a manual roster is an unresolved product decision; inventing one would hard-code it into the state machine's core. Injecting the predicate keeps both the state machine honest and the decision open.
- **LangGraph orchestration now**: Rejected for this PR. The lifecycle is deterministic, frame-driven, and small enough that plain Python state transitions are clearer and fully testable. A graph adds nodes, edges, and checkpointing for no behavior gained — but this logic is exactly what a graph should wrap, so the seam is deliberately clean: `update()` in, deltas out.
- **Silent no-ops in `record_response`/`resolve`**: Rejected. A guard action that finds no matching incident is a bug or a race in the caller; swallowing it would hide a lost response, which in a security system is the worst possible failure mode.

## Deferred Work

- **LangGraph orchestration**: a later PR wraps this state machine as a graph node set. The logic here is what it will wrap; the transition table above is the node graph.
- **Real notification delivery**: actually paging the guard, which is what will make `record_response` callable automatically and `GUARD_NOTIFIED` mean the guard was reached.
- **Wall-clock timeout conversion**: budgets in seconds via the runtime frame rate, once notification delivery makes "how long did the guard have" a meaningful question.
- **Compound-trigger incidents**: multiple concurrent incidents per `entity_id`, and incidents naming more than one triggering event type.
- **The whitelist / identity mechanism's actual implementation**: what `is_authorized` resolves against, and how an identity becomes known.
- **Renderer visualization**: drawing incident state on the annotated video. Deliberately out of scope for this PR, which is scoped to the state machine and its wiring.

