# ADR-0003: Video Ingestion and Streaming

## Status

Accepted

## Context

PR-003 introduces the ingestion boundary of the deterministic pipeline:
turning a stream of images into the `FrameData` units that downstream
stages (detection, tracking, event engine) consume. Before writing any
code, three questions had to be fixed, because their answers shape every
later PR:

1. **Memory strategy** — should a video be loaded into RAM as one big
   array, or pulled frame by frame?
2. **Timestamp semantics** — what does a frame's timestamp mean, and what
   guarantees does the pipeline get from it?
3. **Test fixtures** — real video files are binary, large, codec-dependent,
   and platform-fragile. How do we test ingestion and everything built on
   top of it without them?

## Decision

### 1. Frames are consumed lazily, one at a time

`BaseFrameProvider` is the contract every source of frames implements. It
exposes stream `metadata` and a `read_next()` that returns exactly one
`FrameData` or `None` when the stream is exhausted. It also provides the
Python iterator protocol (`__iter__`/`__next__`, terminating in
`StopIteration`) and the context manager protocol (`__enter__`/`__exit__`)
so that resource cleanup is deterministic and automatic.

This is an **iterator pattern, not a full-video-in-RAM strategy**. At most
one frame is materialized at a time. Nothing in the ingestion layer ever
loads an entire clip into memory, because detection and tracking stages are
streaming consumers: they process frames as they arrive and hold only the
state they need (e.g. track history). Whole-clip operations are an explicit
consumer choice, made by whoever actually needs random access, not a
property of the provider.

### 2. Timestamps are zero-based relative milliseconds

`FrameData.timestamp_ms` is the frame's time **relative to stream start,
in milliseconds**: `frame_id * (1000.0 / fps)`. It is not a system epoch
clock reading and it is not wall-clock time.

This is a strict guarantee: timestamps are zero-based, monotonic,
deterministic, and reproducible — the same provider parameters produce the
same timestamps every run, on every machine. Wall-clock time is a property
of the environment, not the stream, and would make the pipeline
non-reproducible and unit tests flaky. When a downstream stage genuinely
needs to correlate frames with real time, it does so explicitly at its own
boundary, never by trusting the stream's timestamps to be epoch time.

### 3. Synthetic streams are the CI/CD test backbone

`SyntheticFrameStream` generates deterministic `FrameData` objects in pure
NumPy, with no video file and no capture device. It is fully specified by
its constructor arguments (width, height, fps, frame count, and whether a
moving object is drawn), and is byte-identical for identical arguments.

This is deliberate: CI must never depend on binary media fixtures (which
are unreviewable, bloat the repository, and fail differently across
platforms/codecs), nor on hardware cameras. The synthetic stream is the
canonical frame source for unit and integration tests across the whole
pipeline — every later PR that consumes frames can rely on it being
deterministic. When real capture lands, it implements the same
`BaseFrameProvider` contract, so no consumer code changes.

## Consequences

- Any consumer of frames is written against `BaseFrameProvider`, so
  swapping a synthetic stream for a real capture device (or a video file)
  later requires zero consumer changes.
- Consumers must be streaming: they cannot assume random access or
  re-iteration over a provider. A provider is single-pass; holding the
  whole clip is an explicit, justified consumer decision.
- A frame's `timestamp_ms` is only meaningful relative to its own stream;
  correlating across streams or with wall-clock time must be added
  deliberately downstream.
- The synthetic stream makes the pipeline testable before any real capture
  exists — tests can be written now, in CI, and kept hermetic forever.
- NumPy becomes a runtime dependency of `sentinel-vision` (first non-stdlib
  dependency), because `FrameData.image` is a typed NumPy array; the cost is
  justified by the array being the standard interchange format for image
  data and later vision stages.

## Alternatives Considered

**Load the entire video into RAM at open.** Rejected: unbounded memory for
long clips, a long startup stall, and it forces every consumer to either
hold the whole clip or awkwardly index into it. Streaming consumers
(detection, tracking) only ever need the current frame plus bounded state.

**Use system epoch/wall-clock timestamps.** Rejected: non-deterministic,
machine-dependent, and makes tests flaky. The pipeline's core value is
deterministic, auditable state (ADR-0001); timestamps must carry that
property too.

**Ship real sample video files as test fixtures.** Rejected: binary blobs
in the repository, unreviewable in diffs, codec and platform differences
make CI results inconsistent. The synthetic stream gives us determinism and
hermeticity for free.
