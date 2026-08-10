# Data Layout Contract

This directory is intentionally **empty of data**. It exists to fix, in
writing, the layout that PR-003 (frame acquisition), later detection work,
and any future dataset curation must produce and consume. This README is
the contract; no clips or labels are checked in.

## Directory layout

```
data/
  <clip_id>/
    frames/
      <frame_index>.jpg        # 1-based, zero-padded to a fixed width
      ...
    annotations.json           # one GroundTruthAnnotation-compatible record
```

- `<clip_id>` is a stable, unique slug for one recorded sequence (e.g.
  `workspace-1-2026-08-10`). It is the value used as `source_id` in every
  `FrameRef` for that clip.
- `frames/` holds the extracted frame images, named by zero-padded,
  ​1-based frame index (e.g. `000001.jpg`, `000002.jpg`, ...). The padding
  width is a property of the clip and is chosen so that
  `sorted(os.listdir(frames))` is chronological.
- `annotations.json` holds the hand-labeled ground truth for the clip.

## annotations.json

`annotations.json` is a JSON object keyed by frame, matching the shape of
the `GroundTruthAnnotation` contract in
`src/sentinel_vision/data/contracts.py`:

```json
{
  "1": {
    "frame": {
      "source_id": "<clip_id>",
      "frame_index": 1,
      "timestamp": 1723000000.0
    },
    "boxes": [
      {"x_min": 10.0, "y_min": 20.0, "x_max": 90.0, "y_max": 120.0}
    ],
    "labels": ["part"],
    "track_ids": [1]
  },
  "2": { "...": "same shape, one record per labeled frame" }
}
```

Rules every curating tool must obey:

- The JSON key is the 1-based frame index; `frame.frame_index` must match
  the key, and `frame.source_id` must equal `<clip_id>`.
- `boxes`, `labels`, and `track_ids` are parallel lists and always have
  equal length (the `GroundTruthAnnotation.__post_init__` enforces this at
  load time).
- Coordinates are in pixel space of the frames in `frames/`, with
  `x_min < x_max` and `y_min < y_max` (inclusive-min, exclusive-max, as
  documented on `BoundingBox`).
- `track_ids` entries are per-object identities across the clip; a ground-
  truth box that should carry no identity (e.g. a static fixture) uses
  `-1`. Track-based evaluation (PR-005/PR-006) only consumes boxes with
  real track ids.
- Only labeled frames appear; frames with no objects are simply absent
  from the JSON (an empty annotation list is equivalent to no annotation).

## Loaders

PR-003 and the evaluation harness (PR-011) are responsible for reading this
layout into the frozen contracts and validating them on load (rejecting a
malformed file is a hard error — a data-quality bug must surface, not be
silently accepted).
