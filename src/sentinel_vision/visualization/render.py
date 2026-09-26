"""Visualization of pipeline state onto frames (PR-010).

Pure drawing layer: ``render_frame`` returns a NEW, writeable array and
never mutates ``frame.image``, matching the project's immutability
discipline. Callers own the returned copy and may hand it to
``cv2.VideoWriter`` (converting RGB -> BGR themselves, as the CLI pipeline
does).

Color-order contract (load-bearing):
    ``FrameData.image`` is RGB — the pipeline-wide convention established by
    ``VideoFileFrameProvider`` and used by ``SyntheticFrameStream``. OpenCV
    drawing primitives write their color argument into the image buffer in
    channel order, so the color triple passed here IS the visible RGB color:
    a green box on an RGB frame is ``(0, 255, 0)``. Never pass BGR triples
    here, and never assume this module knows about BGR. The CLI pipeline
    converts the rendered RGB frame to BGR only at the ``cv2.VideoWriter``
    boundary so the output video displays the same colors.

Visual language:
    - VISIBLE   : solid green box, labeled with entity id and state.
    - OCCLUDED  : solid amber box (held last-known box).
    - PREDICTED : blue dashed box (extrapolated position).
    - LOST/RETIRED: nothing is drawn — LOST carries no box by contract and
      RETIRED is terminal bookkeeping.
    - Zones     : thin polygon outline plus a name label.
    - OPEN events: a black banner across the top naming each open event's
      type and involved entity ids.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from sentinel_vision.data.contracts import BoundingBox
from sentinel_vision.events.event import Event, EventStatus
from sentinel_vision.ingestion.contracts import FrameData, ImageArray
from sentinel_vision.spatial.zone import Zone
from sentinel_vision.state.entity import EntityObservation, EntityState

_COLOR_VISIBLE = (0, 255, 0)
_COLOR_OCCLUDED = (255, 165, 0)
_COLOR_PREDICTED = (0, 0, 255)
_COLOR_ZONE = (255, 0, 255)
_COLOR_BANNER_BG = (0, 0, 0)
_COLOR_BANNER_TEXT = (255, 255, 255)

_BOX_THICKNESS = 2
_ZONE_THICKNESS = 1
_FONT_SCALE = 0.5
_DASH_LEN = 8
_GAP_LEN = 4
_BANNER_LINE_HEIGHT = 20


def _color_for_state(state: EntityState) -> tuple[int, int, int] | None:
    if state is EntityState.VISIBLE:
        return _COLOR_VISIBLE
    if state is EntityState.OCCLUDED:
        return _COLOR_OCCLUDED
    if state is EntityState.PREDICTED:
        return _COLOR_PREDICTED
    return None


def _box_int_tuple(box: BoundingBox) -> tuple[int, int, int, int]:
    return (
        int(round(box.x_min)),
        int(round(box.y_min)),
        int(round(box.x_max)),
        int(round(box.y_max)),
    )


def _draw_label(
    img: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int]
) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, color, 1, cv2.LINE_AA)


def _draw_box(
    img: np.ndarray,
    box: BoundingBox,
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = _box_int_tuple(box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, _BOX_THICKNESS)
    _draw_label(img, x1, max(y1 - 6, 16), label, color)


def _draw_dashed_line(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
) -> None:
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 0.0:
        return
    ux = dx / length
    uy = dy / length
    pos = 0.0
    while pos < length:
        end = min(pos + _DASH_LEN, length)
        cv2.line(
            img,
            (round(x1 + ux * pos), round(y1 + uy * pos)),
            (round(x1 + ux * end), round(y1 + uy * end)),
            color,
            1,
        )
        pos += _DASH_LEN + _GAP_LEN


def _draw_dashed_box(
    img: np.ndarray,
    box: BoundingBox,
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = _box_int_tuple(box)
    _draw_dashed_line(img, x1, y1, x2, y1, color)
    _draw_dashed_line(img, x2, y1, x2, y2, color)
    _draw_dashed_line(img, x2, y2, x1, y2, color)
    _draw_dashed_line(img, x1, y2, x1, y1, color)
    _draw_label(img, x1, max(y1 - 6, 16), label, color)


def draw_observation(img: np.ndarray, obs: EntityObservation) -> None:
    """Draw one entity observation's box onto ``img`` (mutates ``img``)."""
    color = _color_for_state(obs.state)
    if color is None or obs.bounding_box is None:
        return
    label = f"{obs.entity_id}:{obs.state.value}"
    if obs.state is EntityState.PREDICTED:
        _draw_dashed_box(img, obs.bounding_box, color, label)
    else:
        _draw_box(img, obs.bounding_box, color, label)


def draw_zone(img: np.ndarray, zone: Zone) -> None:
    """Draw one zone's polygon outline and name onto ``img`` (mutates ``img``)."""
    points = np.array(
        [(int(round(x)), int(round(y))) for x, y in zone.vertices],
        dtype=np.int32,
    )
    cv2.polylines(
        img,
        [points],
        isClosed=True,
        color=_COLOR_ZONE,
        thickness=_ZONE_THICKNESS,
    )
    first_x, first_y = zone.vertices[0]
    _draw_label(
        img,
        int(round(first_x)) + 2,
        max(int(round(first_y)) - 2, 16),
        zone.name,
        _COLOR_ZONE,
    )


def _event_label(event: Event) -> str:
    entity_text = ",".join(str(entity_id) for entity_id in event.entity_ids)
    if event.zone_name is not None:
        return (
            f"OPEN {event.event_type.value} entities={entity_text} "
            f"zone={event.zone_name}"
        )
    return f"OPEN {event.event_type.value} entities={entity_text}"


def draw_event_banner(img: np.ndarray, events: list[Event]) -> None:
    """Overlay a top banner naming this frame's OPEN events (mutates ``img``)."""
    open_events = [event for event in events if event.status is EventStatus.OPEN]
    if not open_events:
        return
    height, width = img.shape[:2]
    banner_height = min(_BANNER_LINE_HEIGHT * len(open_events) + 8, height)
    cv2.rectangle(img, (0, 0), (width, banner_height), _COLOR_BANNER_BG, -1)
    for i, event in enumerate(open_events):
        baseline = 16 + i * _BANNER_LINE_HEIGHT
        if baseline >= height:
            break
        _draw_label(img, 6, baseline, _event_label(event), _COLOR_BANNER_TEXT)


def render_frame(
    frame: FrameData,
    observations: list[EntityObservation],
    zones: list[Zone],
    events: list[Event],
) -> ImageArray:
    """Render pipeline state onto a copy of ``frame.image`` and return it.

    The input frame is never mutated: a fresh, writeable copy is returned.
    Zones are drawn first (beneath boxes), then entity boxes, then the event
    banner on top.
    """
    output = frame.image.copy()
    for zone in zones:
        draw_zone(output, zone)
    for obs in observations:
        draw_observation(output, obs)
    draw_event_banner(output, events)
    return output
