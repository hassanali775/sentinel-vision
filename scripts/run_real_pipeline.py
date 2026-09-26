"""End-to-end real video pipeline runner (PR-010).

Wires the real-perception pipeline end to end:

    VideoFileFrameProvider -> YoloDetector -> GreedyIoUTracker
    -> PersistentEntityTracker (with SpatialReidentifier)
    -> WorkspaceModel -> EventEngine -> render_frame -> cv2.VideoWriter

and writes an annotated output video. Every Event is printed to the console
as it opens or closes — a plain-text preview of what PR-011's audit trail
will formalize — and the measured wall-clock throughput (frames/sec) is
reported at the end. This is real inference throughput data for the FYP's
evaluation objective, not synthetic.

Usage examples::

    python scripts/run_real_pipeline.py --input clip.mp4 --output out.mp4
    python scripts/run_real_pipeline.py --input clip.mp4 --output out.mp4 \
        --class-filter person --proximity-threshold-px 150 \
        --zone dock:0,0;320,0;320,120;0,120

The rendered frame is RGB (the pipeline-wide color order); the frame is
converted RGB -> BGR only here, at the ``cv2.VideoWriter`` boundary, so the
output video displays the same colors the render layer drew.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2

from sentinel_vision.detection.yolo import YoloDetector
from sentinel_vision.events.engine import EventEngine
from sentinel_vision.events.event import Event, EventStatus
from sentinel_vision.events.rules import BaseEventRule, ProximityHazardRule, ZoneIntrusionRule
from sentinel_vision.ingestion.video import VideoFileFrameProvider
from sentinel_vision.reidentification.spatial import SpatialReidentifier
from sentinel_vision.spatial.workspace import WorkspaceModel
from sentinel_vision.spatial.zone import Zone
from sentinel_vision.state.tracker import PersistentEntityTracker
from sentinel_vision.tracking.greedy import GreedyIoUTracker
from sentinel_vision.visualization.render import render_frame


def _parse_zones(zone_args: list[str]) -> list[Zone]:
    """Parse repeatable ``--zone name:x1,y1;x2,y2;x3,y3;...`` specs."""
    zones: list[Zone] = []
    for spec in zone_args:
        name, _, points_text = spec.partition(":")
        if not name.strip():
            raise ValueError(
                f"invalid --zone '{spec}': expected 'name:x,y;x,y;...' with a "
                "non-empty zone name"
            )
        vertices: list[tuple[float, float]] = []
        for point_text in points_text.split(";"):
            x_text, _, y_text = point_text.strip().partition(",")
            if not x_text or not y_text:
                raise ValueError(
                    f"invalid --zone '{spec}': expected vertices as 'x,y' "
                    "separated by ';'"
                )
            vertices.append((float(x_text), float(y_text)))
        zones.append(Zone(name=name.strip(), vertices=vertices))
    return zones


def _event_key(event: Event) -> tuple[str, tuple[int, ...], str | None]:
    return (event.event_type.value, event.entity_ids, event.zone_name)


def _describe(event: Event) -> str:
    ids = ",".join(str(entity_id) for entity_id in event.entity_ids)
    zone = f" zone={event.zone_name}" if event.zone_name is not None else ""
    if event.status is EventStatus.OPEN:
        return f"OPEN {event.event_type.value} entities={ids}{zone} (frame {event.opened_frame_id})"
    return (
        f"CLOSED {event.event_type.value} entities={ids}{zone} "
        f"(frames {event.opened_frame_id}..{event.closed_frame_id})"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Sentinel Vision real-video pipeline and write an "
            "annotated output video."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="input video file")
    parser.add_argument("--output", required=True, type=Path, help="output annotated video file")
    parser.add_argument(
        "--model",
        default="yolov8n",
        help="ultralytics model path (default: pretrained yolov8n, auto-downloaded once)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="detector confidence threshold in [0, 1] (default: 0.25)",
    )
    parser.add_argument(
        "--class-filter",
        nargs="+",
        default=None,
        help="keep only these COCO classes, e.g. --class-filter person",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="greedy IoU association threshold in [0, 1] (default: 0.5)",
    )
    parser.add_argument(
        "--proximity-threshold-px",
        type=float,
        default=120.0,
        help="PROXIMITY_HAZARD center-distance threshold in pixels (default: 120)",
    )
    parser.add_argument(
        "--zone",
        action="append",
        default=[],
        help=(
            "workspace zone as 'name:x1,y1;x2,y2;x3,y3;...' (repeatable). "
            "Zones enable ZONE_INTRUSION events and are drawn on the output."
        ),
    )
    parser.add_argument(
        "--sustain-frames",
        type=int,
        default=3,
        help="consecutive frames a condition must hold to open an event (default: 3)",
    )
    parser.add_argument(
        "--clear-frames",
        type=int,
        default=5,
        help="consecutive frames a condition must be absent to close an event (default: 5)",
    )
    parser.add_argument(
        "--occlusion-budget",
        type=int,
        default=1,
        help="OCCLUDED budget for the persistent entity tracker (default: 1)",
    )
    parser.add_argument(
        "--prediction-budget",
        type=int,
        default=3,
        help="PREDICTED budget for the persistent entity tracker (default: 3)",
    )
    parser.add_argument(
        "--retirement-budget",
        type=int,
        default=5,
        help="LOST/RETIRED budget for the persistent entity tracker (default: 5)",
    )
    parser.add_argument(
        "--retention-window",
        type=int,
        default=10,
        help="spatial re-id retention window in frames (default: 10)",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=60.0,
        help="spatial re-id max center distance in pixels (default: 60)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    zones = _parse_zones(args.zone)

    provider = VideoFileFrameProvider(args.input)
    try:
        metadata = provider.metadata
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            metadata.fps,
            (metadata.width, metadata.height),
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"could not open output video writer for '{args.output}'"
            )

        detector = YoloDetector(
            model_path=args.model,
            confidence_threshold=args.confidence,
            class_filter=args.class_filter,
        )
        tracker = GreedyIoUTracker(
            iou_threshold=args.iou_threshold,
            max_age=args.retirement_budget,
        )
        reidentifier = SpatialReidentifier(
            retention_window=args.retention_window,
            max_distance=args.max_distance,
        )
        entity_tracker = PersistentEntityTracker(
            occlusion_budget=args.occlusion_budget,
            prediction_budget=args.prediction_budget,
            retirement_budget=args.retirement_budget,
            reidentifier=reidentifier,
        )
        workspace = WorkspaceModel(zones=zones)
        rules: list[tuple[BaseEventRule[Any], int, int]] = [
            (
                ProximityHazardRule(threshold_px=args.proximity_threshold_px),
                args.sustain_frames,
                args.clear_frames,
            )
        ]
        for zone in zones:
            rules.append(
                (ZoneIntrusionRule(zone_name=zone.name), args.sustain_frames, args.clear_frames)
            )
        engine = EventEngine(rules)

        open_events: dict[tuple[str, tuple[int, ...], str | None], Event] = {}

        start = time.perf_counter()
        frame_count = 0
        for frame in provider:
            detections = detector.detect(frame)
            tracked = tracker.track(frame, detections)
            observations = entity_tracker.update(frame.frame_id, tracked)
            spatial = workspace.evaluate(frame.frame_id, observations)
            events = engine.update(frame.frame_id, spatial)

            for event in events:
                print(f"frame {frame.frame_id:5d}: {_describe(event)}")
                key = _event_key(event)
                if event.status is EventStatus.OPEN:
                    open_events[key] = event
                else:
                    open_events.pop(key, None)

            rendered = render_frame(frame, observations, zones, list(open_events.values()))
            writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
            frame_count += 1

        writer.release()
        elapsed = time.perf_counter() - start
    finally:
        provider.close()

    measured_fps = frame_count / elapsed if elapsed > 0.0 else 0.0
    total = (
        f" (file reports {metadata.total_frames} frames)"
        if metadata.total_frames is not None
        else ""
    )
    print(f"processed {frame_count} frames of '{args.input}'{total} in {elapsed:.2f}s")
    print(f"measured wall-clock throughput: {measured_fps:.2f} frames/sec")


if __name__ == "__main__":
    raise SystemExit(main())
