"""Real-video end-to-end pipeline runner / benchmark (PR-011).

Wires the FULL existing pipeline against a real video file, using the
project's actual established contracts end to end — no parallel/invented
data shapes, no silent error swallowing:

    VideoFileFrameProvider -> YoloDetector -> GreedyIoUTracker
        -> PersistentEntityTracker -> WorkspaceModel -> EventEngine
        -> render_frame -> cv2.VideoWriter

Any per-frame failure is reported loudly (frame id + exception), counted,
and the frame is skipped with an honest empty-detections fallback for that
frame only. The run is never silently "successful" with zero real work
done — the final summary always reports how many frames actually failed.

Usage:
    python -m sentinel_vision.evaluation.run_benchmark \\
        --input D:/sentinel-vision/data/raw_videos/test_clip_01.mp4 \\
        --output D:/sentinel-vision/data/output/annotated.mp4
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from sentinel_vision.reidentification.spatial import SpatialReidentifier
from sentinel_vision.detection.yolo import YoloDetector
from sentinel_vision.events.engine import EventEngine
from sentinel_vision.events.event import EventStatus
from sentinel_vision.events.rules import ProximityHazardRule, ZoneIntrusionRule
from sentinel_vision.ingestion.video import VideoFileFrameProvider
from sentinel_vision.spatial.workspace import WorkspaceModel
from sentinel_vision.spatial.zone import Zone
from sentinel_vision.state.tracker import PersistentEntityTracker
from sentinel_vision.tracking.greedy import GreedyIoUTracker
from sentinel_vision.visualization.render import render_frame


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentinel Vision real-video pipeline runner")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to write the annotated output video")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument(
        "--class-filter",
        nargs="*",
        default=["person"],
        help="COCO class labels to keep (default: person only)",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="GreedyIoUTracker match threshold")
    parser.add_argument("--tracker-max-age", type=int, default=10, help="Frames a raw track survives unmatched")
    parser.add_argument("--occlusion-budget", type=int, default=2)
    parser.add_argument("--prediction-budget", type=int, default=5)
    parser.add_argument("--retirement-budget", type=int, default=8)
    parser.add_argument("--proximity-threshold-px", type=float, default=150.0)
    parser.add_argument("--sustain-frames", type=int, default=3)
    parser.add_argument("--clear-frames", type=int, default=5)
    parser.add_argument(
        "--zone",
        action="append",
        nargs=5,
        metavar=("NAME", "X1", "Y1", "X2", "Y2"),
        default=[],
        help="Add a rectangular zone: --zone lane_a 40 0 60 25 (repeatable)",
    )
    return parser


def build_zones(raw_zones: list[list[str]]) -> list[Zone]:
    zones: list[Zone] = []
    for name, x1, y1, x2, y2 in raw_zones:
        x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)
        zones.append(
            Zone(
                name=name,
                vertices=[(x1f, y1f), (x2f, y1f), (x2f, y2f), (x1f, y2f)],
            )
        )
    return zones


def main() -> None:
    args = build_argparser().parse_args()

    try:
        provider = VideoFileFrameProvider(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Could not open input video: {exc}", file=sys.stderr)
        sys.exit(1)

    zones = build_zones(args.zone)

    detector = YoloDetector(
        confidence_threshold=args.confidence_threshold,
        class_filter=args.class_filter if args.class_filter else None,
    )
    tracker = GreedyIoUTracker(iou_threshold=args.iou_threshold, max_age=args.tracker_max_age)
    reidentifier = SpatialReidentifier(retention_window=15, max_distance=60.0)
    entity_tracker = PersistentEntityTracker(
    occlusion_budget=args.occlusion_budget,
    prediction_budget=args.prediction_budget,
    retirement_budget=args.retirement_budget,
    reidentifier=reidentifier,
    )
    workspace = WorkspaceModel(zones=zones)

    rules: list[tuple] = [
        (ProximityHazardRule(threshold_px=args.proximity_threshold_px), args.sustain_frames, args.clear_frames)
    ]
    for zone in zones:
        rules.append((ZoneIntrusionRule(zone_name=zone.name), args.sustain_frames, args.clear_frames))
    event_engine = EventEngine(rules)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        args.output, fourcc, provider.metadata.fps, (provider.metadata.width, provider.metadata.height)
    )
    if not writer.isOpened():
        print(f"Could not open output video for writing: {args.output}", file=sys.stderr)
        provider.close()
        sys.exit(1)

    frame_count = 0
    failed_frame_count = 0
    total_events_opened = 0
    start_time = time.time()

    print(f"\n[Sentinel Vision] Running real-video pipeline on: {args.input}")
    if zones:
        print(f"Zones: {[z.name for z in zones]}")

    try:
        with provider:
            for frame in provider:
                frame_count += 1
                try:
                    detections = detector.detect(frame)
                    tracked = tracker.track(frame, detections)
                    observations = entity_tracker.update(frame.frame_id, tracked)
                    spatial = workspace.evaluate(frame.frame_id, observations)
                    events = event_engine.update(frame.frame_id, spatial)
                except Exception as exc:  # noqa: BLE001 — reported loudly, never silent
                    failed_frame_count += 1
                    print(
                        f"  [frame {frame.frame_id}] pipeline error, skipping frame: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    observations, events = [], []

                for event in events:
                    if event.status is EventStatus.OPEN:
                        total_events_opened += 1
                        print(f"  [frame {event.opened_frame_id}] OPEN  {event.event_type.value} {event.entity_ids}")
                    else:
                        print(
                            f"  [frame {event.closed_frame_id}] CLOSE {event.event_type.value} "
                            f"{event.entity_ids} (opened at {event.opened_frame_id})"
                        )

                rendered_rgb = render_frame(frame, observations, zones, events)
                rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
                writer.write(rendered_bgr)
    finally:
        writer.release()

    total_time = time.time() - start_time
    fps = frame_count / total_time if total_time > 0 else 0.0

    print("\n==========================================")
    print("      SENTINEL VISION — RUN SUMMARY")
    print("==========================================")
    print(f"Processed frames   : {frame_count}")
    print(f"Failed frames      : {failed_frame_count}")
    print(f"Total wall time    : {total_time:.2f} s")
    print(f"Measured FPS       : {fps:.2f}")
    print(f"Events opened      : {total_events_opened}")
    print(f"Output video       : {args.output}")
    print("==========================================\n")

    if failed_frame_count > 0:
        print(
            f"WARNING: {failed_frame_count}/{frame_count} frames failed pipeline "
            "processing — inspect the per-frame errors above before treating this "
            "run as a valid benchmark.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()