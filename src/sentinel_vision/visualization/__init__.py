"""Visualization of pipeline state onto frames (PR-010).

This package renders deterministic pipeline state — entity boxes, zone
outlines, and open events — onto frame copies for human visual review and
annotated output video. See docs/adr/0010-real-video-and-detector-integration.md.
"""

__all__ = ["render_frame"]

from sentinel_vision.visualization.render import render_frame
