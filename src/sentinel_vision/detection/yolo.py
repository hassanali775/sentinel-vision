"""Real object detection via ultralytics YOLO (PR-010).

``YoloDetector`` fulfills ``BaseDetector`` with a pretrained ultralytics
YOLO model. It is the first non-synthetic detector and the first consumer
of the ultralytics/PyTorch dependency boundary opened by this PR (see
docs/adr/0010-real-video-and-detector-integration.md).

Model weights:
    The default ``model_path`` (``yolov8n``) is a pretrained COCO checkpoint
    that ultralytics downloads automatically on first use. This requires
    network access exactly once; the checkpoint is then cached locally.
    Supplying an explicit ``model_path`` to a locally cached weights file
    avoids the download entirely.

Class coverage (load-bearing):
    The pretrained COCO model only knows COCO's 80 classes. COCO has no
    industrial-specific classes such as forklift or PPE; out of the box this
    detector reliably finds "person" and generic COCO categories only.
    Anything industrial-specific requires a fine-tuned model, which is
    explicitly deferred (see ADR-0010 and the FYP proposal's PPE scope note).
"""

from __future__ import annotations

from typing import Any

# ultralytics ships partial type stubs but does not explicitly re-export the
# YOLO class from its package root; scoped per-import mypy ignore, strict mode
# is otherwise unaffected (ADR-0010).
from ultralytics import YOLO  # type: ignore[attr-defined]

from sentinel_vision.data.contracts import BoundingBox, Detection
from sentinel_vision.detection.base import BaseDetector
from sentinel_vision.ingestion.contracts import FrameData

DEFAULT_MODEL_PATH = "yolov8n"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25


class YoloDetector(BaseDetector):
    """Detect objects in a frame with a pretrained ultralytics YOLO model.

    The confidence threshold is applied twice — passed to the model's
    ``predict`` and applied again to the returned boxes — so the conversion
    logic below is independently correct even if a model's built-in filtering
    behaves differently. Ultralytics output boxes are already xyxy pixel
    coordinates, matching ``BoundingBox``'s inclusive-min/exclusive-max
    convention, and class ids are resolved through ``model.names`` to COCO
    labels. A frame with no detections yields ``[]`` cleanly, never an error.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        class_filter: list[str] | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold ({confidence_threshold}) must be within [0.0, 1.0]"
            )
        self._confidence_threshold = confidence_threshold
        self._class_filter: tuple[str, ...] | None = (
            tuple(class_filter) if class_filter is not None else None
        )
        # ultralytics provides no (or incomplete) type stubs; the model is
        # typed Any by the scoped mypy override for the "ultralytics" module.
        self._model: Any = YOLO(model_path)

    def detect(self, frame: FrameData) -> list[Detection]:
        """Return the detections found in ``frame``, or ``[]`` if none."""
        results = self._model.predict(
            source=frame.image,
            conf=self._confidence_threshold,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i]
                confidence = float(boxes.conf[i])
                class_label = self._model.names[int(boxes.cls[i])]
                if confidence < self._confidence_threshold:
                    continue
                if (
                    self._class_filter is not None
                    and class_label not in self._class_filter
                ):
                    continue
                detections.append(
                    Detection(
                        bounding_box=BoundingBox(
                            x_min=float(x1),
                            y_min=float(y1),
                            x_max=float(x2),
                            y_max=float(y2),
                        ),
                        confidence=confidence,
                        class_label=class_label,
                    )
                )
        return detections
