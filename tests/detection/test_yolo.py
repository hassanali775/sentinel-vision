"""Tests for the real YOLO detector (PR-010).

Stubs an ultralytics model and Results objects so the Detection conversion
logic (box format, confidence, class label, class filtering, confidence
thresholding) is verified in isolation, without downloading model weights in
CI.
"""

from __future__ import annotations

import contextlib
from unittest import mock

import numpy as np
import pytest

from sentinel_vision.data.contracts import BoundingBox
from sentinel_vision.detection import yolo as yolo_module
from sentinel_vision.detection.yolo import DEFAULT_MODEL_PATH, YoloDetector
from sentinel_vision.ingestion.contracts import FrameData

NAMES = {0: "person", 1: "forklift"}


class _FakeBoxes:
    def __init__(self, rows: list[tuple[float, float, float, float, float, int]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def xyxy(self) -> np.ndarray:
        return np.array([row[:4] for row in self._rows], dtype=np.float32)

    @property
    def conf(self) -> np.ndarray:
        return np.array([row[4] for row in self._rows], dtype=np.float32)

    @property
    def cls(self) -> np.ndarray:
        return np.array([row[5] for row in self._rows], dtype=np.int32)


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes) -> None:
        self.boxes = boxes


class _FakeModel:
    def __init__(self, names: dict[int, str], results: list[_FakeResult]) -> None:
        self.names = names
        self._results = results
        self.predict_calls: list[tuple[object, float, bool]] = []

    def predict(self, source: object, conf: float, verbose: bool):
        self.predict_calls.append((source, conf, verbose))
        return self._results


@contextlib.contextmanager
def yolo_detector(
    names: dict[int, str] | None = None,
    results: list[_FakeResult] | None = None,
    **kwargs: object,
):
    model = _FakeModel(NAMES if names is None else names, [] if results is None else results)
    with mock.patch.object(yolo_module, "YOLO", return_value=model):
        detector = YoloDetector(**kwargs)
        yield detector, model


def make_frame() -> FrameData:
    return FrameData(frame_id=0, timestamp_ms=0.0, image=np.zeros((16, 16, 3), dtype=np.uint8))


class TestYoloDetector:
    def test_boxes_convert_to_detections(self) -> None:
        rows = [(10.0, 20.0, 30.0, 40.0, 0.9, 0)]
        with yolo_detector(results=[_FakeResult(_FakeBoxes(rows))]) as (detector, _):
            detections = detector.detect(make_frame())
        assert len(detections) == 1
        detection = detections[0]
        assert detection.bounding_box == BoundingBox(10.0, 20.0, 30.0, 40.0)
        assert detection.confidence == pytest.approx(0.9)
        assert detection.class_label == "person"

    def test_confidence_threshold_filters_low_confidence(self) -> None:
        rows = [
            (10.0, 20.0, 30.0, 40.0, 0.9, 0),
            (50.0, 60.0, 70.0, 80.0, 0.1, 0),
        ]
        with yolo_detector(
            results=[_FakeResult(_FakeBoxes(rows))], confidence_threshold=0.5
        ) as (detector, _):
            detections = detector.detect(make_frame())
        assert len(detections) == 1
        assert detections[0].confidence == pytest.approx(0.9)

    def test_class_filter_keeps_only_matching_labels(self) -> None:
        rows = [
            (10.0, 20.0, 30.0, 40.0, 0.9, 0),
            (50.0, 60.0, 70.0, 80.0, 0.8, 1),
        ]
        with yolo_detector(
            results=[_FakeResult(_FakeBoxes(rows))], class_filter=["person"]
        ) as (detector, _):
            detections = detector.detect(make_frame())
        assert len(detections) == 1
        assert detections[0].class_label == "person"
        assert detections[0].bounding_box == BoundingBox(10.0, 20.0, 30.0, 40.0)

    def test_empty_boxes_return_empty_list(self) -> None:
        with yolo_detector(results=[_FakeResult(_FakeBoxes([]))]) as (detector, _):
            assert detector.detect(make_frame()) == []

    def test_empty_results_return_empty_list(self) -> None:
        with yolo_detector(results=[]) as (detector, _):
            assert detector.detect(make_frame()) == []

    def test_multiple_results_are_combined(self) -> None:
        rows_a = [(10.0, 20.0, 30.0, 40.0, 0.9, 0)]
        rows_b = [(50.0, 60.0, 70.0, 80.0, 0.7, 1)]
        results = [_FakeResult(_FakeBoxes(rows_a)), _FakeResult(_FakeBoxes(rows_b))]
        with yolo_detector(results=results) as (detector, _):
            detections = detector.detect(make_frame())
        assert len(detections) == 2
        assert [d.class_label for d in detections] == ["person", "forklift"]

    def test_predict_receives_frame_image_and_confidence(self) -> None:
        rows = [(10.0, 20.0, 30.0, 40.0, 0.9, 0)]
        frame = make_frame()
        with yolo_detector(
            results=[_FakeResult(_FakeBoxes(rows))], confidence_threshold=0.42
        ) as (detector, model):
            detector.detect(frame)
        source, conf, verbose = model.predict_calls[0]
        assert source is frame.image
        assert conf == pytest.approx(0.42)
        assert verbose is False

    def test_detect_does_not_mutate_frame(self) -> None:
        frame = make_frame()
        original = frame.image.copy()
        rows = [(10.0, 20.0, 30.0, 40.0, 0.9, 0)]
        with yolo_detector(results=[_FakeResult(_FakeBoxes(rows))]) as (detector, _):
            detector.detect(frame)
        assert np.array_equal(frame.image, original)
        assert not frame.image.flags.writeable

    def test_default_model_path_is_pretrained_yolov8n(self) -> None:
        with mock.patch.object(yolo_module, "YOLO") as model_class:
            YoloDetector()
        assert model_class.call_args[0][0] == "yolov8n"
        assert DEFAULT_MODEL_PATH == "yolov8n"

    def test_rejects_invalid_confidence_threshold(self) -> None:
        with mock.patch.object(yolo_module, "YOLO"):
            with pytest.raises(ValueError, match="confidence_threshold"):
                YoloDetector(confidence_threshold=1.5)
