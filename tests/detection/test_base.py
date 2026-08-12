"""Tests for the detector contract (PR-004).

``BaseDetector`` is an ABC with a single abstract method — detection is a
pure function of one frame, so the interface carries no other operations.
It therefore cannot be instantiated directly.
"""

import pytest

from sentinel_vision.detection.base import BaseDetector


class TestBaseDetector:
    def test_abstract_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseDetector()  # type: ignore[abstract]
