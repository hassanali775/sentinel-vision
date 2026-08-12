"""Tests for the tracker contract (PR-005).

``BaseTracker`` is an ABC with a single abstract method — tracking is the
one operation every tracker must provide — so it cannot be instantiated
directly.
"""

import pytest

from sentinel_vision.tracking.base import BaseTracker


class TestBaseTracker:
    def test_abstract_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseTracker()  # type: ignore[abstract]
