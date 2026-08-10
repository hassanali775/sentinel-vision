"""Package boundary test for PR-001.

This test intentionally asserts nothing beyond what PR-001 actually ships:
the package imports cleanly and exposes its version. It must NOT import any
module that doesn't exist yet (capture, perception, tracking, etc.) — those
boundaries get their own tests when the corresponding PR introduces them.
"""

import sentinel_vision


def test_package_imports() -> None:
    assert sentinel_vision is not None


def test_version_is_defined() -> None:
    assert sentinel_vision.__version__ == "0.0.1"
