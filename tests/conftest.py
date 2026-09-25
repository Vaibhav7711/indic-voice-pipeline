"""Shared fixtures.

The fake serving engine lives in `test_serving_end_to_end.py` next to the
tests that established it. Re-exporting the fixture here makes pytest
discover it for every module, so a test that needs a live server takes
`engine_server` as an argument rather than importing it -- importing a
fixture shadows the name with the parameter, which is a lint error and, more
to the point, obscures where the fixture comes from.
"""

from __future__ import annotations

from tests.test_serving_end_to_end import engine_server  # noqa: F401

__all__ = ["engine_server"]
