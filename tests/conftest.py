"""Fixture data, generated on demand.

The CSVs are not committed. They are derived entirely from
``tests/fixtures/make_fixtures.py`` and regenerating them is deterministic, so
committing them would add a directory of files that can only ever disagree with
the code that makes them. A tmp_path_factory session fixture writes them once
per test run instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.fixtures.make_fixtures import SPECS, write  # noqa: E402


@pytest.fixture(scope="session")
def fixture_dirs(tmp_path_factory) -> dict:
    """condition name -> directory of replicate CSVs."""
    return write(tmp_path_factory.mktemp("frap_fixtures"))


@pytest.fixture(scope="session")
def specs() -> dict:
    return {s.name: s for s in SPECS}