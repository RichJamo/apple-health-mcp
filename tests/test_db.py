"""Tests for the SQLite storage layer (db.py).

These run against the synthetic fixture (tests/fixtures/sample_export.xml) — never
real health data. They are the spec for Rich's two functions: get these green and
the storage layer does its one job.

Until workout_id and upsert_workouts are implemented they raise NotImplementedError,
so these tests are expected to be RED first and go green when the functions are written.
"""

from pathlib import Path

import pytest

from apple_health_mcp.db import connect, init_schema, upsert_workouts
from apple_health_mcp.parse import iter_workouts

FIXTURE = Path(__file__).parent / "fixtures" / "sample_export.xml"


@pytest.fixture
def conn():
    """A fresh in-memory database with the schema created.

    In-memory keeps the test fast and self-contained (F.I.R.S.T.): nothing touches
    the disk, so there is no file to clean up and no order dependence between tests.
    """
    c = connect(":memory:")
    init_schema(c)
    yield c
    c.close()


def _count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0]


def test_import_is_idempotent(conn):
    """Importing the same export twice inserts the 3 workouts once, never twice.

    This is the real test: a version that only imported once would pass even if
    dedup were broken. The second import proves the primary key does the work.
    """
    workouts = list(iter_workouts(FIXTURE))

    first = upsert_workouts(conn, workouts)
    assert first == 3  # all three are new
    assert _count(conn) == 3

    second = upsert_workouts(conn, workouts)
    assert second == 0  # same ids -> nothing inserted
    assert _count(conn) == 3  # table unchanged


def test_fields_round_trip(conn):
    """A workout's fields survive the write unchanged.

    Reads the Elliptical row back and checks the identifying type and the active
    energy that must come from the ActiveEnergyBurned statistic (325.774), proving
    the value reaches the database intact.
    """
    upsert_workouts(conn, iter_workouts(FIXTURE))

    row = conn.execute(
        "SELECT activity_type, active_energy_kcal FROM workouts WHERE activity_type = ?",
        ("HKWorkoutActivityTypeElliptical",),
    ).fetchone()

    assert row is not None
    assert row[0] == "HKWorkoutActivityTypeElliptical"
    assert row[1] == 325.774
