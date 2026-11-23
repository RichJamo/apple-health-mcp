"""Tests for the streaming workout parser.

These run against the synthetic fixture (tests/fixtures/sample_export.xml) — never
real health data.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from apple_health_mcp.parse import Workout, iter_workouts

FIXTURE = Path(__file__).parent / "fixtures" / "sample_export.xml"

SAST = timezone(timedelta(hours=2))  # the +0200 offset in the fixture


@pytest.fixture
def workouts() -> list[Workout]:
    return list(iter_workouts(FIXTURE))


def test_parses_every_workout(workouts):
    assert len(workouts) == 3


def test_first_workout_scalar_fields(workouts):
    w = workouts[0]
    assert w.activity_type == "HKWorkoutActivityTypeElliptical"
    assert w.duration_s == 1950.0  # 32.5 min -> seconds
    assert w.source_name == "Test Watch"


def test_active_energy_comes_from_active_not_basal(workouts):
    # The ActiveEnergyBurned sum (325.774), not BasalEnergyBurned (57.7682).
    assert workouts[0].active_energy_kcal == 325.774


def test_workout_without_statistics_has_no_energy(workouts):
    yoga = workouts[2]
    assert yoga.activity_type == "HKWorkoutActivityTypeYoga"
    assert yoga.active_energy_kcal is None


def test_timestamps_preserve_utc_offset(workouts):
    assert workouts[0].start == datetime(2019, 1, 22, 8, 5, 11, tzinfo=SAST)
    assert workouts[0].end == datetime(2019, 1, 22, 8, 37, 34, tzinfo=SAST)


def test_duplicate_metadata_collapses_by_key(workouts):
    meta = workouts[0].metadata
    assert meta == {"HKIndoorWorkout": "0", "HKTimeZone": "Africa/Johannesburg"}
