"""Tests for per-workout heart rate: Apple's own statistic is parsed and stored.

Apple writes a ``WorkoutStatistics`` child of type HeartRate with the workout's
average and maximum. Spans in the stored readings are averages and hide the peak,
so this statistic is the only accurate peak; it is kept on the workout row.

Synthetic data only, never real health data.
"""

from io import BytesIO
from pathlib import Path

from apple_health_mcp.db import connect
from apple_health_mcp.ingest import import_export
from apple_health_mcp.parse import iter_workouts

FIXTURE = Path(__file__).parent / "fixtures" / "sample_export.xml"


def _by_type(workouts):
    return {w.activity_type.removeprefix("HKWorkoutActivityType"): w for w in workouts}


# --- parsing -------------------------------------------------------------------------


def test_heart_rate_statistic_is_parsed_from_the_workout():
    """The fixture's HIIT workout carries a HeartRate statistic (average 140,
    maximum 175); the Elliptical and Yoga workouts carry none and get None."""
    workouts = _by_type(iter_workouts(FIXTURE))

    hiit = workouts["HighIntensityIntervalTraining"]
    assert (hiit.heart_rate_average_bpm, hiit.heart_rate_maximum_bpm) == (140.0, 175.0)
    for name in ("Elliptical", "Yoga"):
        assert workouts[name].heart_rate_average_bpm is None
        assert workouts[name].heart_rate_maximum_bpm is None


def test_a_resting_heart_rate_statistic_is_not_read_as_the_workouts():
    """Only the HeartRate type itself counts. RestingHeartRate also ends in
    "HeartRate", so a suffix match would wrongly take its figures."""
    export = BytesIO(
        b"<HealthData>"
        b'<Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="30" '
        b'durationUnit="min" sourceName="Test Watch" '
        b'startDate="2026-03-01 10:00:00 +0200" endDate="2026-03-01 10:30:00 +0200">'
        b'<WorkoutStatistics type="HKQuantityTypeIdentifierRestingHeartRate" '
        b'average="55" minimum="50" maximum="60" unit="count/min"/>'
        b"</Workout></HealthData>"
    )

    (run,) = iter_workouts(export)

    assert run.heart_rate_average_bpm is None
    assert run.heart_rate_maximum_bpm is None


# --- storage -------------------------------------------------------------------------


def test_heart_rate_statistic_is_stored_and_absent_ones_are_null(tmp_path):
    """Through the real import, the HIIT workout's average and maximum reach the
    workouts table, and a workout without the statistic stores NULL, not 0."""
    db = tmp_path / "health.sqlite"
    import_export(FIXTURE, db)

    conn = connect(db)
    try:
        rows = dict(
            (kind.removeprefix("HKWorkoutActivityType"), (average, maximum))
            for kind, average, maximum in conn.execute(
                "SELECT activity_type, heart_rate_average_bpm, heart_rate_maximum_bpm FROM workouts"
            )
        )
    finally:
        conn.close()

    assert rows == {
        "Elliptical": (None, None),
        "HighIntensityIntervalTraining": (140.0, 175.0),
        "Yoga": (None, None),
    }
