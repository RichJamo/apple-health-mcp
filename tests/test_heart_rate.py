"""Tests for per-workout heart rate: Apple's statistic is parsed and stored, and
analyze.heart_rate_summary turns it and the stored readings into one summary.

Apple writes a ``WorkoutStatistics`` child of type HeartRate with the workout's
average and maximum. Spans in the stored readings are averages and hide the peak,
so this statistic is the only accurate peak; it is kept on the workout row.

Synthetic data only, never real health data.
"""

import traceback
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import pytest

from apple_health_mcp.analyze import (
    HeartRateSummary,
    Reading,
    StoredWorkout,
    heart_rate_summary,
    load_workouts,
    readings_around,
)
from apple_health_mcp.db import Sample, connect, upsert_samples
from apple_health_mcp.ingest import HEART_RATE, HEART_RATE_RECOVERY, import_export, import_samples
from apple_health_mcp.parse import MalformedRecordError, iter_workouts

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


# --- the per-session summary (analyze.heart_rate_summary) ----------------------------
#
# Times are minutes or seconds from a fixed workout: 10:00 to 10:30 UTC. Readings are
# built directly, so each test shows exactly what the summary is given.

_START = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
_END = _START + timedelta(minutes=30)


def _workout(average: float | None = None, maximum: float | None = None) -> StoredWorkout:
    return StoredWorkout(
        id="W1",
        activity_type="HKWorkoutActivityTypeSurfingSports",
        start=_START,
        end=_END,
        duration_s=1800.0,
        active_energy_kcal=None,
        heart_rate_average_bpm=average,
        heart_rate_maximum_bpm=maximum,
        source_name="Test Watch",
        tz="Africa/Johannesburg",
    )


def _instant(at: datetime, bpm: float, kind: str = HEART_RATE) -> Reading:
    return Reading(type=kind, start=at, end=at, bpm=bpm)


def _span(start: datetime, end: datetime, bpm: float) -> Reading:
    return Reading(type=HEART_RATE, start=start, end=end, bpm=bpm)


def _minute(n: float) -> datetime:
    return _START + timedelta(minutes=n)


def _after_end(seconds: float) -> datetime:
    return _END + timedelta(seconds=seconds)


def test_average_and_peak_are_apples_when_the_statistic_exists():
    """With Apple's statistic present, average and peak are Apple's figures, even
    though the readings would give other numbers."""
    readings = [_span(_minute(0), _minute(30), 120.0), _instant(_minute(10), 190.0)]

    summary = heart_rate_summary(_workout(average=140.0, maximum=175.0), readings)

    assert (summary.average_bpm, summary.peak_bpm) == (140.0, 175.0)


def test_without_the_statistic_average_is_the_time_weighted_mean_of_spans():
    """Spans are weighted by the time they cover inside the workout: 0-10 at 100 bpm
    and 10-40 at 140 bpm, of which only 10-30 is inside, give (100*10 + 140*20) / 30.
    An instant reading is ignored when spans exist, because spans cover the time."""
    readings = [
        _span(_minute(0), _minute(10), 100.0),
        _span(_minute(10), _minute(40), 140.0),
        _instant(_minute(5), 200.0),
    ]

    summary = heart_rate_summary(_workout(), readings)

    assert summary.average_bpm == pytest.approx((100 * 10 + 140 * 20) / 30)


def test_without_the_statistic_or_spans_average_is_the_mean_of_instants_inside():
    """Only instants inside [start, end] count: the one 2 minutes after the end is
    recovery, not workout."""
    readings = [
        _instant(_minute(5), 120.0),
        _instant(_minute(25), 160.0),
        _instant(_after_end(120), 100.0),
    ]

    assert heart_rate_summary(_workout(), readings).average_bpm == 140.0


def test_without_the_statistic_there_is_no_peak_and_no_measured_recovery():
    """Spans are averages that hide the peak, so the readings never stand in for
    Apple's maximum: with no statistic, peak and measured recovery are both absent,
    even with a reading one minute after the end."""
    readings = [_span(_minute(0), _minute(30), 150.0), _instant(_after_end(60), 120.0)]

    summary = heart_rate_summary(_workout(), readings)

    assert summary.peak_bpm is None
    assert summary.measured_recovery_bpm is None


def test_measured_recovery_uses_the_instant_nearest_one_minute_after_the_end():
    """Peak 170. Of the instants 45-75 s after the end, the one at +62 s is nearest
    to +60 s, so recovery is 170 - 140 = 30. A span starting at exactly +60 s is not
    an instant reading and is ignored; +80 s is outside the window."""
    readings = [
        _instant(_after_end(50), 150.0),
        _instant(_after_end(62), 140.0),
        _span(_after_end(60), _after_end(600), 110.0),
        _instant(_after_end(80), 120.0),
    ]

    summary = heart_rate_summary(_workout(average=140.0, maximum=170.0), readings)

    assert summary.measured_recovery_bpm == 30.0


@pytest.mark.parametrize(
    ("offsets", "expected"),
    [
        ([45], 170.0 - 145),  # the window's start is inclusive
        ([75], 170.0 - 175),  # and so is its end (a reading above the peak is kept as is)
        ([44, 76], None),  # just outside on both sides: nothing to measure
        ([55, 65], 170.0 - 155),  # equally near +60 s: the earlier reading wins
    ],
    ids=["at 45 s", "at 75 s", "outside 45-75 s", "tie goes to the earlier"],
)
def test_the_one_minute_window_bounds_and_tie(offsets, expected):
    """Each instant's bpm is 100 + its offset in seconds, so the result names which
    reading was used."""
    readings = [_instant(_after_end(s), 100.0 + s) for s in offsets]

    summary = heart_rate_summary(_workout(average=140.0, maximum=170.0), readings)

    assert summary.measured_recovery_bpm == expected


def test_apples_recovery_figure_is_reported_as_an_estimate():
    """Apple's one-minute recovery record in the 5-minute tail is reported in its
    own field, apart from the measured figure. One before the end is not this
    workout's, so with none in the tail the field is absent."""
    tail = [_instant(_after_end(178), 25.0, kind=HEART_RATE_RECOVERY)]
    before_end = [_instant(_minute(29), 25.0, kind=HEART_RATE_RECOVERY)]

    assert heart_rate_summary(_workout(), tail).apple_recovery_estimate_bpm == 25.0
    assert heart_rate_summary(_workout(), before_end).apple_recovery_estimate_bpm is None


def test_with_two_apple_recovery_records_in_the_tail_the_earliest_is_used():
    """If Apple wrote two recovery records after one workout, the first one, given
    here second, is the one reported."""
    records = [
        _instant(_after_end(240), 30.0, kind=HEART_RATE_RECOVERY),
        _instant(_after_end(178), 25.0, kind=HEART_RATE_RECOVERY),
    ]

    assert heart_rate_summary(_workout(), records).apple_recovery_estimate_bpm == 25.0


def test_with_no_readings_and_no_statistic_everything_is_absent():
    """Nothing to go on gives None in every field, never 0, which would read as a
    real heart rate of zero."""
    summary = heart_rate_summary(_workout(), [])

    assert summary == HeartRateSummary(
        average_bpm=None,
        peak_bpm=None,
        measured_recovery_bpm=None,
        apple_recovery_estimate_bpm=None,
    )


def test_a_generator_of_readings_gives_the_same_summary_as_a_list():
    """The summary reads its input once, so a generator is not used up halfway: with
    a generator, Apple's recovery figure (read after the heart rate) used to be lost."""
    readings = [
        _instant(_minute(10), 150.0),
        _instant(_after_end(60), 130.0),
        _instant(_after_end(178), 25.0, kind=HEART_RATE_RECOVERY),
    ]
    workout = _workout(average=140.0, maximum=170.0)

    from_generator = heart_rate_summary(workout, (r for r in readings))

    assert from_generator == heart_rate_summary(workout, readings)
    assert from_generator.apple_recovery_estimate_bpm == 25.0


def test_spans_that_cover_no_time_inside_fall_back_to_the_instants():
    """A span starting exactly at the end covers no time inside the workout, so the
    average falls back to the instants: (120 + 160) / 2."""
    readings = [
        _span(_END, _END + timedelta(minutes=10), 90.0),
        _instant(_minute(5), 120.0),
        _instant(_minute(25), 160.0),
    ]

    assert heart_rate_summary(_workout(), readings).average_bpm == 140.0


def test_instants_at_exactly_the_start_and_end_count_in_the_average():
    """[start, end] is inclusive at both ends; one second after the end does not count."""
    readings = [
        _instant(_START, 100.0),
        _instant(_END, 140.0),
        _instant(_after_end(1), 400.0),
    ]

    assert heart_rate_summary(_workout(), readings).average_bpm == 120.0


def test_a_span_ending_before_it_starts_gets_no_weight():
    """Never seen in an export, and not checked at import: a reversed span would get a
    negative weight and drag the mean outside its inputs. It gets none instead."""
    readings = [
        _span(_minute(0), _minute(10), 100.0),
        _span(_minute(20), _minute(15), 200.0),  # ends five minutes before it starts
    ]

    assert heart_rate_summary(_workout(), readings).average_bpm == 100.0


@pytest.mark.parametrize(
    ("seconds_after_end", "reported"),
    [(0, True), (300, True), (301, False)],
    ids=["at the end", "at end + 5 min", "one second later"],
)
def test_apples_estimate_window_is_inclusive(seconds_after_end, reported):
    """Apple's recovery record counts from exactly the end to exactly 5 minutes after."""
    record = _instant(_after_end(seconds_after_end), 25.0, kind=HEART_RATE_RECOVERY)

    estimate = heart_rate_summary(_workout(), [record]).apple_recovery_estimate_bpm

    assert estimate == (25.0 if reported else None)


# A workout at 12:00-12:30 +0200 (10:00-10:30 UTC), so the stored times are converted;
# a second workout two hours later; readings placed on and around the bounds. The
# recovery record at 12:31:30 starts between two heart-rate readings, so a result
# grouped by type instead of sorted by start shows up.
_AROUND_EXPORT = """<?xml version="1.0" encoding="UTF-8"?>
<!-- SYNTHETIC: one workout with readings on and around its bounds, and another. -->
<HealthData>
 <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="30" durationUnit="min"
          sourceName="Test Watch" startDate="2026-03-01 12:00:00 +0200" endDate="2026-03-01 12:30:00 +0200"/>
 <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="30" durationUnit="min"
          sourceName="Test Watch" startDate="2026-03-01 14:00:00 +0200" endDate="2026-03-01 14:30:00 +0200"/>
 <!-- exactly at the start: in -->
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" unit="count/min" value="100"
         startDate="2026-03-01 12:00:00 +0200" endDate="2026-03-01 12:00:00 +0200"/>
 <!-- a span inside: in -->
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" unit="count/min" value="150"
         startDate="2026-03-01 12:10:00 +0200" endDate="2026-03-01 12:24:00 +0200"/>
 <!-- stored without a value: left out -->
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" unit="count/min"
         startDate="2026-03-01 12:20:00 +0200" endDate="2026-03-01 12:20:00 +0200"/>
 <!-- the tail: heart rate, Apple's recovery figure, heart rate again: all in, by start -->
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" unit="count/min" value="130"
         startDate="2026-03-01 12:31:00 +0200" endDate="2026-03-01 12:31:00 +0200"/>
 <Record type="HKQuantityTypeIdentifierHeartRateRecoveryOneMinute" sourceName="Test Watch"
         unit="count/min" value="25" startDate="2026-03-01 12:31:30 +0200" endDate="2026-03-01 12:31:30 +0200"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" unit="count/min" value="120"
         startDate="2026-03-01 12:33:00 +0200" endDate="2026-03-01 12:33:00 +0200"/>
 <!-- exactly at end + 5 min: in -->
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" unit="count/min" value="110"
         startDate="2026-03-01 12:35:00 +0200" endDate="2026-03-01 12:35:00 +0200"/>
 <!-- inside the other workout: not this one's -->
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" unit="count/min" value="160"
         startDate="2026-03-01 14:10:00 +0200" endDate="2026-03-01 14:10:00 +0200"/>
</HealthData>
"""


def _first_workout_and_db(tmp_path):
    export = tmp_path / "export.xml"
    export.write_text(_AROUND_EXPORT, encoding="utf-8")
    db = tmp_path / "health.sqlite"
    import_export(export, db)
    import_samples(export, db)
    conn = connect(db)
    return conn, load_workouts(conn)[0]


def test_readings_around_loads_the_window_and_its_tail_in_order(tmp_path):
    """Through the database, from a +0200 export: readings from exactly the start to
    exactly 5 minutes after the end come back typed and in start order, the recovery
    figure between two heart-rate readings. The reading without a value and the one in
    the other workout are left out."""
    conn, first = _first_workout_and_db(tmp_path)
    try:
        readings = readings_around(conn, first)
    finally:
        conn.close()

    at = datetime(2026, 3, 1, tzinfo=UTC)
    assert readings == [
        _instant(at.replace(hour=10), 100.0),
        Reading(HEART_RATE, at.replace(hour=10, minute=10), at.replace(hour=10, minute=24), 150.0),
        _instant(at.replace(hour=10, minute=31), 130.0),
        _instant(at.replace(hour=10, minute=31, second=30), 25.0, kind=HEART_RATE_RECOVERY),
        _instant(at.replace(hour=10, minute=33), 120.0),
        _instant(at.replace(hour=10, minute=35), 110.0),
    ]


def test_readings_around_takes_its_bounds_in_utc_whatever_the_workouts_offset(tmp_path):
    """The same workout with its times shown at +0200 selects the same readings: the
    bounds are compared with stored UTC strings, so they are converted first."""
    conn, first = _first_workout_and_db(tmp_path)
    sast = timezone(timedelta(hours=2))
    shown_locally = replace(
        first, start=first.start.astimezone(sast), end=first.end.astimezone(sast)
    )
    try:
        assert readings_around(conn, shown_locally) == readings_around(conn, first)
    finally:
        conn.close()


@pytest.mark.parametrize("raw", ["72,5", "NaN", "inf"])
def test_a_non_numeric_stored_value_raises_without_quoting_it(tmp_path, raw):
    """A value that is not a finite number stops the read with MalformedRecordError.
    Its rendered traceback names the type but not the raw value, and no other error is
    attached to it. "NaN" and "inf" parse as floats, but would read as real figures."""
    conn, first = _first_workout_and_db(tmp_path)
    try:
        upsert_samples(
            conn,
            [
                Sample(
                    type=HEART_RATE,
                    start=first.start + timedelta(minutes=5),
                    value=raw,
                    unit="count/min",
                    source_name="Test Watch",
                    workout_id=first.id,
                )
            ],
        )
        with pytest.raises(MalformedRecordError) as caught:
            readings_around(conn, first)
    finally:
        conn.close()

    rendered = "".join(traceback.format_exception(caught.value))
    assert HEART_RATE in rendered
    assert raw not in rendered
    assert caught.value.__context__ is None  # Python's own error, quoting it, is not kept
