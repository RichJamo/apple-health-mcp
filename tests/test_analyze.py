"""Tests for analyze.py: counting a session recorded by two sources once (#7).

Two groups:
  1. distinct_sessions — the rule on hand-built workouts: which records are one
     session, and which record of a session is kept.
  2. through the database — load_workouts and heart_rate_counts_by_source read what
     the import stored, and the whole path keeps the Watch's record.

The rule (Rich, 2026-11-23): two records from different sources are one session
when they share positive time of at least half the shorter recording, whatever
activity type each source gave it. Records are taken best first (the source that
wrote the most heart rate, which is the Watch, then the longer recording), and each
is kept unless it is the same session as one already kept.

Synthetic data only, never real health data.
"""

import itertools
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from apple_health_mcp.analyze import (
    StoredWorkout,
    distinct_sessions,
    heart_rate_counts_by_source,
    is_same_session,
    load_workouts,
)
from apple_health_mcp.db import connect
from apple_health_mcp.ingest import import_export, import_samples

FIXTURE = Path(__file__).parent / "fixtures" / "sample_export.xml"

WATCH = "Test Watch"
SURF_APP = "Surf App"
RUN_APP = "Run App"
HR_COUNTS = {WATCH: 500}  # only the Watch writes heart rate, as in a real export

SURFING = "HKWorkoutActivityTypeSurfingSports"
_BASE = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


def _w(
    workout_id: str, start_min: int, end_min: int, source: str = WATCH, kind: str = SURFING
) -> StoredWorkout:
    """A stored workout from ``start_min`` to ``end_min`` minutes after a fixed base."""
    return StoredWorkout(
        id=workout_id,
        activity_type=kind,
        start=_BASE + timedelta(minutes=start_min),
        end=_BASE + timedelta(minutes=end_min),
        duration_s=None,
        active_energy_kcal=None,
        heart_rate_average_bpm=None,
        heart_rate_maximum_bpm=None,
        source_name=source,
        tz="Africa/Johannesburg",
    )


def _ids(workouts: list[StoredWorkout]) -> list[str]:
    return [w.id for w in workouts]


# --- 1. distinct_sessions ---------------------------------------------------------


def test_one_session_from_two_sources_counts_once_keeping_the_watch():
    """The app's record comes first in the input and starts first, but the Watch's is
    kept, because its source wrote the heart rate. Fails if nothing is merged (two
    results) or if the first record wins."""
    app = _w("app", 0, 60, SURF_APP)
    watch = _w("watch", 1, 61, WATCH)

    assert _ids(distinct_sessions([app, watch], HR_COUNTS)) == ["watch"]


def test_different_activity_types_still_merge_and_keep_the_watch_type():
    """Strava can say Running while the Watch says Other for one session. Type is
    ignored, so it is one session, and the Watch record is kept with its own type."""
    run = _w("run", 0, 30, RUN_APP, kind="HKWorkoutActivityTypeRunning")
    other = _w("other", 0, 30, WATCH, kind="HKWorkoutActivityTypeOther")

    kept = distinct_sessions([run, other], HR_COUNTS)

    assert _ids(kept) == ["other"]
    assert kept[0].activity_type == "HKWorkoutActivityTypeOther"


@pytest.mark.parametrize(
    ("watch_start", "merged"),
    [(10, True), (11, False)],
    ids=["overlap exactly half the shorter", "a minute under half"],
)
def test_overlap_of_half_the_shorter_recording_is_the_boundary(watch_start, merged):
    """The app records 20 minutes, 0 to 20. A Watch record starting at minute 10
    overlaps it by 10 minutes, exactly half the shorter, and merges; one starting at
    minute 11 overlaps by 9 and stays separate. The check is the same either way round."""
    app = _w("app", 0, 20, SURF_APP)
    watch = _w("watch", watch_start, 60, WATCH)

    assert is_same_session(app, watch) is merged
    assert is_same_session(watch, app) is merged
    expected = ["watch"] if merged else ["app", "watch"]
    assert _ids(distinct_sessions([app, watch], HR_COUNTS)) == expected


def test_two_records_from_one_source_are_never_merged():
    """One source does not record a session twice, so two overlapping Watch records
    are two sessions, however much they overlap."""
    first = _w("first", 0, 30, WATCH)
    second = _w("second", 5, 35, WATCH)

    assert not is_same_session(first, second)
    assert _ids(distinct_sessions([first, second], HR_COUNTS)) == ["first", "second"]


@pytest.mark.parametrize(
    ("app_start", "app_end"),
    [(30, 60), (10, 10)],
    ids=["back to back", "zero-length record inside"],
)
def test_records_without_a_positive_overlap_are_not_merged(app_start, app_end):
    """A record that only touches the Watch's end, or has no length at all, shares no
    time with it. Half of a zero-length recording is zero, so without the
    positive-overlap condition the zero-length case would wrongly merge."""
    watch = _w("watch", 0, 30, WATCH)
    app = _w("app", app_start, app_end, SURF_APP)

    assert sorted(_ids(distinct_sessions([watch, app], HR_COUNTS))) == ["app", "watch"]


def test_without_heart_rate_the_longer_recording_is_kept():
    """Neither source wrote heart rate (never seen in a real export so far), so the
    tie goes to the longer recording: 38 minutes beats 30."""
    short = _w("short", 0, 30, SURF_APP)
    long = _w("long", 2, 40, RUN_APP)

    assert _ids(distinct_sessions([short, long], {})) == ["long"]


def test_three_sources_recording_one_session_count_once():
    """The Watch, a surf app and a run app all recording one session is one session."""
    records = [_w("surf", 0, 50, SURF_APP), _w("watch", 1, 52, WATCH), _w("run", 2, 49, RUN_APP)]

    assert _ids(distinct_sessions(records, HR_COUNTS)) == ["watch"]


def test_a_long_app_recording_does_not_merge_two_watch_workouts():
    """The app recorded 10 to 70 while the Watch recorded two workouts, 0 to 30 and
    40 to 70. The app record is the same session as each Watch workout, so it is
    dropped. The two Watch workouts are never the same session as each other (one
    source), so both are kept. Linking every matching pair into one group instead
    would chain all three and drop the Watch's second 30 minutes."""
    first = _w("first", 0, 30, WATCH)
    app = _w("app", 10, 70, SURF_APP)
    second = _w("second", 40, 70, WATCH)

    assert _ids(distinct_sessions([first, app, second], HR_COUNTS)) == ["first", "second"]


def test_a_record_is_not_dropped_through_a_chain():
    """Watch 0 to 30, app A 20 to 40, app B 30 to 45. A is the same session as the
    Watch and is dropped. B is the same session as A, but shares no time with the
    Watch, so it is kept: dropping it through A would lose its 15 minutes."""
    watch = _w("watch", 0, 30, WATCH)
    app_a = _w("app-a", 20, 40, SURF_APP)
    app_b = _w("app-b", 30, 45, RUN_APP)

    assert not is_same_session(watch, app_b)
    assert _ids(distinct_sessions([watch, app_a, app_b], HR_COUNTS)) == ["watch", "app-b"]


def test_an_app_that_splits_one_watch_session_has_both_halves_dropped():
    """The Watch recorded 10 to 40 while the app split it into 0 to 20 and 25 to 45.
    Each half is the same session as the Watch record, so both are dropped and the
    25 to 40 overlap is not counted twice."""
    first_half = _w("half-1", 0, 20, SURF_APP)
    watch = _w("watch", 10, 40, WATCH)
    second_half = _w("half-2", 25, 45, SURF_APP)

    assert _ids(distinct_sessions([first_half, watch, second_half], HR_COUNTS)) == ["watch"]


def test_output_is_in_start_order_not_preference_order():
    """Records are chosen best first, but returned by start. The longer Watch record
    ranks first and starts second, so the two orders differ."""
    records = [_w("short", 0, 10, WATCH), _w("long", 20, 80, WATCH)]

    assert _ids(distinct_sessions(records, HR_COUNTS)) == ["short", "long"]


def _preference(w: StoredWorkout, counts: dict[str, int]) -> tuple:
    """The documented ranking, restated here so the property test checks it."""
    return (-counts.get(w.source_name, 0), -w.recorded_seconds, w.start, w.id)


def test_postconditions_hold_on_random_overlapping_records():
    """On 300 seeded random sets of 8 records from 3 sources, crowded into two hours
    so overlaps are common: no two kept records are the same session; every dropped
    record is the same session as a kept record that ranks above it; and reversing
    the input changes nothing. The seed is fixed, so every run checks the same sets."""
    rng = random.Random(7)
    counts = {WATCH: 500, SURF_APP: 3}  # the surf app wrote a little heart rate too
    for trial in range(300):
        records = []
        for n in range(8):
            start = rng.randrange(0, 120)
            records.append(
                _w(
                    f"{trial}-{n}",
                    start,
                    start + rng.randrange(-10, 60),  # some reversed, some zero-length
                    rng.choice([WATCH, SURF_APP, RUN_APP]),
                )
            )

        kept = distinct_sessions(records, counts)
        kept_ids = set(_ids(kept))

        assert all(not is_same_session(a, b) for a, b in itertools.combinations(kept, 2))
        for dropped in (r for r in records if r.id not in kept_ids):
            assert any(
                is_same_session(k, dropped)
                and _preference(k, counts) < _preference(dropped, counts)
                for k in kept
            )
        assert distinct_sessions(records[::-1], counts) == kept


def test_result_is_sorted_and_does_not_depend_on_input_order():
    """Every ordering of the same records gives the same result, in start order: a
    pair that merges, a lone app session, and a pair tied on everything but id."""
    records = [
        _w("watch-a", 0, 30, WATCH),
        _w("app-a", 1, 31, SURF_APP),
        _w("app-b", 100, 130, SURF_APP),
        _w("tie-2", 200, 230, RUN_APP),  # tied with tie-1 on heart rate, length and start
        _w("tie-1", 200, 230, SURF_APP),
    ]

    results = {
        tuple(_ids(distinct_sessions(list(order), HR_COUNTS)))
        for order in itertools.permutations(records)
    }

    assert results == {("watch-a", "app-b", "tie-1")}  # one answer; the smaller id wins the tie


# --- 2. through the database --------------------------------------------------------

# One surf session recorded twice: by a surf app (10:00-11:00 +0200) and by the
# Watch a minute later. The Watch's record comes first in the file, so reading back
# in start order is observable. Only the Watch writes heart rate; its recovery figure
# is a different type and must not count as heart rate.
_TWICE_RECORDED_EXPORT = """<?xml version="1.0" encoding="UTF-8"?>
<!-- SYNTHETIC: one session recorded by two sources. -->
<HealthData>
 <Workout workoutActivityType="HKWorkoutActivityTypeSurfingSports" duration="60" durationUnit="min"
          sourceName="Test Watch" startDate="2026-03-01 10:01:00 +0200" endDate="2026-03-01 11:01:00 +0200"/>
 <Workout workoutActivityType="HKWorkoutActivityTypeSurfingSports" duration="60" durationUnit="min"
          sourceName="Surf App" startDate="2026-03-01 10:00:00 +0200" endDate="2026-03-01 11:00:00 +0200"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" unit="count/min" value="120"
         startDate="2026-03-01 10:20:00 +0200" endDate="2026-03-01 10:20:00 +0200"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" unit="count/min" value="130"
         startDate="2026-03-01 10:40:00 +0200" endDate="2026-03-01 10:40:00 +0200"/>
 <Record type="HKQuantityTypeIdentifierHeartRateRecoveryOneMinute" sourceName="Test Watch"
         unit="count/min" value="25" startDate="2026-03-01 11:02:00 +0200" endDate="2026-03-01 11:02:00 +0200"/>
</HealthData>
"""


def test_load_workouts_reads_back_typed_utc_fields(tmp_path):
    """The fixture's Elliptical workout comes back with aware UTC times and its
    stored duration, energy and zone, so later analysis can trust the types."""
    db = tmp_path / "health.sqlite"
    import_export(FIXTURE, db)

    conn = connect(db)
    try:
        workouts = load_workouts(conn)
    finally:
        conn.close()

    assert len(workouts) == 3
    elliptical = next(w for w in workouts if w.activity_type.endswith("Elliptical"))
    assert elliptical.start == datetime(2019, 1, 22, 6, 5, 11, tzinfo=UTC)
    assert elliptical.duration_s == 1950.0  # 32.5 min
    assert elliptical.active_energy_kcal == 325.774
    assert elliptical.tz == "Africa/Johannesburg"


def test_the_import_and_analysis_together_keep_the_watch_record(tmp_path):
    """End to end: the import stores both records and the Watch's readings; they load
    in start order although the file lists the Watch first; only the two heart-rate
    readings count, not the recovery figure; the session is counted once, as the
    Watch's record."""
    export = tmp_path / "export.xml"
    export.write_text(_TWICE_RECORDED_EXPORT, encoding="utf-8")
    db = tmp_path / "health.sqlite"
    import_export(export, db)
    import_samples(export, db)

    conn = connect(db)
    try:
        workouts = load_workouts(conn)
        counts = heart_rate_counts_by_source(conn)
    finally:
        conn.close()

    assert [w.source_name for w in workouts] == [SURF_APP, WATCH]  # both kept, start order
    assert counts == {WATCH: 2}  # the recovery figure is not heart rate
    kept = distinct_sessions(workouts, counts)
    assert [w.source_name for w in kept] == [WATCH]
