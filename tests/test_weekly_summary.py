"""Tests for the weekly training summary (step 4).

Three functions in analyze.py, each built on the one before:
  1. local_start(workout)       — a workout's start in its stored zone.
  2. weekly_summary(sessions)   — one WeekSummary per Monday-to-Sunday local week.
  3. weekly_summary_from_db(conn) — the same, from the database, counting a session
     recorded by two sources once.

Weeks are local because a Monday 00:30 run in Johannesburg is Sunday 22:30 in UTC:
bucketing by UTC would put it in the previous week.

Synthetic data only, never real health data.
"""

import itertools
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from apple_health_mcp.analyze import (
    StoredWorkout,
    WeekSummary,
    local_start,
    pad_to_week_of,
    weekly_summary,
    weekly_summary_from_db,
)
from apple_health_mcp.db import connect
from apple_health_mcp.ingest import import_export, import_samples
from apple_health_mcp.parse import MalformedRecordError

JOBURG = "Africa/Johannesburg"  # UTC+2 all year, no daylight saving
RUN = "HKWorkoutActivityTypeRunning"
SURF = "HKWorkoutActivityTypeSurfingSports"

_ids = itertools.count()


def _session(
    start_utc: datetime,
    minutes: float = 30,
    *,
    kind: str = RUN,
    tz: str = JOBURG,
    energy: float | None = 300.0,
    duration_minutes: float | None = None,
) -> StoredWorkout:
    """A session ``minutes`` long from ``start_utc``. ``duration_minutes`` is the
    recorded duration, which on a Watch workout leaves pauses out; it defaults to the
    whole recorded time."""
    duration = minutes if duration_minutes is None else duration_minutes
    return StoredWorkout(
        id=f"w{next(_ids)}",
        activity_type=kind,
        start=start_utc,
        end=start_utc + timedelta(minutes=minutes),
        duration_s=duration * 60,
        active_energy_kcal=energy,
        heart_rate_average_bpm=None,
        heart_rate_maximum_bpm=None,
        source_name="Test Watch",
        tz=tz,
    )


def _utc(day: int, hour: int, minute: int = 0) -> datetime:
    """A moment in March 2026, in UTC. 2026-03-02 is a Monday."""
    return datetime(2026, 3, day, hour, minute, tzinfo=UTC)


# --- 1. local_start -------------------------------------------------------------------


def test_local_start_converts_to_a_named_zone():
    """Sunday 22:30 UTC is Monday 00:30 in Johannesburg (UTC+2)."""
    start = local_start(_session(_utc(1, 22, 30), tz=JOBURG))

    assert (start.date(), start.hour, start.minute) == (date(2026, 3, 2), 0, 30)
    assert start.utcoffset() == timedelta(hours=2)


@pytest.mark.parametrize(
    ("tz", "offset"),
    [
        ("+0200", timedelta(hours=2)),
        ("-0500", timedelta(hours=-5)),
        ("+0530", timedelta(hours=5, minutes=30)),
    ],
)
def test_local_start_accepts_a_bare_offset(tz, offset):
    """Some workouts store an offset such as "+0200" instead of a zone name, when the
    export had no HKTimeZone. Negative offsets and half-hour offsets work too."""
    start = local_start(_session(_utc(1, 22, 30), tz=tz))

    assert start.utcoffset() == offset
    assert start == _utc(1, 22, 30)  # the same moment, only shown in another zone


@pytest.mark.parametrize("tz", ["Mars/Olympus_Mons", "GMT+0200", "+02x0", "/etc/localtime"])
def test_an_unrecognised_zone_is_refused_without_quoting_it(tz):
    """A stored tz this machine cannot read (from a database written elsewhere; the
    importer itself stores only readable ones) is refused. The error names the
    workout, not the value, and carries no error that does."""
    workout = _session(_utc(1, 22, 30), tz=tz)

    with pytest.raises(MalformedRecordError) as caught:
        local_start(workout)

    assert workout.id in str(caught.value)
    assert "import again" in str(caught.value)  # re-importing alone keeps the old row
    assert tz not in str(caught.value)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


# --- 2. weekly_summary ----------------------------------------------------------------


def test_no_sessions_gives_no_weeks():
    assert weekly_summary([]) == []


def test_two_sessions_in_one_week_are_added_together():
    """Monday and Thursday of one week: one entry with both sessions' minutes and
    energy added, and a count per activity type."""
    sessions = [
        _session(_utc(2, 6), 30, kind=RUN, energy=300.0),
        _session(_utc(5, 6), 60, kind=SURF, energy=450.0),
    ]

    assert weekly_summary(sessions) == [
        WeekSummary(
            week_start=date(2026, 3, 2),
            sessions=2,
            minutes=90.0,
            active_energy_kcal=750.0,
            sessions_by_type={RUN: 1, SURF: 1},
        )
    ]


def test_weeks_are_split_at_local_midnight_not_utc():
    """Sunday 23:30 and Monday 00:30 in Johannesburg are both on Sunday in UTC
    (21:30 and 22:30). Bucketed by local time they fall in two weeks; a version that
    used the UTC date would put both in the first week."""
    sunday_night = _session(_utc(8, 21, 30))  # Sun 8 Mar 23:30 local
    monday_early = _session(_utc(8, 22, 30))  # Mon 9 Mar 00:30 local

    weeks = weekly_summary([sunday_night, monday_early])

    assert [(w.week_start, w.sessions) for w in weeks] == [
        (date(2026, 3, 2), 1),
        (date(2026, 3, 9), 1),
    ]


@pytest.mark.parametrize(
    ("start_utc", "week_start"),
    [
        (datetime(2026, 7, 5, 23, 30, tzinfo=UTC), date(2026, 7, 6)),  # Mon 00:30 BST
        (datetime(2026, 1, 4, 23, 30, tzinfo=UTC), date(2025, 12, 29)),  # Sun 23:30 GMT
    ],
)
def test_a_named_zone_follows_daylight_saving(start_utc, week_start):
    """Sunday 23:30 UTC is Monday 00:30 in London in July (BST, UTC+1), so it falls in
    the next week; in January it is still Sunday 23:30 (GMT), so the same week.
    Johannesburg alone cannot catch a version that reads a zone name as one fixed
    offset all year."""
    (week,) = weekly_summary([_session(start_utc, tz="Europe/London")])

    assert week.week_start == week_start


def test_a_session_belongs_to_the_week_of_its_start():
    """A session that starts Sunday 23:00 local and runs past midnight counts in the
    week it started in."""
    weeks = weekly_summary([_session(_utc(8, 21), 120)])  # Sun 23:00 to Mon 01:00 local

    assert [w.week_start for w in weeks] == [date(2026, 3, 2)]


def test_empty_weeks_between_sessions_are_included():
    """Weeks with no session between the first and the last appear with zero
    sessions and minutes, no energy figure and no types, so gaps show in a trend.
    Nothing is added before the first week or after the last."""
    weeks = weekly_summary([_session(_utc(4, 6)), _session(_utc(18, 6))])  # Wed, Wed

    assert [w.week_start for w in weeks] == [date(2026, 3, 2), date(2026, 3, 9), date(2026, 3, 16)]
    assert weeks[1] == WeekSummary(
        week_start=date(2026, 3, 9),
        sessions=0,
        minutes=0.0,
        active_energy_kcal=None,
        sessions_by_type={},
    )


def test_energy_adds_the_known_figures_and_is_absent_when_none_is_known():
    """A Yoga-like session has no energy figure. In a week with one known and one
    unknown, the total is the known one; in a week with only unknowns, it is None,
    not 0, which would read as a measured zero. A measured zero stays 0.0."""
    weeks = weekly_summary(
        [
            _session(_utc(2, 6), energy=300.0),
            _session(_utc(3, 6), energy=None),
            _session(_utc(9, 6), energy=None),
            _session(_utc(16, 6), energy=0.0),
        ]
    )

    assert [w.active_energy_kcal for w in weeks] == [300.0, None, 0.0]


def test_minutes_are_the_recorded_duration_not_the_span():
    """A 60-minute Watch recording with 15 minutes paused has a duration of 45
    minutes; the summary counts 45. With no duration stored, the recorded span is used.
    A stored duration of 0 counts as 0, not as missing."""
    paused = _session(_utc(2, 6), 60, duration_minutes=45)
    no_duration = replace(_session(_utc(3, 6), 20), duration_s=None)
    zero_duration = _session(_utc(4, 6), 10, duration_minutes=0)

    (week,) = weekly_summary([paused, no_duration, zero_duration])

    assert week.minutes == 45.0 + 20.0 + 0.0


def test_result_does_not_depend_on_input_order():
    """Every order of the same sessions gives the same summary, down to the order of
    the types in sessions_by_type (dict equality ignores key order; a report does not)."""
    sessions = [_session(_utc(d, 6), kind=k) for d, k in [(2, RUN), (10, SURF), (3, SURF)]]

    results = [
        [(w, list(w.sessions_by_type)) for w in weekly_summary(list(order))]
        for order in itertools.permutations(sessions)
    ]

    assert [(w.week_start, w.sessions) for w, _ in results[0]] == [
        (date(2026, 3, 2), 2),
        (date(2026, 3, 9), 1),
    ]
    assert all(result == results[0] for result in results)


def test_weeks_are_consecutive_mondays_and_account_for_every_session():
    """Invariants on a spread of sessions across zones: every week_start is a Monday,
    each is seven days after the last, and the week counts add up to the input."""
    sessions = [
        _session(_utc(1 + 3 * i, (5 * i) % 24), tz=tz)
        for i, tz in enumerate([JOBURG, "+0200", "-0500", JOBURG, "+0530", JOBURG, "-0500"])
    ]

    weeks = weekly_summary(sessions)

    assert all(w.week_start.weekday() == 0 for w in weeks)
    assert all(
        b.week_start - a.week_start == timedelta(days=7) for a, b in itertools.pairwise(weeks)
    )
    assert sum(w.sessions for w in weeks) == len(sessions)
    assert all(sum(w.sessions_by_type.values()) == w.sessions for w in weeks)


# --- 3. weekly_summary_from_db --------------------------------------------------------

# One surf session recorded by a surf app and by the Watch, plus a run two days
# later. Only the Watch writes heart rate. The summary must show 2 sessions, not 3.
# The app filed the surf under "Other", starts first and is as long as the Watch's
# record, so it would win the tie-break: only the heart-rate counts pick the Watch's.
_TWICE_RECORDED_EXPORT = """<?xml version="1.0" encoding="UTF-8"?>
<!-- SYNTHETIC: one session recorded by two sources, and one other session. -->
<HealthData>
 <Workout workoutActivityType="HKWorkoutActivityTypeOther" duration="60" durationUnit="min"
          sourceName="Surf App" startDate="2026-03-02 07:00:00 +0200" endDate="2026-03-02 08:00:00 +0200">
  <MetadataEntry key="HKTimeZone" value="Africa/Johannesburg"/>
 </Workout>
 <Workout workoutActivityType="HKWorkoutActivityTypeSurfingSports" duration="60" durationUnit="min"
          sourceName="Test Watch" startDate="2026-03-02 07:01:00 +0200" endDate="2026-03-02 08:01:00 +0200">
  <MetadataEntry key="HKTimeZone" value="Africa/Johannesburg"/>
 </Workout>
 <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="30" durationUnit="min"
          sourceName="Test Watch" startDate="2026-03-04 06:00:00 +0200" endDate="2026-03-04 06:30:00 +0200">
  <MetadataEntry key="HKTimeZone" value="Africa/Johannesburg"/>
 </Workout>
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" unit="count/min" value="120"
         startDate="2026-03-02 07:30:00 +0200" endDate="2026-03-02 07:30:00 +0200"/>
</HealthData>
"""

# Two runs with no HKTimeZone, on either side of local midnight at +0200. Both start
# on Sunday 8 March in UTC (21:30 and 22:30).
_NO_ZONE_NAME_EXPORT = """<?xml version="1.0" encoding="UTF-8"?>
<!-- SYNTHETIC: workouts without an HKTimeZone entry. -->
<HealthData>
 <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="20" durationUnit="min"
          sourceName="Test Watch" startDate="2026-03-08 23:30:00 +0200" endDate="2026-03-08 23:50:00 +0200"/>
 <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="20" durationUnit="min"
          sourceName="Test Watch" startDate="2026-03-09 00:30:00 +0200" endDate="2026-03-09 00:50:00 +0200"/>
</HealthData>
"""


def _import(tmp_path, export_text: str):
    """Run the real import on ``export_text`` and return the database path."""
    export = tmp_path / "export.xml"
    export.write_text(export_text, encoding="utf-8")
    db = tmp_path / "health.sqlite"
    import_export(export, db)
    import_samples(export, db)
    return db


def test_from_the_database_a_twice_recorded_session_counts_once(tmp_path):
    """Through the real import: the workouts table holds 3 rows, but the week shows 2
    sessions and 90 minutes, and the surf counts as the Watch's type. A summary that
    read the table without distinct_sessions would show 3 sessions and 150 minutes;
    one that left out the heart-rate counts would keep the app's "Other"."""
    conn = connect(_import(tmp_path, _TWICE_RECORDED_EXPORT))
    try:
        assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 3
        weeks = weekly_summary_from_db(conn)
    finally:
        conn.close()

    assert [(w.week_start, w.sessions, w.minutes) for w in weeks] == [(date(2026, 3, 2), 2, 90.0)]
    assert weeks[0].sessions_by_type == {SURF: 1, RUN: 1}


def test_from_the_database_a_workout_without_a_zone_name_uses_its_stored_offset(tmp_path):
    """With no HKTimeZone the importer stores the startDate offset ("+0200"), and the
    summary reads that back. The two runs are both on Sunday in UTC, so they fall in
    two weeks only if the stored offset is read the way it was written."""
    conn = connect(_import(tmp_path, _NO_ZONE_NAME_EXPORT))
    try:
        assert conn.execute("SELECT DISTINCT tz FROM workouts").fetchall() == [("+0200",)]
        weeks = weekly_summary_from_db(conn)
    finally:
        conn.close()

    assert [(w.week_start, w.sessions) for w in weeks] == [
        (date(2026, 3, 2), 1),
        (date(2026, 3, 9), 1),
    ]


def test_from_the_database_an_unloadable_zone_name_is_placed_by_its_offset(tmp_path):
    """An HKTimeZone this machine cannot load ("GMT+0200", as an app might write) is
    stored as the startDate offset instead, so the session is still counted, in the
    week its offset gives, and the summary does not stop. The bad name is on the
    Monday 00:30 run, which is still Sunday in UTC, so a fallback to UTC would put
    both runs in one week."""
    export = _NO_ZONE_NAME_EXPORT.replace(
        'endDate="2026-03-09 00:50:00 +0200"/>',
        'endDate="2026-03-09 00:50:00 +0200">\n'
        '  <MetadataEntry key="HKTimeZone" value="GMT+0200"/>\n </Workout>',
    )
    assert export.count("GMT+0200") == 1
    conn = connect(_import(tmp_path, export))
    try:
        weeks = weekly_summary_from_db(conn)
    finally:
        conn.close()

    assert [(w.week_start, w.sessions) for w in weeks] == [
        (date(2026, 3, 2), 1),
        (date(2026, 3, 9), 1),
    ]


def test_from_the_database_an_unreadable_stored_zone_stops_the_summary(tmp_path):
    """The importer stores only zones this machine can read, but a database written
    elsewhere (another machine's time zone database, or an older version of this
    code) can still hold one it cannot. That stops the whole summary with
    MalformedRecordError: skipping the session would make the weekly counts quietly
    stop adding up to the stored sessions."""
    conn = connect(_import(tmp_path, _NO_ZONE_NAME_EXPORT))
    try:
        with conn:
            conn.execute("UPDATE workouts SET tz = 'GMT+0200'")  # as if written elsewhere
        with pytest.raises(MalformedRecordError):
            weekly_summary_from_db(conn)
    finally:
        conn.close()


# --- 4. pad_to_week_of ----------------------------------------------------------------

# Sessions in the weeks of Monday 2 March and Monday 9 March 2026.
_TWO_WEEKS = [_session(_utc(4, 6)), _session(_utc(10, 6))]


@pytest.mark.parametrize(
    ("day", "last_week"),
    [
        (date(2026, 3, 25), date(2026, 3, 23)),  # a Wednesday: its own week is included
        (date(2026, 3, 29), date(2026, 3, 23)),  # a Sunday: still that week
        (date(2026, 3, 30), date(2026, 3, 30)),  # the next Monday starts a new week
    ],
)
def test_padding_adds_empty_weeks_up_to_the_week_of_the_day(day, last_week):
    """A break in training up to today shows as empty weeks, not as nothing."""
    weeks = weekly_summary(_TWO_WEEKS)

    padded = pad_to_week_of(weeks, day)

    assert padded[: len(weeks)] == weeks
    assert padded[-1].week_start == last_week
    assert all(
        b.week_start - a.week_start == timedelta(days=7) for a, b in itertools.pairwise(padded)
    )
    assert all(w.sessions == 0 and w.active_energy_kcal is None for w in padded[len(weeks) :])


@pytest.mark.parametrize("day", [date(2026, 3, 12), date(2026, 3, 1)])
def test_padding_never_drops_a_week(day):
    """A day in or before the last week (a session dated in the future) leaves the
    weeks as they are."""
    weeks = weekly_summary(_TWO_WEEKS)

    assert pad_to_week_of(weeks, day) == weeks


def test_padding_no_weeks_gives_no_weeks():
    """With no sessions there is no first week to pad from."""
    assert pad_to_week_of([], date(2026, 3, 25)) == []
