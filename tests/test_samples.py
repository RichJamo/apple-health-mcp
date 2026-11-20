"""Tests for the samples half: storage, window-matching, and skip-and-report.

Three groups:
  1. db.py  — upsert_samples stores readings idempotently and round-trips fields.
  2. ingest — _samples_in_windows keeps a reading iff some workout window holds it.
  3. parse  — iter_records skips malformed records (reporting type only) or, in
              strict mode, raises an error that carries no raw text.

Synthetic data only; constructed in-memory, never real health data. The production
path that adds the recovery tail (_windows_from_db) is tested end to end in
test_ingest.py; here the matcher is handed windows whose ends already include it.
"""

import traceback
from datetime import UTC, datetime, timedelta, timezone
from io import BytesIO

import pytest

from apple_health_mcp.db import Sample, connect, init_schema, sample_id, upsert_samples
from apple_health_mcp.ingest import RECOVERY_WINDOW, _samples_in_windows
from apple_health_mcp.parse import MalformedRecordError, Record, iter_records

HEART_RATE = "HKQuantityTypeIdentifierHeartRate"


@pytest.fixture
def conn():
    c = connect(":memory:")
    init_schema(c)
    yield c
    c.close()


def _sample(start: datetime, value: str = "100", workout_id: str | None = "W1") -> Sample:
    return Sample(
        type=HEART_RATE,
        start=start,
        value=value,
        unit="count/min",
        source_name="Test Watch",
        workout_id=workout_id,
    )


def _hr(start: datetime, value: str = "100") -> Record:
    return Record(
        type=HEART_RATE,
        start=start,
        end=start,
        value=value,
        unit="count/min",
        source_name="Test Watch",
    )


def _at(hour: int, minute: int) -> datetime:
    return datetime(2026, 1, 1, hour, minute, tzinfo=UTC)


def _window(start: datetime, end: datetime, workout_id: str):
    """A keep-window as _windows_from_db builds it: the end already includes the tail."""
    return (start, end + RECOVERY_WINDOW, workout_id)


# --- 1. storage ---------------------------------------------------------------


def test_upsert_samples_is_idempotent(conn):
    """Importing the same readings twice stores them once; the second pass inserts 0."""
    samples = [_sample(_at(10, 0), value="100"), _sample(_at(10, 1), value="101")]

    assert upsert_samples(conn, samples) == 2
    assert upsert_samples(conn, samples) == 0  # same ids -> nothing new
    assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 2


def test_sample_fields_round_trip(conn):
    """A reading's fields survive the write; start is stored as UTC, link preserved."""
    start = datetime(2026, 1, 1, 8, 5, 11, tzinfo=timezone(timedelta(hours=2)))  # +0200

    upsert_samples(conn, [_sample(start, value="92", workout_id="W1")])

    row = conn.execute(
        "SELECT type, start_utc, value, unit, source_name, workout_id FROM samples"
    ).fetchone()
    assert row == (
        HEART_RATE,
        "2026-01-01T06:05:11+00:00",  # +0200 converted to UTC
        "92",
        "count/min",
        "Test Watch",
        "W1",
    )


def test_same_instant_with_a_different_offset_is_one_reading(conn):
    """The id is keyed on the UTC instant, so one reading written as +02:00 in one
    export and +00:00 in another is stored once, not counted twice."""
    sast = datetime(2026, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    utc = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)  # the same moment

    assert sample_id(_sample(sast)) == sample_id(_sample(utc))
    assert upsert_samples(conn, [_sample(sast), _sample(utc)]) == 1


# --- 2. window-matching -------------------------------------------------------


def test_window_bounds_are_inclusive_and_outside_readings_dropped():
    """Given one window (tail already in its end), readings from its start to its end
    are kept, both bounds inclusive; readings before or after are dropped."""
    windows = [_window(_at(10, 0), _at(10, 30), "W1")]  # keep 10:00 .. 10:35

    records = [
        _hr(_at(9, 59)),  # before start -> drop
        _hr(_at(10, 0)),  # exactly at start -> keep
        _hr(_at(10, 15)),  # inside -> keep
        _hr(_at(10, 35)),  # exactly at tail end -> keep
        _hr(_at(10, 36)),  # past the tail -> drop
    ]

    kept = list(_samples_in_windows(windows, records))

    assert [s.start for s in kept] == [_at(10, 0), _at(10, 15), _at(10, 35)]
    assert all(s.workout_id == "W1" for s in kept)


def test_reading_between_two_workouts_is_dropped_and_each_tagged_correctly():
    """With two separate workouts, a reading in the gap is dropped, and a reading
    inside the later one is tagged with that workout, not the earlier."""
    windows = sorted(
        [_window(_at(10, 0), _at(10, 30), "W1"), _window(_at(12, 0), _at(12, 30), "W2")]
    )

    kept = list(_samples_in_windows(windows, [_hr(_at(11, 0)), _hr(_at(12, 10))]))

    assert [(s.start, s.workout_id) for s in kept] == [(_at(12, 10), "W2")]


def test_nested_workout_does_not_hide_the_outer_workouts_readings():
    """A short workout B inside a longer one A must not swallow A's later readings.

    The latest-starting window (B) misses 10:40 and 11:03, but A still holds them —
    the tail at 11:03 is exactly what recovery needs. A plain "latest start" lookup
    drops both; the furthest-reaching fallback keeps them, tagged A.
    """
    windows = sorted(
        [_window(_at(10, 0), _at(11, 0), "A"), _window(_at(10, 10), _at(10, 20), "B")]
    )  # A keeps 10:00 .. 11:05, B keeps 10:10 .. 10:25

    records = [
        _hr(_at(10, 5)),
        _hr(_at(10, 15)),
        _hr(_at(10, 40)),
        _hr(_at(11, 3)),
        _hr(_at(11, 6)),
    ]

    kept = [(s.start, s.workout_id) for s in _samples_in_windows(windows, records)]

    assert kept == [
        (_at(10, 5), "A"),
        (_at(10, 15), "B"),  # inside both; the latest-starting one wins
        (_at(10, 40), "A"),  # after B's window, still inside A
        (_at(11, 3), "A"),  # A's recovery tail
    ]  # 11:06 is past A's tail -> dropped


# --- 3. skip-and-report in the parser -----------------------------------------

_GOOD = (
    b'<Record type="HKQuantityTypeIdentifierHeartRate" '
    b'startDate="2026-01-01 10:00:00 +0000" endDate="2026-01-01 10:00:00 +0000" '
    b'value="100" unit="count/min"/>'
)
_BAD = (  # startDate is not a valid Apple timestamp
    b'<Record type="HKQuantityTypeIdentifierHeartRate" '
    b'startDate="NOT-A-DATE" endDate="2026-01-01 10:00:00 +0000" '
    b'value="110" unit="count/min"/>'
)


def _export(*records: bytes) -> BytesIO:
    return BytesIO(b"<HealthData>" + b"".join(records) + b"</HealthData>")


def test_on_skip_skips_malformed_and_reports_type_only():
    """A malformed record is skipped, the good ones still yielded, and on_skip is
    told only the record's type — never the raw value or timestamp (privacy)."""
    skipped: list[str] = []

    records = list(iter_records(_export(_GOOD, _BAD), on_skip=skipped.append))

    assert len(records) == 1  # only the good record came through
    assert records[0].value == "100"
    assert skipped == [HEART_RATE]  # the type, not "NOT-A-DATE" or "110"


def test_strict_mode_raises_a_redacted_error():
    """Without on_skip (strict), a malformed record aborts with MalformedRecordError,
    and the rendered traceback — what a terminal, log or AI would see — carries the
    record type but not the raw timestamp that Python's own parse error quotes."""
    with pytest.raises(MalformedRecordError) as caught:
        list(iter_records(_export(_GOOD, _BAD)))

    rendered = "".join(traceback.format_exception(caught.value))
    assert HEART_RATE in rendered
    assert "NOT-A-DATE" not in rendered
