"""Tests for the schema change: span readings keep their end, and the database
records which schema version built it.

Two groups:
  1. spans  — a sample's end is stored in UTC, an instant's end equals its start,
              a span and an instant that start together are two readings, and the
              real import stores a span's end.
  2. version — a fresh database is stamped with SCHEMA_VERSION, repeating init is
              safe, a database at another version is refused untouched, and the CLI
              turns that refusal into exit 1 and a plain message.

Synthetic data only, never real health data.
"""

import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from apple_health_mcp.db import (
    SCHEMA_VERSION,
    Sample,
    SchemaVersionError,
    connect,
    init_schema,
    sample_id,
    upsert_samples,
    workout_id,
)
from apple_health_mcp.ingest import HEART_RATE, import_export, import_samples, main
from apple_health_mcp.parse import Workout

FIXTURE = Path(__file__).parent / "fixtures" / "sample_export.xml"
SAST = timezone(timedelta(hours=2))  # +0200, so a missed UTC conversion shows


@pytest.fixture
def conn():
    c = connect(":memory:")
    init_schema(c)
    yield c
    c.close()


def _sample(start: datetime, end: datetime | None = None) -> Sample:
    """A heart-rate reading; leave ``end`` out for an instant, as the import would not."""
    return Sample(
        type=HEART_RATE,
        start=start,
        value="150",
        unit="count/min",
        source_name="Test Watch",
        workout_id="W1",
        end=end,
    )


def _user_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def _schema(conn: sqlite3.Connection) -> list[tuple]:
    """Every table and index with the SQL that made it, to compare before and after."""
    return conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY name").fetchall()


def _sample_columns(conn: sqlite3.Connection) -> list[str]:
    return [row[1] for row in conn.execute("PRAGMA table_info(samples)")]


def _build_unversioned_db(conn: sqlite3.Connection) -> None:
    """Tables in the shape this code wrote before the version stamp: no ``end_utc``
    on samples, and ``user_version`` left at SQLite's default of 0."""
    conn.execute("CREATE TABLE workouts (id TEXT PRIMARY KEY, activity_type TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE samples (id TEXT PRIMARY KEY, type TEXT NOT NULL,"
        " start_utc TEXT NOT NULL, value TEXT, unit TEXT, source_name TEXT NOT NULL,"
        " workout_id TEXT)"
    )
    conn.commit()


# --- 1. spans ------------------------------------------------------------------


def test_span_end_is_stored_in_utc(conn):
    """A span's end is converted to UTC like its start, not stored with its offset.

    10:20 to 10:34 at +0200 is 08:20 to 08:34 UTC. Fails if ``end_utc`` keeps the
    original offset, or if the start is written into it instead of the end.
    """
    span = _sample(
        datetime(2026, 3, 1, 10, 20, tzinfo=SAST), datetime(2026, 3, 1, 10, 34, tzinfo=SAST)
    )

    upsert_samples(conn, [span])

    row = conn.execute("SELECT start_utc, end_utc FROM samples").fetchone()
    assert row == ("2026-03-01T08:20:00+00:00", "2026-03-01T08:34:00+00:00")


def test_instant_reading_ends_when_it_starts(conn):
    """Left out, a sample's end defaults to its start, both on the object and in the
    stored row, so an instant reading never reaches the NOT NULL column as None."""
    instant = _sample(datetime(2026, 3, 1, 10, 20, tzinfo=SAST))

    assert instant.end == instant.start

    upsert_samples(conn, [instant])
    row = conn.execute("SELECT start_utc, end_utc FROM samples").fetchone()
    assert row == ("2026-03-01T08:20:00+00:00", "2026-03-01T08:20:00+00:00")


def test_span_and_instant_with_the_same_start_are_two_readings(conn):
    """Same type, start, value and source, but one is a 14-minute span: two ids, two
    rows. If the end were left out of the id, the second would be silently dropped
    by INSERT OR IGNORE and the import would report 1."""
    start = datetime(2026, 3, 1, 10, 20, tzinfo=UTC)
    instant = _sample(start)
    span = _sample(start, start + timedelta(minutes=14))

    assert sample_id(instant) != sample_id(span)
    assert upsert_samples(conn, [instant, span]) == 2


def test_import_stores_a_spans_end(tmp_path):
    """Through the real parse-and-import path, a heart-rate record whose endDate is
    later than its startDate keeps that end. Fails if ingest stops passing the
    record's end on, because the sample would then default to an instant."""
    export = tmp_path / "export.xml"
    export.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<!-- SYNTHETIC -->\n<HealthData>'
        '<Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="30" '
        'durationUnit="min" sourceName="Test Watch" '
        'startDate="2026-03-01 10:00:00 +0200" endDate="2026-03-01 10:30:00 +0200"/>'
        '<Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" '
        'unit="count/min" value="150" '
        'startDate="2026-03-01 10:20:00 +0200" endDate="2026-03-01 10:34:00 +0200"/>'
        "</HealthData>",
        encoding="utf-8",
    )
    db = tmp_path / "health.sqlite"
    import_export(export, db)

    import_samples(export, db)

    check = connect(db)
    try:
        rows = check.execute("SELECT start_utc, end_utc FROM samples").fetchall()
    finally:
        check.close()
    assert rows == [("2026-03-01T08:20:00+00:00", "2026-03-01T08:34:00+00:00")]


# --- 2. schema version ------------------------------------------------------------


def test_fresh_database_is_stamped_with_the_schema_version():
    """A new database starts at SQLite's default user_version of 0; after init it
    records SCHEMA_VERSION, which is what lets a later run recognise it."""
    fresh = connect(":memory:")
    try:
        assert _user_version(fresh) == 0  # SQLite's default, before init
        init_schema(fresh)
        assert _user_version(fresh) == SCHEMA_VERSION
    finally:
        fresh.close()


def test_repeat_init_is_safe_and_keeps_the_data(conn):
    """Ingest calls init_schema on every run, so on a database it built itself a
    second call must neither raise nor drop what is stored."""
    upsert_samples(conn, [_sample(datetime(2026, 3, 1, 10, 20, tzinfo=UTC))])

    init_schema(conn)  # the second run against the same file

    assert _user_version(conn) == SCHEMA_VERSION
    assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 1


@pytest.mark.parametrize("found", [0, SCHEMA_VERSION + 1], ids=["older", "newer"])
def test_database_at_another_version_is_refused_untouched(found):
    """Tables built at any other version raise SchemaVersionError carrying both
    versions, and init_schema writes nothing first: the stamp is not overwritten, and
    no table, column or index is created or altered (the whole schema is compared
    before and after). Version 0 is the real case, a database from before the stamp.
    """
    old = connect(":memory:")
    try:
        _build_unversioned_db(old)
        old.execute(f"PRAGMA user_version = {found}")  # an int from this test, not input
        schema_before = _schema(old)

        with pytest.raises(SchemaVersionError) as caught:
            init_schema(old)

        assert (caught.value.found, caught.value.expected) == (found, SCHEMA_VERSION)
        assert _user_version(old) == found  # not re-stamped
        assert _schema(old) == schema_before  # nothing created or altered
        assert "end_utc" not in _sample_columns(old)
    finally:
        old.close()


def test_cli_exits_1_with_a_rebuild_message_on_an_old_database(tmp_path, capsys):
    """Given a database from before the version stamp, the CLI stops with exit code 1
    and tells the user to delete and re-import, instead of a traceback or an SQL error
    about a missing column. Nothing is reported as imported."""
    db = tmp_path / "health.sqlite"
    old = connect(db)
    try:
        _build_unversioned_db(old)
    finally:
        old.close()

    rc = main([str(FIXTURE), "--db", str(db)])

    assert rc == 1
    captured = capsys.readouterr()
    assert "Delete it and run the import again" in captured.err
    assert f"this version needs {SCHEMA_VERSION}" in captured.err
    assert captured.out == ""  # no "imported ..." line


def test_ids_are_pinned_to_the_schema_version():
    """Golden ids: two fixed synthetic readings must hash to these exact values.

    If this fails, ``sample_id`` or ``workout_id`` now computes its key differently.
    Every stored row would get a new id, and a re-import into an existing database
    would store it again without any error. So bump SCHEMA_VERSION (an existing
    database is then refused and rebuilt), and update both the version and the ids
    below in the same commit.
    """
    start = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
    span = Sample(
        type=HEART_RATE,
        start=start,
        value="150",
        unit="count/min",
        source_name="Test Watch",
        workout_id=None,
        end=datetime(2026, 3, 1, 8, 14, tzinfo=UTC),
    )
    run = Workout(
        activity_type="HKWorkoutActivityTypeRunning",
        start=start,
        end=datetime(2026, 3, 1, 8, 30, tzinfo=UTC),
        duration_s=1800.0,
        active_energy_kcal=None,
        heart_rate_average_bpm=None,
        heart_rate_maximum_bpm=None,
        source_name="Test Watch",
        metadata={},
    )

    assert (SCHEMA_VERSION, sample_id(span), workout_id(run)) == (
        2,  # bumped for the heart-rate columns; the id keys did not change
        "be1af874e08737fe00bf2019202e242b061a1352990106696a76c3552c25dee3",
        "dc3af6136b4c5cc026e41b995f2f55b5cd4ed3a90857e52e2555216ab6537fe5",
    )
