"""Tests that a stored timestamp the code cannot read is refused without quoting it.

Python's own error from ``datetime.fromisoformat`` quotes the text, and an MCP tool's
error text reaches the AI, so every place a stored time is read back goes through
``db.stored_time``. Only a database written elsewhere can hold such a value, so these
tests corrupt one directly. Synthetic data only, never real health data.
"""

from contextlib import closing
from datetime import UTC, datetime

import pytest

from apple_health_mcp.analyze import load_workouts, readings_around, weekly_summary_from_db
from apple_health_mcp.db import StoredValueError, connect, stored_time
from apple_health_mcp.ingest import import_export, import_samples, main
from apple_health_mcp.parse import MalformedRecordError

BAD = "2026-13-45T99:00:00+00:00"  # not a date; it must never appear in a message

_EXPORT = """<?xml version="1.0" encoding="UTF-8"?>
<!-- SYNTHETIC: one run with one heart-rate reading. -->
<HealthData>
 <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="30" durationUnit="min"
          sourceName="Test Watch" startDate="2026-03-02 06:00:00 +0200" endDate="2026-03-02 06:30:00 +0200"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" unit="count/min" value="140"
         startDate="2026-03-02 06:10:00 +0200" endDate="2026-03-02 06:10:00 +0200"/>
</HealthData>
"""


def _database(tmp_path):
    export = tmp_path / "export.xml"
    export.write_text(_EXPORT, encoding="utf-8")
    db = tmp_path / "health.sqlite"
    import_export(export, db)
    import_samples(export, db)
    return db, export


def _corrupt(db, sql: str, bad: str = BAD) -> tuple[str, int]:
    """Run ``sql`` (which sets a bad value on the table's one row) and return that
    row's id and SQLite rowid."""
    with closing(connect(db)) as conn, conn:
        conn.execute(sql, (bad,))
        table = "workouts" if "workouts" in sql else "samples"
        return conn.execute(f"SELECT id, rowid FROM {table}").fetchone()


def _assert_refused_without_quoting(caught, names: str, bad: str = BAD) -> None:
    message = str(caught.value)
    assert isinstance(caught.value, StoredValueError)
    assert names in message
    assert bad not in message and bad[:10] not in message
    assert caught.value.__context__ is None and caught.value.__cause__ is None


def test_stored_time_reads_back_what_the_database_holds():
    moment = datetime(2026, 3, 2, 4, 0, tzinfo=UTC)

    assert stored_time(moment.isoformat(), "workouts", "w1") == moment


@pytest.mark.parametrize(
    "text",
    [
        BAD,
        "2026-03-02T04:00:00",
        "2026-03-02T06:00:00+02:00",
        "2026-03-02T04:00:00Z",
        "20260302T040000+0000",
        "",
        None,
    ],
    ids=["not-a-date", "naive", "another-offset", "z-suffix", "basic-format", "empty", "null"],
)
def test_stored_time_refuses_what_it_cannot_read(text):
    """Only the exact form this code writes is read. The queries compare stored times
    as text, so a naive time, another offset, a "Z" or the basic ISO format would
    sort wrongly and silently, rather than being refused."""
    with pytest.raises(StoredValueError) as caught:
        stored_time(text, "workouts", "w1")

    assert "workouts row w1" in str(caught.value)
    assert caught.value.__context__ is None
    assert isinstance(caught.value, MalformedRecordError)  # callers that catch the parent


@pytest.mark.parametrize("column", ["start_utc", "end_utc"])
def test_loading_workouts_refuses_an_unreadable_time_without_quoting_it(tmp_path, column):
    db, _ = _database(tmp_path)
    workout_id, _rowid = _corrupt(db, f"UPDATE workouts SET {column} = ?")

    with closing(connect(db)) as conn, pytest.raises(MalformedRecordError) as caught:
        load_workouts(conn)

    _assert_refused_without_quoting(caught, f"workouts row {workout_id}")


# Still sorts inside the run's window as text, so the query selects it, but is not
# a timestamp: it reaches stored_time through the start column.
_BAD_IN_WINDOW = "2026-03-02T04:10:00+00:00x"


@pytest.mark.parametrize(("column", "bad"), [("start_utc", _BAD_IN_WINDOW), ("end_utc", BAD)])
def test_reading_heart_rate_refuses_an_unreadable_time_naming_only_the_rowid(tmp_path, column, bad):
    """The message locates the reading by its SQLite rowid. Its id is an unsalted hash
    of its type, times, value and source, so naming it would let anyone holding the
    message recover the reading's time and heart rate by guessing."""
    db, _ = _database(tmp_path)
    sample_id, rowid = _corrupt(db, f"UPDATE samples SET {column} = ?", bad)

    with closing(connect(db)) as conn:
        (workout,) = load_workouts(conn)
        with pytest.raises(MalformedRecordError) as caught:
            readings_around(conn, workout)

    _assert_refused_without_quoting(caught, f"samples row {rowid};", bad)
    assert sample_id not in str(caught.value)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("column", ["start_utc", "end_utc"])
def test_the_import_refuses_an_unreadable_stored_time_with_a_one_line_message(
    tmp_path, capsys, column, strict
):
    """The heart-rate pass reads the workouts' windows back from the database. A
    corrupted time there stops it with the rebuild advice, not a traceback, and,
    with or without --strict, not the --strict hint, which is about the export."""
    db, export = _database(tmp_path)
    _corrupt(db, f"UPDATE workouts SET {column} = ?")

    assert main([str(export), "--db", str(db)] + (["--strict"] if strict else [])) == 1

    err = capsys.readouterr().err
    assert "unreadable timestamp in workouts row" in err
    assert "import again to rebuild it" in err
    assert "--strict" not in err
    assert BAD not in err


def test_the_strict_hint_still_comes_with_a_malformed_record_in_the_export(tmp_path, capsys):
    """The other side of the hint's rule: under --strict, a malformed export record
    stops the import and the message says how to skip and count it instead."""
    export = tmp_path / "export.xml"
    export.write_text(
        _EXPORT.replace(
            'value="140"\n         startDate="2026-03-02 06:10:00 +0200"',
            'value="140"\n         startDate="NOT-A-DATE"',
        ),
        encoding="utf-8",
    )
    assert "NOT-A-DATE" in export.read_text()

    assert main([str(export), "--db", str(tmp_path / "health.sqlite"), "--strict"]) == 1

    assert "(run without --strict to skip and count it)" in capsys.readouterr().err


@pytest.mark.parametrize(
    "sql",
    ["UPDATE workouts SET tz = 'GMT+0200'", "UPDATE samples SET value = 'NaN'"],
    ids=["time-zone", "heart-rate-value"],
)
def test_every_unreadable_stored_value_is_a_stored_value_error(tmp_path, sql):
    """A time zone or a heart-rate value the code cannot read is a StoredValueError
    too, like a timestamp: a value already in the database, as opposed to a record in
    the export, which callers such as the import's --strict hint tell apart."""
    db, _ = _database(tmp_path)
    with closing(connect(db)) as conn, conn:
        conn.execute(sql)

    with closing(connect(db)) as conn, pytest.raises(StoredValueError):
        (workout,) = load_workouts(conn)
        weekly_summary_from_db(conn)
        readings_around(conn, workout)
