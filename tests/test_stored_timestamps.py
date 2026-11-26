"""Tests that a stored timestamp the code cannot read is refused without quoting it.

Python's own error from ``datetime.fromisoformat`` quotes the text, and an MCP tool's
error text reaches the AI, so every place a stored time is read back goes through
``db.stored_time``. Only a database written elsewhere can hold such a value, so these
tests corrupt one directly. Synthetic data only, never real health data.
"""

from contextlib import closing
from datetime import UTC, datetime

import pytest

from apple_health_mcp.analyze import load_workouts, readings_around
from apple_health_mcp.db import connect, stored_time
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


def _corrupt(db, sql: str) -> str:
    """Run ``sql`` (which sets a bad value) and return the id of the row it changed."""
    with closing(connect(db)) as conn, conn:
        conn.execute(sql, (BAD,))
        table = "workouts" if "workouts" in sql else "samples"
        return conn.execute(f"SELECT id FROM {table}").fetchone()[0]


def _assert_refused_without_quoting(caught, row_id: str) -> None:
    message = str(caught.value)
    assert row_id in message
    assert BAD not in message and "2026-13-45" not in message
    assert caught.value.__context__ is None and caught.value.__cause__ is None


def test_stored_time_reads_back_what_the_database_holds():
    moment = datetime(2026, 3, 2, 4, 0, tzinfo=UTC)

    assert stored_time(moment.isoformat(), "workouts", "w1") == moment


@pytest.mark.parametrize(
    "text",
    [BAD, "2026-03-02T04:00:00", "", None],
    ids=["not-a-date", "naive", "empty", "null"],
)
def test_stored_time_refuses_what_it_cannot_read(text):
    """A naive timestamp is refused too: compared with an aware one later it would
    raise somewhere less clear."""
    with pytest.raises(MalformedRecordError) as caught:
        stored_time(text, "workouts", "w1")

    assert "workouts row w1" in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize("column", ["start_utc", "end_utc"])
def test_loading_workouts_refuses_an_unreadable_time_without_quoting_it(tmp_path, column):
    db, _ = _database(tmp_path)
    workout_id = _corrupt(db, f"UPDATE workouts SET {column} = ?")

    with closing(connect(db)) as conn, pytest.raises(MalformedRecordError) as caught:
        load_workouts(conn)

    _assert_refused_without_quoting(caught, workout_id)


def test_reading_heart_rate_refuses_an_unreadable_time_without_quoting_it(tmp_path):
    """The end, because the query selects readings by their start and a corrupted
    start would simply fall outside the window."""
    db, _ = _database(tmp_path)
    sample_id = _corrupt(db, "UPDATE samples SET end_utc = ?")

    with closing(connect(db)) as conn:
        (workout,) = load_workouts(conn)
        with pytest.raises(MalformedRecordError) as caught:
            readings_around(conn, workout)

    _assert_refused_without_quoting(caught, sample_id)


def test_the_import_refuses_an_unreadable_stored_time_with_a_one_line_message(tmp_path, capsys):
    """The heart-rate pass reads the workouts' windows back from the database. A
    corrupted time there stops it with the rebuild advice, not a traceback, and not
    the --strict hint, which is about the export."""
    db, export = _database(tmp_path)
    _corrupt(db, "UPDATE workouts SET end_utc = ?")

    assert main([str(export), "--db", str(db)]) == 1

    err = capsys.readouterr().err
    assert "unreadable timestamp in workouts row" in err
    assert "import again to rebuild it" in err
    assert "--strict" not in err
    assert BAD not in err
