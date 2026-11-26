"""Tests that each import leaves the database mirroring the export it read.

Everything in the database is derived from the export, so importing a newer export
must also drop what the newer one no longer has: a workout deleted in Apple Health,
the old copy of a workout whose details changed, and the heart rate that belonged to
either. Synthetic exports only, never real health data.
"""

import sqlite3
from contextlib import closing

import pytest

from apple_health_mcp.db import connect, replace_workouts
from apple_health_mcp.ingest import import_export, import_samples, main
from apple_health_mcp.parse import MalformedRecordError, iter_workouts

HEART_RATE = "HKQuantityTypeIdentifierHeartRate"


def _workout(kind: str, start: str, end: str) -> str:
    return (
        f' <Workout workoutActivityType="HKWorkoutActivityType{kind}" duration="30"'
        f' durationUnit="min" sourceName="Test Watch" startDate="2026-03-02 {start} +0200"'
        f' endDate="2026-03-02 {end} +0200"/>\n'
    )


def _reading(at: str, bpm: str) -> str:
    return (
        f' <Record type="{HEART_RATE}" sourceName="Test Watch" unit="count/min" value="{bpm}"'
        f' startDate="2026-03-02 {at} +0200" endDate="2026-03-02 {at} +0200"/>\n'
    )


def _export(*parts: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!-- SYNTHETIC: a mirror-import fixture. -->\n<HealthData>\n"
        + "".join(parts)
        + "</HealthData>\n"
    )


RUN = _workout("Running", "06:00:00", "06:30:00")
SURF = _workout("SurfingSports", "09:00:00", "09:30:00")
RUN_READING = _reading("06:10:00", "140")
SURF_READING = _reading("09:10:00", "120")


def _import(tmp_path, text: str, name: str):
    """Write ``text`` as an export called ``name``, import it into the one test
    database, and return that database's path."""
    export = tmp_path / name
    export.write_text(text, encoding="utf-8")
    db = tmp_path / "health.sqlite"
    import_export(export, db)
    import_samples(export, db)
    return db


def _rows(db, table: str) -> int:
    with closing(connect(db)) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_a_workout_missing_from_a_newer_export_is_removed_with_its_heart_rate(tmp_path):
    """A workout deleted in Apple Health is absent from the next export, so it and the
    readings kept for it leave the database, and the weekly summary stops counting it."""
    _import(tmp_path, _export(RUN, SURF, RUN_READING, SURF_READING), "first.xml")

    db = _import(tmp_path, _export(SURF, SURF_READING), "later.xml")

    assert (_rows(db, "workouts"), _rows(db, "samples")) == (1, 1)


def test_a_workout_whose_details_changed_replaces_its_old_copy(tmp_path):
    """A changed end time gives the workout a new id. The old row would otherwise stay
    beside the new one, and a session from one source would count twice."""
    _import(tmp_path, _export(RUN), "first.xml")
    later = tmp_path / "later.xml"
    later.write_text(_export(_workout("Running", "06:00:00", "06:40:00")), encoding="utf-8")

    assert import_export(later, tmp_path / "health.sqlite") == (1, 1)  # new id stored
    assert _rows(tmp_path / "health.sqlite", "workouts") == 1  # old copy gone


def test_a_reading_missing_from_a_newer_export_is_removed(tmp_path):
    """A reading inside a workout that is still there, but absent from the newer
    export, leaves the database too: the samples mirror the export as well."""
    second_reading = _reading("06:20:00", "150")
    _import(tmp_path, _export(RUN, RUN_READING, second_reading), "first.xml")

    db = _import(tmp_path, _export(RUN, RUN_READING), "later.xml")

    assert _rows(db, "samples") == 1


def test_importing_the_same_export_again_removes_nothing(tmp_path):
    text = _export(RUN, SURF, RUN_READING, SURF_READING)
    _import(tmp_path, text, "first.xml")

    db = _import(tmp_path, text, "again.xml")

    assert (_rows(db, "workouts"), _rows(db, "samples")) == (2, 2)


def test_a_strict_abort_removes_no_readings(tmp_path):
    """Readings are removed only after a complete pass. A strict-mode abort part way
    through leaves the stored readings as they were."""
    db = _import(tmp_path, _export(RUN, SURF, RUN_READING, SURF_READING), "first.xml")
    broken = tmp_path / "broken.xml"
    broken.write_text(
        _export(
            RUN,
            SURF,
            RUN_READING,
            _reading("09:10:00", "120").replace("2026-03-02 09:10:00", "NOT-A-DATE", 1),
        ),
        encoding="utf-8",
    )

    with pytest.raises(MalformedRecordError):
        import_samples(broken, db, strict=True)

    assert _rows(db, "samples") == 2


def test_the_command_line_says_how_many_rows_it_removed(tmp_path, capsys):
    _import(tmp_path, _export(RUN, SURF, RUN_READING, SURF_READING), "first.xml")
    later = tmp_path / "later.xml"
    later.write_text(_export(SURF, SURF_READING), encoding="utf-8")

    assert main([str(later), "--db", str(tmp_path / "health.sqlite")]) == 0

    out = capsys.readouterr().out
    assert "0 new of 1 workouts" in out and "(removed 1 no longer in the export)" in out
    assert "0 new of 1 heart-rate samples (removed 1 no longer in the export)" in out


def test_the_command_line_says_nothing_about_removal_when_nothing_was_removed(tmp_path, capsys):
    text = _export(RUN, RUN_READING)
    _import(tmp_path, text, "first.xml")
    again = tmp_path / "again.xml"
    again.write_text(text, encoding="utf-8")

    assert main([str(again), "--db", str(tmp_path / "health.sqlite")]) == 0

    assert "removed" not in capsys.readouterr().out


class _CommitCountingConnection(sqlite3.Connection):
    commits = 0

    def commit(self):
        self.commits += 1
        super().commit()


def test_inserts_and_removals_are_committed_together(tmp_path):
    """One commit for the whole replacement: an interruption between inserting and
    removing can then never leave the database holding both old and new rows."""
    db = _import(tmp_path, _export(RUN, SURF), "first.xml")
    later_workouts = list(_workouts_in(_export(SURF), tmp_path / "later.xml"))
    conn = sqlite3.connect(db, factory=_CommitCountingConnection)
    try:
        assert replace_workouts(conn, later_workouts) == (0, 1)
        assert conn.commits == 1
    finally:
        conn.close()


def _workouts_in(text: str, path):
    path.write_text(text, encoding="utf-8")
    return iter_workouts(path)
