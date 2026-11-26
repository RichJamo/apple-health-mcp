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
from apple_health_mcp.ingest import ExportRejectedError, import_export, import_samples, main
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


def test_a_strict_abort_commits_nothing(tmp_path):
    """Readings are removed only after a complete pass, and a strict-mode abort part
    way through commits nothing: the stored readings stay as they were (none
    removed), and a new reading read before the bad one is not stored either."""
    db = _import(tmp_path, _export(RUN, SURF, RUN_READING, SURF_READING), "first.xml")
    new_reading = _reading("06:15:00", "145")
    broken = tmp_path / "broken.xml"
    broken.write_text(
        _export(
            RUN,
            SURF,
            RUN_READING,
            new_reading,
            _reading("09:10:00", "120").replace("2026-03-02 09:10:00", "NOT-A-DATE", 1),
        ),
        encoding="utf-8",
    )

    with pytest.raises(MalformedRecordError):
        import_samples(broken, db, strict=True)

    with closing(connect(db)) as conn:
        values = sorted(row[0] for row in conn.execute("SELECT value FROM samples"))
    assert values == ["120", "140"]  # as before: nothing removed, "145" not stored


def test_the_command_line_says_how_many_rows_it_removed(tmp_path, capsys):
    _import(tmp_path, _export(RUN, SURF, RUN_READING, SURF_READING), "first.xml")
    later = tmp_path / "later.xml"
    later.write_text(_export(SURF, SURF_READING), encoding="utf-8")

    assert main([str(later), "--db", str(tmp_path / "health.sqlite")]) == 0

    workouts_line, samples_line = capsys.readouterr().out.splitlines()
    assert workouts_line.startswith("imported 0 new of 1 workouts into ")
    assert workouts_line.endswith(" (removed 1 no longer in the export)")
    assert (
        samples_line == "imported 0 new of 1 heart-rate samples (removed 1 no longer in the export)"
    )


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


# A file with another root element, shaped like the clinical export_cda.xml that sits
# beside export.xml in the unzipped folder. It has no workouts.
_CLINICAL_DOCUMENT = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<ClinicalDocument>\n</ClinicalDocument>\n'
)


@pytest.mark.parametrize(
    "text",
    [_CLINICAL_DOCUMENT, "", "not xml at all\n"],
    ids=["another-root", "empty-file", "not-xml"],
)
def test_a_file_that_is_not_an_export_is_refused_before_anything_is_removed(tmp_path, text):
    """The import removes whatever the export lacks, so a wrong file would empty the
    database. Both import steps refuse it, and nothing changes."""
    db = _import(tmp_path, _export(RUN, SURF, RUN_READING, SURF_READING), "first.xml")
    wrong = tmp_path / "export_cda.xml"
    wrong.write_text(text, encoding="utf-8")

    with pytest.raises(ExportRejectedError, match="is not an Apple Health export.xml"):
        import_export(wrong, db)
    with pytest.raises(ExportRejectedError):
        import_samples(wrong, db)

    assert (_rows(db, "workouts"), _rows(db, "samples")) == (2, 2)


def test_an_export_with_no_workouts_is_refused_while_the_database_holds_some(tmp_path):
    db = _import(tmp_path, _export(RUN, SURF, RUN_READING, SURF_READING), "first.xml")
    empty = tmp_path / "empty.xml"
    empty.write_text(_export(), encoding="utf-8")

    with pytest.raises(ExportRejectedError, match="has no workouts, but the database holds 2"):
        import_export(empty, db)

    assert _rows(db, "workouts") == 2


def test_an_export_with_no_workouts_imports_into_an_empty_database(tmp_path):
    """Nothing would be removed, so there is nothing to protect."""
    empty = tmp_path / "empty.xml"
    empty.write_text(_export(), encoding="utf-8")

    assert import_export(empty, tmp_path / "health.sqlite") == (0, 0)


def test_the_command_line_refuses_a_wrong_file_with_a_message(tmp_path, capsys):
    db = _import(tmp_path, _export(RUN, SURF, RUN_READING, SURF_READING), "first.xml")
    wrong = tmp_path / "export_cda.xml"
    wrong.write_text(_CLINICAL_DOCUMENT, encoding="utf-8")

    assert main([str(wrong), "--db", str(db)]) == 1

    assert "is not an Apple Health export.xml" in capsys.readouterr().err
    assert (_rows(db, "workouts"), _rows(db, "samples")) == (2, 2)


def test_a_failed_import_leaves_no_lock_behind(tmp_path, monkeypatch):
    """import_export closes its connection even when it raises in the middle of a
    write, so a later writer is not left waiting on a lock for as long as the error
    is kept. The write is made to fail after taking the lock."""
    db = _import(tmp_path, _export(RUN, SURF), "first.xml")

    def fail_while_writing(conn, _workouts):
        conn.execute("BEGIN IMMEDIATE")  # takes the write lock
        raise RuntimeError("failed part way through")

    monkeypatch.setattr("apple_health_mcp.ingest.replace_workouts", fail_while_writing)
    later = tmp_path / "later.xml"
    later.write_text(_export(RUN), encoding="utf-8")
    with pytest.raises(RuntimeError) as caught:
        import_export(later, db)
    assert caught.value is not None  # the error, and its frames, are still held here

    with closing(sqlite3.connect(db, timeout=0)) as other:
        other.execute("BEGIN IMMEDIATE")  # raises "database is locked" if still held
        other.rollback()


def test_a_replacement_that_fails_part_way_keeps_nothing_even_in_autocommit_mode(tmp_path):
    """The replacement begins its own transaction, so a failure part way through
    leaves nothing behind, even on a connection opened with isolation_level=None,
    where each statement would otherwise commit by itself."""
    db = _import(tmp_path, _export(RUN), "first.xml")
    (surf,) = _workouts_in(_export(SURF), tmp_path / "later.xml")

    def surf_then_failure():
        yield surf
        raise RuntimeError("failed part way through")

    conn = sqlite3.connect(db, isolation_level=None)
    try:
        with pytest.raises(RuntimeError):
            replace_workouts(conn, surf_then_failure())
    finally:
        conn.close()

    assert _rows(db, "workouts") == 1  # the surf was not kept, the run not removed
