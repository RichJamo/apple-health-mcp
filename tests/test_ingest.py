"""Tests for the CLI import flow (ingest.py), end to end against a real SQLite file.

Synthetic data only, never real health data. Two inputs:
- tests/fixtures/sample_export.xml — the shared fixture: three workouts whose heart
  rate all falls inside [start, end].
- _write_window_export below — one workout with readings placed around its window,
  so the recovery tail, the type filter and skip-and-report are each observable
  through the production code that implements them.
"""

from pathlib import Path

import pytest

from apple_health_mcp.db import connect
from apple_health_mcp.ingest import (
    HEART_RATE,
    HEART_RATE_RECOVERY,
    import_export,
    import_samples,
    main,
)
from apple_health_mcp.parse import MalformedRecordError

FIXTURE = Path(__file__).parent / "fixtures" / "sample_export.xml"

# One 30-minute run, 10:00-10:30 +0200 (08:00-08:30 UTC), so it keeps 10:00-10:35.
# Each record's comment says what the import must do with it.
_WINDOW_EXPORT = """<?xml version="1.0" encoding="UTF-8"?>
<!-- SYNTHETIC: one workout and readings placed around its window. -->
<HealthData locale="en_ZA">
 <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="30" durationUnit="min"
          sourceName="Test Watch" startDate="2026-03-01 10:00:00 +0200" endDate="2026-03-01 10:30:00 +0200"/>
 <!-- inside the workout: keep -->
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" unit="count/min" value="140"
         startDate="2026-03-01 10:15:00 +0200" endDate="2026-03-01 10:15:00 +0200"/>
 <!-- inside the workout, but resting HR is not a kept type: leave out -->
 <Record type="HKQuantityTypeIdentifierRestingHeartRate" sourceName="Test Watch" unit="count/min" value="55"
         startDate="2026-03-01 10:20:00 +0200" endDate="2026-03-01 10:20:00 +0200"/>
 <!-- Apple's one-minute recovery figure, in the tail: keep -->
 <Record type="HKQuantityTypeIdentifierHeartRateRecoveryOneMinute" sourceName="Test Watch" unit="count/min"
         value="25" startDate="2026-03-01 10:31:00 +0200" endDate="2026-03-01 10:31:00 +0200"/>
 <!-- heart rate in the recovery tail: keep -->
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" unit="count/min" value="120"
         startDate="2026-03-01 10:33:00 +0200" endDate="2026-03-01 10:33:00 +0200"/>
 <!-- past the tail: drop -->
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" unit="count/min" value="95"
         startDate="2026-03-01 10:40:00 +0200" endDate="2026-03-01 10:40:00 +0200"/>
 <!-- malformed timestamp: skipped and counted, or an abort under strict -->
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Test Watch" unit="count/min" value="130"
         startDate="NOT-A-DATE" endDate="2026-03-01 10:16:00 +0200"/>
</HealthData>
"""


def _write_window_export(tmp_path: Path) -> Path:
    path = tmp_path / "export.xml"
    path.write_text(_WINDOW_EXPORT, encoding="utf-8")
    return path


def _write_one_workout(path: Path, start: str, end: str) -> Path:
    """A minimal export holding one 30-minute run with the given start and end text."""
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<HealthData>'
        '<Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="30" '
        f'durationUnit="min" sourceName="Test Watch" startDate="{start}" endDate="{end}"/>'
        "</HealthData>",
        encoding="utf-8",
    )
    return path


# --- workouts (import_export) -------------------------------------------------


def test_import_reports_counts(tmp_path):
    """A fresh import reports all three fixture workouts as new."""
    db = tmp_path / "health.sqlite"

    inserted, seen = import_export(FIXTURE, db)

    assert seen == 3  # read from the export
    assert inserted == 3  # all three are new


def test_import_is_idempotent(tmp_path):
    """Re-importing the same export inserts nothing and leaves the table unchanged.

    This is the end-to-end proof of idempotency: the same property db.py guarantees,
    now exercised through the whole parse -> store path against a real file on disk.
    """
    db = tmp_path / "health.sqlite"

    import_export(FIXTURE, db)  # first import
    inserted, seen = import_export(FIXTURE, db)  # same export again

    assert seen == 3
    assert inserted == 0  # nothing new the second time

    conn = connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 3
    finally:
        conn.close()


def test_same_workout_written_with_a_different_offset_is_stored_once(tmp_path):
    """A later export that writes the same session with a different UTC offset adds
    nothing: 10:00 +0200 and 08:00 +0000 are the same moment, so the same id (#6)."""
    db = tmp_path / "health.sqlite"
    first = _write_one_workout(
        tmp_path / "first.xml", "2026-03-01 10:00:00 +0200", "2026-03-01 10:30:00 +0200"
    )
    later = _write_one_workout(
        tmp_path / "later.xml", "2026-03-01 08:00:00 +0000", "2026-03-01 08:30:00 +0000"
    )

    assert import_export(first, db) == (1, 1)
    assert import_export(later, db) == (0, 1)  # recognised, not stored again


# --- samples on the shared fixture --------------------------------------------


def test_import_samples_keeps_workout_adjacent_hr(tmp_path):
    """The fixture's three HeartRate records fall inside workout windows, so all are
    kept and each is linked to the right workout. (The fixture's resting-HR and sleep
    records fall outside every window, so the type filter is tested separately.)"""
    db = tmp_path / "health.sqlite"
    import_export(FIXTURE, db)  # workouts first — the windows come from them

    inserted, kept, skipped = import_samples(FIXTURE, db)

    assert kept == 3
    assert inserted == 3
    assert skipped == 0

    conn = connect(db)
    try:
        linked = conn.execute(
            "SELECT COUNT(*) FROM samples WHERE workout_id IS NOT NULL"
        ).fetchone()[0]
        assert linked == 3  # every kept reading is tagged with its workout

        hiit = conn.execute(
            "SELECT id FROM workouts WHERE activity_type = ?",
            ("HKWorkoutActivityTypeHighIntensityIntervalTraining",),
        ).fetchone()[0]
        hiit_samples = conn.execute(
            "SELECT COUNT(*) FROM samples WHERE workout_id = ?", (hiit,)
        ).fetchone()[0]
        assert hiit_samples == 2  # the 150 and 170 readings, both in the HIIT window
    finally:
        conn.close()


def test_import_samples_is_idempotent(tmp_path):
    """Re-importing the samples inserts nothing new and selects the same readings."""
    db = tmp_path / "health.sqlite"
    import_export(FIXTURE, db)
    import_samples(FIXTURE, db)  # first samples import

    inserted, kept, skipped = import_samples(FIXTURE, db)  # same export again

    assert inserted == 0  # nothing new the second time
    assert kept == 3  # still selects the same three


# --- samples around one window: tail, type filter, skip-and-report ------------


def test_import_keeps_the_recovery_tail_and_only_kept_types(tmp_path):
    """Through the real import, the window's end includes the 5-minute tail, and only
    heart rate and Apple's recovery figure are kept.

    Stored: the 10:15 reading inside the run, Apple's 10:31 recovery figure and the
    10:33 reading in the tail. Absent: the 10:40 reading past the tail, and the
    resting-HR reading, which sits inside the run but is not a kept type. This fails
    if the tail is not added, if every type is let through, or if the recovery type is
    dropped from the kept set.
    """
    export = _write_window_export(tmp_path)
    db = tmp_path / "health.sqlite"
    import_export(export, db)

    import_samples(export, db)

    conn = connect(db)
    try:
        rows = conn.execute(
            "SELECT type, start_utc, value FROM samples ORDER BY start_utc"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [
        (HEART_RATE, "2026-03-01T08:15:00+00:00", "140"),
        (HEART_RATE_RECOVERY, "2026-03-01T08:31:00+00:00", "25"),
        (HEART_RATE, "2026-03-01T08:33:00+00:00", "120"),
    ]


def test_malformed_record_is_skipped_and_counted_by_default(tmp_path):
    """By default the one malformed record is skipped and counted, and the import goes on."""
    export = _write_window_export(tmp_path)
    db = tmp_path / "health.sqlite"
    import_export(export, db)

    inserted, kept, skipped = import_samples(export, db)

    assert (inserted, kept, skipped) == (3, 3, 1)


def test_strict_import_aborts_on_the_malformed_record(tmp_path):
    """With strict=True the same malformed record aborts the samples import."""
    export = _write_window_export(tmp_path)
    db = tmp_path / "health.sqlite"
    import_export(export, db)

    with pytest.raises(MalformedRecordError):
        import_samples(export, db, strict=True)


# --- the CLI ------------------------------------------------------------------


def test_main_imports_workouts_and_samples(tmp_path, capsys):
    """The CLI runs the whole thing and reports both workouts and samples."""
    db = tmp_path / "health.sqlite"

    rc = main([str(FIXTURE), "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "3 new of 3 workouts" in out
    assert "3 new of 3 heart-rate samples" in out


def test_main_reports_skipped_records(tmp_path, capsys):
    """By default the CLI finishes and says how many malformed records it skipped."""
    export = _write_window_export(tmp_path)

    rc = main([str(export), "--db", str(tmp_path / "health.sqlite")])

    assert rc == 0
    assert "3 new of 3 heart-rate samples (skipped 1 malformed)" in capsys.readouterr().out


def test_main_strict_exits_nonzero_without_exposing_raw_text(tmp_path, capsys):
    """--strict reaches the import: the CLI stops with exit code 1 and a message that
    names the record type but not the malformed timestamp."""
    export = _write_window_export(tmp_path)

    rc = main([str(export), "--db", str(tmp_path / "health.sqlite"), "--strict"])

    assert rc == 1
    err = capsys.readouterr().err
    assert f"malformed {HEART_RATE} record" in err
    assert "NOT-A-DATE" not in err
