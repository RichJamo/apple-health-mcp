"""Tests for the weekly report (report.py): labels, the table, and the command line.

Synthetic data only, never real health data. The date is always passed in, so no
test depends on the day it runs.
"""

from contextlib import closing
from datetime import date

import pytest

from apple_health_mcp.analyze import WeekSummary
from apple_health_mcp.db import connect
from apple_health_mcp.ingest import import_export, import_samples
from apple_health_mcp.report import activity_label, main, render

RUN = "HKWorkoutActivityTypeRunning"
SURF = "HKWorkoutActivityTypeSurfingSports"
YOGA = "HKWorkoutActivityTypeYoga"

# Wednesday 25 March 2026, in the week of Monday 23 March.
TODAY = date(2026, 3, 25)


# --- activity_label --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("activity_type", "label"),
    [
        (RUN, "Running"),
        (SURF, "Surfing Sports"),
        ("HKWorkoutActivityTypeHighIntensityIntervalTraining", "High Intensity Interval Training"),
        ("SomethingNew", "Something New"),  # no prefix: split the same way
        ("HKWorkoutActivityType", "HKWorkoutActivityType"),  # a bare prefix, as given
    ],
)
def test_activity_label_drops_apples_prefix_and_splits_the_words(activity_type, label):
    assert activity_label(activity_type) == label


# --- render ----------------------------------------------------------------------------


def _week(day: int, sessions: int, minutes: float, kcal: float | None, types: dict) -> WeekSummary:
    return WeekSummary(date(2026, 3, day), sessions, minutes, kcal, types)


def test_render_lays_out_one_row_per_week():
    """Numbers are right-aligned and rounded; a week with no energy figure shows "-",
    and an empty week has no types and no trailing spaces."""
    table = render(
        [
            _week(2, 2, 90.4, 300.4, {RUN: 1, SURF: 1}),
            _week(9, 0, 0.0, None, {}),
        ]
    )

    assert table == (
        "Week of     Sessions  Minutes  Active kcal  Types\n"
        "2026-03-02         2       90          300  Running 1, Surfing Sports 1\n"
        "2026-03-09         0        0            -"
    )


def test_render_lists_types_most_sessions_first_then_by_name():
    (_, row) = render([_week(2, 4, 200.0, 0.0, {YOGA: 1, SURF: 2, RUN: 1})]).splitlines()

    assert row.endswith("  Surfing Sports 2, Running 1, Yoga 1")


def test_render_shows_a_measured_zero_energy_as_0_not_as_missing():
    (_, row) = render([_week(2, 1, 30.0, 0.0, {YOGA: 1})]).splitlines()

    assert row.split()[3] == "0"


# --- main ------------------------------------------------------------------------------

# A run and a surf in the week of 2 March (the surf has no energy figure), nothing in
# the week of 9 March, and a surf in the week of 16 March.
_EXPORT = """<?xml version="1.0" encoding="UTF-8"?>
<!-- SYNTHETIC: three sessions over three weeks. -->
<HealthData>
 <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="30" durationUnit="min"
          sourceName="Test Watch" startDate="2026-03-02 06:00:00 +0200" endDate="2026-03-02 06:30:00 +0200">
  <MetadataEntry key="HKTimeZone" value="Africa/Johannesburg"/>
  <WorkoutStatistics type="HKQuantityTypeIdentifierActiveEnergyBurned" sum="300.4" unit="kcal"/>
 </Workout>
 <Workout workoutActivityType="HKWorkoutActivityTypeSurfingSports" duration="60.4" durationUnit="min"
          sourceName="Test Watch" startDate="2026-03-05 07:00:00 +0200" endDate="2026-03-05 08:01:00 +0200">
  <MetadataEntry key="HKTimeZone" value="Africa/Johannesburg"/>
 </Workout>
 <Workout workoutActivityType="HKWorkoutActivityTypeSurfingSports" duration="45" durationUnit="min"
          sourceName="Test Watch" startDate="2026-03-16 06:00:00 +0200" endDate="2026-03-16 06:45:00 +0200">
  <MetadataEntry key="HKTimeZone" value="Africa/Johannesburg"/>
  <WorkoutStatistics type="HKQuantityTypeIdentifierActiveEnergyBurned" sum="400.6" unit="kcal"/>
 </Workout>
</HealthData>
"""

_EMPTY_EXPORT = '<?xml version="1.0" encoding="UTF-8"?>\n<HealthData>\n</HealthData>\n'


def _database(tmp_path, export_text: str = _EXPORT):
    export = tmp_path / "export.xml"
    export.write_text(export_text, encoding="utf-8")
    db = tmp_path / "health.sqlite"
    import_export(export, db)
    import_samples(export, db)
    return db


def _report(db, *extra: str) -> int:
    return main(["--db", str(db), *extra], today=lambda: TODAY)


def test_main_prints_every_week_through_the_current_one(tmp_path, capsys):
    """The empty week between sessions and the empty current week both show, so a
    break in training is visible."""
    assert _report(_database(tmp_path)) == 0

    assert capsys.readouterr().out == (
        "Week of     Sessions  Minutes  Active kcal  Types\n"
        "2026-03-02         2       90          300  Running 1, Surfing Sports 1\n"
        "2026-03-09         0        0            -\n"
        "2026-03-16         1       45          401  Surfing Sports 1\n"
        "2026-03-23         0        0            -\n"
    )


def test_main_weeks_option_keeps_the_most_recent_weeks(tmp_path, capsys):
    assert _report(_database(tmp_path), "--weeks", "2") == 0

    rows = capsys.readouterr().out.splitlines()[1:]
    assert [row.split()[0] for row in rows] == ["2026-03-16", "2026-03-23"]


@pytest.mark.parametrize("weeks", ["0", "-3", "two"])
def test_main_refuses_a_weeks_value_below_one(tmp_path, weeks):
    with pytest.raises(SystemExit) as caught:
        _report(_database(tmp_path), "--weeks", weeks)

    assert caught.value.code == 2  # argparse's usage error


def test_main_does_not_change_the_database(tmp_path):
    db = _database(tmp_path)
    before = db.read_bytes()

    assert _report(db) == 0

    assert db.read_bytes() == before


def test_main_says_so_when_there_are_no_workouts(tmp_path, capsys):
    db = _database(tmp_path, _EMPTY_EXPORT)

    assert _report(db) == 0

    assert capsys.readouterr().out == f"no workouts in {db}\n"


def test_main_refuses_a_missing_database_without_creating_it(tmp_path, capsys):
    db = tmp_path / "missing.sqlite"

    assert _report(db) == 1

    assert "no database at" in capsys.readouterr().err
    assert not db.exists()


def test_main_refuses_a_database_built_at_another_version(tmp_path, capsys):
    db = _database(tmp_path)
    with closing(connect(db)) as conn:
        conn.execute("PRAGMA user_version = 1")

    assert _report(db) == 1

    assert "schema is version 1" in capsys.readouterr().err


def test_main_refuses_a_file_that_is_not_a_database(tmp_path, capsys):
    db = tmp_path / "notes.sqlite"
    db.write_text("not a database\n", encoding="utf-8")

    assert _report(db) == 1

    assert "is not a database this tool can read" in capsys.readouterr().err


def test_main_refuses_an_unreadable_stored_zone_without_quoting_it(tmp_path, capsys):
    """A database written elsewhere can hold a zone this machine cannot load. The
    report stops with a message, not a traceback, and does not quote the value."""
    db = _database(tmp_path)
    with closing(connect(db)) as conn, conn:
        conn.execute("UPDATE workouts SET tz = 'GMT+0200'")

    assert _report(db) == 1

    err = capsys.readouterr().err
    assert "unrecognised time zone" in err
    assert "GMT+0200" not in err
