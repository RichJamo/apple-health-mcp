"""Print the weekly training summary from the local database.

Usage::

    uv run python -m apple_health_mcp.report [--db data/health.sqlite] [--weeks 12]

One row per Monday-to-Sunday week, up to the current week, so a recent break in
training shows as empty weeks. "The current week" follows this computer's clock and
time zone. A session whose own local week is later (recorded in a zone ahead of this
computer's, or with a wrong clock) is not dropped: the table then ends with its week.

The output is the person's own training data. It stays in the terminal: don't paste
it into an issue, a commit or a chat with an AI unless you mean to share it.

Read-only: the database is opened with ``connect_read_only``, so nothing here can
change it.

Public API:
- ``activity_label(activity_type)``      -> a readable name for an Apple activity type
- ``render(weeks)``                      -> the weeks as a text table
- ``main(argv=None, *, today=date.today)`` -> the command line; returns the exit code
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections.abc import Callable, Sequence
from contextlib import closing
from datetime import date
from pathlib import Path

from apple_health_mcp.analyze import WeekSummary, pad_to_week_of, weekly_summary_from_db
from apple_health_mcp.db import SchemaVersionError, connect_read_only
from apple_health_mcp.parse import MalformedRecordError

ACTIVITY_PREFIX = "HKWorkoutActivityType"
DEFAULT_WEEKS = 12
_COLUMNS = ("Week of", "Sessions", "Minutes", "Active kcal", "Types")
_NO_FIGURE = "-"


def activity_label(activity_type: str) -> str:
    """A readable name for an Apple activity type.

    "HKWorkoutActivityTypeSurfingSports" becomes "Surfing Sports": the prefix is
    dropped and the words are split where a capital follows a lower-case letter.
    The words are Apple's own; a value without the prefix is split the same way,
    and a bare prefix is returned as given.
    """
    name = activity_type.removeprefix(ACTIVITY_PREFIX)
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name) or activity_type


def render(weeks: Sequence[WeekSummary]) -> str:
    """The weeks as a fixed-width text table: a header row, then one row per week in
    the order given.

    - Minutes and active energy are rounded to whole numbers. Energy shows "-" when
      no session in the week has a figure, which is not the same as 0.
    - Types reads "Surfing Sports 2, Running 1": most sessions first, then by name.
      It is empty for a week with no sessions.
    - No line ends in spaces.
    """
    rows = [_COLUMNS] + [_row(week) for week in weeks]
    widths = [max(len(row[i]) for row in rows) for i in range(len(_COLUMNS) - 1)]
    lines = []
    for row in rows:
        week_of, *numbers, types = row
        cells = [week_of.ljust(widths[0])]
        cells += [n.rjust(w) for n, w in zip(numbers, widths[1:], strict=True)]
        cells.append(types)
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def _row(week: WeekSummary) -> tuple[str, str, str, str, str]:
    energy = week.active_energy_kcal
    types = sorted(
        week.sessions_by_type.items(), key=lambda item: (-item[1], activity_label(item[0]))
    )
    return (
        week.week_start.isoformat(),
        str(week.sessions),
        f"{week.minutes:.0f}",
        _NO_FIGURE if energy is None else f"{energy:.0f}",
        ", ".join(f"{activity_label(kind)} {count}" for kind, count in types),
    )


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0  # refused below, with the same message
    if value < 1:
        raise argparse.ArgumentTypeError("must be a whole number, 1 or more")
    return value


def _database_error_message(db: Path, sqlite_errorname: str) -> str:
    """What to tell the person when SQLite refuses to read ``db``, by SQLite's error
    name: only some of these errors mean the file is not a database."""
    if sqlite_errorname.startswith("SQLITE_BUSY"):
        return (
            f"aborted: {db} is busy, probably because an import is writing to it. "
            "Run the report again when the import has finished."
        )
    if sqlite_errorname == "SQLITE_READONLY_ROLLBACK":
        return (
            f"aborted: {db} was left part-written by an interrupted import. "
            "Run the import again; it repairs the file and brings it up to date."
        )
    return f"aborted: {db} is not a database this tool can read"


def main(argv: list[str] | None = None, *, today: Callable[[], date] = date.today) -> int:
    """Print the last ``--weeks`` weeks, up to the week that contains ``today()``
    (or a later session's week; see the module docstring). ``today`` is there so
    tests can fix the date.

    Returns 0 after printing, including "no workouts" for an empty database, and 1
    when the database is missing, built by another version, busy, part-written, not
    a database, or holds a value it cannot read. The messages it prints name no raw
    health value. One case still escapes as a traceback that quotes a value: a
    stored timestamp that does not parse (issue #3).
    """
    parser = argparse.ArgumentParser(
        prog="apple_health_mcp.report",
        description="Print training per Monday-to-Sunday week from the local database.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/health.sqlite"),
        help="path to the SQLite database (default: data/health.sqlite)",
    )
    parser.add_argument(
        "--weeks",
        type=_positive_int,
        default=DEFAULT_WEEKS,
        help=f"how many weeks to show, up to this week (default: {DEFAULT_WEEKS})",
    )
    args = parser.parse_args(argv)

    try:
        with closing(connect_read_only(args.db)) as conn:
            weeks = weekly_summary_from_db(conn)
    except FileNotFoundError:
        print(
            f"aborted: no database at {args.db}. Import an export first: "
            "uv run python -m apple_health_mcp.ingest <export.xml>",
            file=sys.stderr,
        )
        return 1
    except SchemaVersionError as err:
        print(f"aborted: {args.db}: {err}", file=sys.stderr)
        return 1
    except MalformedRecordError as err:
        # The message names only a record type or a stored row's id.
        print(f"aborted: {err}", file=sys.stderr)
        return 1
    except sqlite3.DatabaseError as err:
        # A DatabaseError raised by Python itself, not SQLite, has no error name.
        error_name = getattr(err, "sqlite_errorname", "")
        print(_database_error_message(args.db, error_name), file=sys.stderr)
        return 1

    if not weeks:
        print(f"no workouts in {args.db}")
        return 0
    print(render(pad_to_week_of(weeks, today())[-args.weeks :]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
