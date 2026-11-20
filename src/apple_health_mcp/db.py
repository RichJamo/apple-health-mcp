"""SQLite storage for parsed Apple Health data.

This is the storage layer: it takes the :class:`~apple_health_mcp.parse.Workout`
objects the parser yields and writes them into a local SQLite file. Its defining
property is **idempotency** — importing the same export twice must never create a
duplicate row. See ``docs/architecture.md`` §5 for the data model.

The idempotency trick: every workout gets a deterministic ``id`` (a hash of the
fields that identify the session), and the table's PRIMARY KEY plus
``INSERT OR IGNORE`` makes the database itself refuse a row it already holds. No
"does this exist?" lookup, no race.

Public API:
- ``connect(path)``               -> open a connection
- ``init_schema(conn)``           -> create tables if absent
- ``workout_id(workout)``         -> the stable hash id        (Rich's task)
- ``upsert_workouts(conn, ws)``   -> insert new, skip seen; returns # inserted (Rich's task)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC
from os import PathLike

from apple_health_mcp.parse import Workout

# One place to name the columns, so the CREATE TABLE and the INSERT can never
# drift apart. ``id`` is first and is the PRIMARY KEY.
_WORKOUT_COLUMNS = (
    "id",
    "activity_type",
    "start_utc",
    "end_utc",
    "duration_s",
    "active_energy_kcal",
    "source_name",
    "tz",
    "raw_metadata",
)


def connect(path: str | PathLike[str]) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite database at ``path``.

    ``path`` may be ``":memory:"`` for an ephemeral database, which the tests use.
    """
    return sqlite3.connect(path)


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the ``workouts`` table if it does not already exist.

    Idempotent by design: ``IF NOT EXISTS`` means calling this on an existing
    database is a no-op, so ingest can call it on every run without caring whether
    the file is new. Nullable columns (``duration_s``, ``active_energy_kcal``) are
    left nullable on purpose — a Yoga session has no energy statistic, and the
    schema must store that as NULL rather than reject the row.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS workouts (
            id                  TEXT PRIMARY KEY,
            activity_type       TEXT NOT NULL,
            start_utc           TEXT NOT NULL,
            end_utc             TEXT NOT NULL,
            duration_s          REAL,
            active_energy_kcal  REAL,
            source_name         TEXT NOT NULL,
            tz                  TEXT NOT NULL,
            raw_metadata        TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _row_values(workout: Workout) -> tuple:
    """The column values for one workout, in ``_WORKOUT_COLUMNS`` order.

    This handles the mechanical conversions so ``upsert_workouts`` can focus on the
    idempotency logic:
    - timestamps are stored as UTC ISO strings (the original offset is not lost — it
      is preserved separately in ``tz``);
    - ``tz`` prefers the export's ``HKTimeZone`` metadata (e.g. "Africa/Johannesburg")
      and falls back to the numeric offset (e.g. "+0200") when it is absent;
    - ``raw_metadata`` is the metadata dict as sorted JSON, so the text is stable.
    """
    return (
        workout_id(workout),
        workout.activity_type,
        workout.start.astimezone(UTC).isoformat(),
        workout.end.astimezone(UTC).isoformat(),
        workout.duration_s,
        workout.active_energy_kcal,
        workout.source_name,
        workout.metadata.get("HKTimeZone") or workout.start.strftime("%z"),
        json.dumps(workout.metadata, sort_keys=True),
    )


def workout_id(workout: Workout) -> str:
    """Return a stable, deterministic id for ``workout``.

    Contract (this is what makes import idempotent):
    - **Deterministic:** the same workout always yields the same id, across runs and
      across exports — so re-importing is a no-op at the primary key.
    - **Distinguishing:** two genuinely different sessions yield different ids.
    """
    key = f"{workout.activity_type}|{workout.start.isoformat()}|{workout.end.isoformat()}|{workout.source_name}"
    return hashlib.sha256(key.encode()).hexdigest()


def upsert_workouts(conn: sqlite3.Connection, workouts: Iterable[Workout]) -> int:
    """Insert each workout, skipping any already stored; return how many were new.

    Contract:
    - Precondition: ``init_schema(conn)`` has been run.
    - Postcondition: every workout's id is present in the table exactly once, and
      the connection is committed.
    - Returns: the number of rows actually inserted (0 when every workout was
      already present — this is what the idempotency test checks).
    """
    columns = ", ".join(_WORKOUT_COLUMNS)
    placeholders = ", ".join("?" for _ in _WORKOUT_COLUMNS)
    sql = f"INSERT OR IGNORE INTO workouts ({columns}) VALUES ({placeholders})"

    inserted = 0
    for workout in workouts:
        cursor = conn.execute(sql, _row_values(workout))
        inserted += cursor.rowcount

    conn.commit()
    return inserted
