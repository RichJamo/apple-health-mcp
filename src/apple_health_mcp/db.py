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
- ``connect(path)``                 -> open a connection
- ``init_schema(conn)``             -> create tables if absent
- ``workout_id(workout)``           -> the stable hash id        (Rich's task)
- ``upsert_workouts(conn, ws)``     -> insert new, skip seen; returns # inserted (Rich's task)
- ``upsert_samples(conn, samples)`` -> same, for time-series :class:`Sample` rows

The samples half stores the heart-rate time-series that the training summary and
heart-rate-recovery features read; each sample may carry the id of the workout it
falls within. The same idempotency trick applies — a stable ``id`` per reading.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
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

# Columns for one time-series sample. ``workout_id`` is a nullable link to the
# workout whose window the reading falls in; a reading outside every window is
# never stored (we keep only workout-adjacent data), so in practice it is set.
_SAMPLE_COLUMNS = (
    "id",
    "type",
    "start_utc",
    "value",
    "unit",
    "source_name",
    "workout_id",
)


def connect(path: str | PathLike[str]) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite database at ``path``.

    ``path`` may be ``":memory:"`` for an ephemeral database, which the tests use.
    """
    return sqlite3.connect(path)


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the ``workouts`` and ``samples`` tables if they do not already exist.

    Idempotent by design: ``IF NOT EXISTS`` means calling this on an existing
    database is a no-op, so ingest can call it on every run without caring whether
    the file is new. Nullable columns (``duration_s``, ``active_energy_kcal``) are
    left nullable on purpose — a Yoga session has no energy statistic, and the
    schema must store that as NULL rather than reject the row.

    ``samples`` holds time-series readings (heart rate), indexed on
    ``(type, start_utc)`` because queries always ask for one kind of reading over a
    time range. ``workout_id`` is nullable and declared ``REFERENCES workouts(id)``,
    which documents the link; SQLite enforces it only when ``PRAGMA foreign_keys`` is
    on, and this module does not turn it on.
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS samples (
            id           TEXT PRIMARY KEY,
            type         TEXT NOT NULL,
            start_utc    TEXT NOT NULL,
            value        TEXT,
            unit         TEXT,
            source_name  TEXT NOT NULL,
            workout_id   TEXT REFERENCES workouts(id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS samples_type_start ON samples (type, start_utc)")
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

    Start and end are taken in UTC, so the same session written with a different
    offset in a later export (``+02:00`` once, ``+00:00`` the next time) keeps its id
    instead of being stored twice (issue #6).
    """
    start = workout.start.astimezone(UTC).isoformat()
    end = workout.end.astimezone(UTC).isoformat()
    key = f"{workout.activity_type}|{start}|{end}|{workout.source_name}"
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


@dataclass(frozen=True)
class Sample:
    """One time-series reading, ready to store.

    Built by the import step from a parsed :class:`~apple_health_mcp.parse.Record`
    plus the id of the workout whose window it falls in.

    Invariants:
    - ``start`` is a timezone-aware datetime (stored as UTC, like a workout's).
    - ``value`` and ``unit`` may be ``None`` (the export marks them optional).
    - ``workout_id`` is the owning workout's id, or ``None`` if unlinked.
    """

    type: str
    start: datetime
    value: str | None
    unit: str | None
    source_name: str
    workout_id: str | None


def sample_id(sample: Sample) -> str:
    """Return a stable, deterministic id for ``sample`` (same idea as ``workout_id``).

    Keyed on what identifies a reading — its type, instant, value and source — so
    re-importing the same export is a no-op at the primary key. The instant is taken
    in UTC, so the same moment written with a different offset (``+02:00`` in one
    export, ``+00:00`` in another) still gets the same id and is stored once. The
    ``workout_id`` link is derived, not part of identity, so it is left out of the key.
    """
    instant = sample.start.astimezone(UTC).isoformat()
    key = f"{sample.type}|{instant}|{sample.value}|{sample.source_name}"
    return hashlib.sha256(key.encode()).hexdigest()


def _sample_row_values(sample: Sample) -> tuple:
    """The column values for one sample, in ``_SAMPLE_COLUMNS`` order.

    Stores ``start`` as a UTC ISO string, matching how workouts store their times.
    """
    return (
        sample_id(sample),
        sample.type,
        sample.start.astimezone(UTC).isoformat(),
        sample.value,
        sample.unit,
        sample.source_name,
        sample.workout_id,
    )


def upsert_samples(conn: sqlite3.Connection, samples: Iterable[Sample]) -> int:
    """Insert each sample, skipping any already stored; return how many were new.

    Same idempotent mechanism as ``upsert_workouts``: a stable primary key plus
    ``INSERT OR IGNORE``. Takes an iterable so the caller can stream millions of
    readings without ever holding them all in memory.
    """
    columns = ", ".join(_SAMPLE_COLUMNS)
    placeholders = ", ".join("?" for _ in _SAMPLE_COLUMNS)
    sql = f"INSERT OR IGNORE INTO samples ({columns}) VALUES ({placeholders})"

    inserted = 0
    for sample in samples:
        cursor = conn.execute(sql, _sample_row_values(sample))
        inserted += cursor.rowcount

    conn.commit()
    return inserted
