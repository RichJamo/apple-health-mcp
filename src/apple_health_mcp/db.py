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
- ``connect_read_only(path)``       -> open an existing database for reading only, at
  this code's schema version (for the report and the MCP server)
- ``init_schema(conn)``             -> create tables if absent; refuses a database
  built at another version with ``SchemaVersionError`` (see ``SCHEMA_VERSION``)
- ``workout_id(workout)``           -> the stable hash id
- ``zone_from_tz(tz)``              -> the zone a stored ``tz`` value names; raises
  one of ``TZ_READ_ERRORS`` when it names none
- ``stored_time(text, table, row)`` -> a stored UTC timestamp read back; refuses one
  it cannot read with ``StoredValueError``, without quoting it
- ``upsert_workouts(conn, ws)``     -> insert new, skip seen; returns # inserted
- ``upsert_samples(conn, samples)`` -> same, for time-series :class:`Sample` rows
- ``replace_workouts(conn, ws)`` / ``replace_samples(conn, samples)``
                                   -> make the table hold exactly these rows;
  returns (# inserted, # removed). The import uses these, so the database mirrors
  the export it last read.

The samples half stores the heart-rate time-series that the training summary and
heart-rate-recovery features read; each sample may carry the id of the workout it
falls within. The same idempotency trick applies — a stable ``id`` per reading.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, tzinfo
from os import PathLike
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apple_health_mcp.parse import MalformedRecordError, Workout

# The schema this code writes, recorded in SQLite's ``user_version``. Bump it whenever
# a table changes shape, or whenever ``workout_id`` or ``sample_id`` computes its key
# differently: new ids would make a re-import store every row a second time, with no
# error. A database built at another version is refused, not patched. The golden-id
# test in tests/test_schema.py fails when a key changes, as the reminder to bump this.
SCHEMA_VERSION = 2  # 2: workouts gained Apple's heart-rate average and maximum


class SchemaVersionError(Exception):
    """The database was built at a different schema version than this code writes.

    Its tables may lack columns this code inserts, which would otherwise fail later
    with an obscure SQL error. Everything in the database is derived from the export,
    so the remedy is to delete the database file and import again.

    That remedy holds only while every table is rebuilt from the export. A table of
    data entered by hand (the planned ``workout_feedback``) would be lost by it, so
    adding one needs a real migration first; a test in tests/test_schema.py fails when
    a table outside the derived set appears, as the reminder.

    The import's mirror needs the same thought (``replace_workouts``): it deletes a
    workout the newer export lacks, and gives an edited one a new id, so notes keyed
    by workout id would come loose from their session. So does every message that
    gives ``REBUILD_ADVICE``: today it costs nothing, and then it would cost the notes.
    """

    def __init__(self, found: int, expected: int) -> None:
        super().__init__(
            f"database schema is version {found}, but this code needs version {expected}; "
            "it is rebuilt from the export, so delete the database file and import again"
        )
        self.found = found
        self.expected = expected


# One place to name the columns, so the CREATE TABLE and the INSERT can never
# drift apart. ``id`` is first and is the PRIMARY KEY.
_WORKOUT_COLUMNS = (
    "id",
    "activity_type",
    "start_utc",
    "end_utc",
    "duration_s",
    "active_energy_kcal",
    "heart_rate_average_bpm",
    "heart_rate_maximum_bpm",
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
    "end_utc",
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


def connect_read_only(path: str | PathLike[str]) -> sqlite3.Connection:
    """Open the existing database at ``path`` for reading only.

    For code that must never change the data, such as the report and the MCP server.

    Postconditions:
    - SQLite refuses every write to the database. ``mode=ro`` covers the main file
      only, so attaching another database is also switched off: otherwise
      ``ATTACH`` could reopen the same file read-write, or create a new file, and
      ``VACUUM INTO`` could copy the whole database to any path. Temporary tables
      still work; they live in memory or a temp file and vanish on close.
    - What a connection can still change is its own view, until it closes: a temp
      table or view named ``workouts`` hides the real one from later queries that
      don't say ``main.workouts``, and ``PRAGMA locking_mode = EXCLUSIVE`` holds a
      lock that makes an import wait. So never reuse one connection for SQL from
      different callers: open one per call, as the report does once.
    - A missing ``path`` is not created. (A database someone switched to WAL mode
      with another tool still gets its ``-wal`` and ``-shm`` side files; this code
      never turns WAL on.)

    Raises:
    - ``FileNotFoundError`` when nothing exists at ``path`` (``connect`` would
      quietly create an empty database there);
    - ``SchemaVersionError`` when the database is at another version than
      ``SCHEMA_VERSION``. An empty file is at version 0, so it is refused too, with
      the same remedy: import again;
    - ``sqlite3.DatabaseError`` when the file is not a SQLite database, or SQLite
      cannot read it (busy, or left mid-write by an interrupted import). The
      connection is closed first in every case.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"no database at {file}")
    conn = sqlite3.connect(f"{file.resolve().as_uri()}?mode=ro", uri=True)
    conn.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
    try:
        found = conn.execute("PRAGMA user_version").fetchone()[0]
    except sqlite3.DatabaseError:
        conn.close()
        raise
    if found != SCHEMA_VERSION:
        conn.close()
        raise SchemaVersionError(found, SCHEMA_VERSION)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the ``workouts`` and ``samples`` tables if they do not already exist.

    Idempotent by design: ``IF NOT EXISTS`` means calling this on an existing
    database is a no-op, so ingest can call it on every run without caring whether
    the file is new. Nullable columns (``duration_s``, ``active_energy_kcal`` and the
    two ``heart_rate_*_bpm`` columns) are left nullable on purpose — a Yoga session
    has no energy statistic, older workouts have no heart-rate statistic, and the
    schema must store that as NULL rather than reject the row.

    ``samples`` holds time-series readings (heart rate), indexed on
    ``(type, start_utc)`` because queries always ask for one kind of reading over a
    time range. ``workout_id`` is nullable and declared ``REFERENCES workouts(id)``,
    which documents the link; SQLite enforces it only when ``PRAGMA foreign_keys`` is
    on, and this module does not turn it on. ``end_utc`` equals ``start_utc`` for an
    instant reading; for a span (Apple stores some heart rate as one value covering
    several minutes) it is when the span ends.

    The database records ``SCHEMA_VERSION``. If the tables already exist at another
    version, this raises :class:`SchemaVersionError` instead of writing to them.
    """
    found = conn.execute("PRAGMA user_version").fetchone()[0]
    has_tables = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master"
        " WHERE type = 'table' AND name IN ('workouts', 'samples')"
    ).fetchone()[0]
    if has_tables and found != SCHEMA_VERSION:
        raise SchemaVersionError(found, SCHEMA_VERSION)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS workouts (
            id                      TEXT PRIMARY KEY,
            activity_type           TEXT NOT NULL,
            start_utc               TEXT NOT NULL,
            end_utc                 TEXT NOT NULL,
            duration_s              REAL,
            active_energy_kcal      REAL,
            heart_rate_average_bpm  REAL,
            heart_rate_maximum_bpm  REAL,
            source_name             TEXT NOT NULL,
            tz                      TEXT NOT NULL,
            raw_metadata            TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS samples (
            id           TEXT PRIMARY KEY,
            type         TEXT NOT NULL,
            start_utc    TEXT NOT NULL,
            end_utc      TEXT NOT NULL,
            value        TEXT,
            unit         TEXT,
            source_name  TEXT NOT NULL,
            workout_id   TEXT REFERENCES workouts(id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS samples_type_start ON samples (type, start_utc)")
    # PRAGMA takes no ? placeholder; SCHEMA_VERSION is our own integer constant.
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def _row_values(workout: Workout) -> tuple:
    """The column values for one workout, in ``_WORKOUT_COLUMNS`` order.

    This handles the mechanical conversions so ``upsert_workouts`` can focus on the
    idempotency logic:
    - timestamps are stored as UTC ISO strings (the original offset is not lost — it
      is preserved separately in ``tz``);
    - ``tz`` is the export's ``HKTimeZone`` (e.g. "Africa/Johannesburg") when this
      machine can load it, else the startDate offset (e.g. "+0200"): see ``_stored_tz``;
    - ``raw_metadata`` is the metadata dict as sorted JSON, so the text is stable.
    """
    return (
        workout_id(workout),
        workout.activity_type,
        workout.start.astimezone(UTC).isoformat(),
        workout.end.astimezone(UTC).isoformat(),
        workout.duration_s,
        workout.active_energy_kcal,
        workout.heart_rate_average_bpm,
        workout.heart_rate_maximum_bpm,
        workout.source_name,
        _stored_tz(workout),
        json.dumps(workout.metadata, sort_keys=True),
    )


# The remedy for a value this code cannot read in a database it did not write.
# Everything stored is derived from the export, so rebuilding loses nothing; that
# stops being true once hand-entered data exists (see ``SchemaVersionError``).
REBUILD_ADVICE = "delete the database and import again to rebuild it"


class StoredValueError(MalformedRecordError):
    """A value in the database could not be read, as opposed to a record in the
    export. Only a database written elsewhere can hold one, and the remedy is
    ``REBUILD_ADVICE``. Like its parent, the message never quotes the value; it
    locates the row by something derived from no health data (see ``stored_time``).
    """


def stored_time(text: str, table: str, row: str | int) -> datetime:
    """A timestamp read back from ``table``, where this module wrote it with
    ``astimezone(UTC).isoformat()``. ``row`` locates the row in the message.

    Postcondition: a timezone-aware UTC datetime, the same moment that was stored.

    Raises ``StoredValueError`` unless ``text`` is exactly what this module writes:
    an ISO timestamp with a ``+00:00`` offset. Only a database written elsewhere can
    hold anything else. Refusing the rest matters twice over:
    - Python's own parse error quotes the text, and an MCP tool's error text reaches
      the AI. So the message names only ``table`` and ``row``, and is raised outside
      the ``except`` block, so nothing quoting is attached as its context.
    - The queries compare stored times as text, which is sound only in this one
      form. A naive time, another offset or a "Z" suffix would sort wrongly, silently.

    For ``row``, pass a workout's id, or a reading's SQLite ``rowid``, never a
    reading's id. A reading's id is an unsalted hash of its type, times, value and
    source, so a few hundred thousand guesses recover the time and heart rate it
    stands for. A workout's id is built the same way, but reversing it needs its
    start, end and type, which the MCP tools return anyway.
    """
    try:
        moment = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        moment = None  # handled below, outside the except block
    if moment is None or moment.utcoffset() != timedelta(0) or moment.isoformat() != text:
        raise StoredValueError(f"unreadable timestamp in {table} row {row}; {REBUILD_ADVICE}")
    return moment


# What ``zone_from_tz`` raises for a value that names no zone. All may quote the value.
# OSError is there because, with the ``tzdata`` package installed (always the case on
# Windows), a name that is a folder there ("America") raises IsADirectoryError.
TZ_READ_ERRORS = (ZoneInfoNotFoundError, ValueError, OSError)


def zone_from_tz(tz: str) -> tzinfo:
    """The zone a stored ``tz`` value names, for placing a workout in local time.

    An offset ("+0200", "-0500") was written with ``strftime("%z")``, so it is read
    back with ``strptime("%z")``. Anything else is a zone name for
    ``zoneinfo.ZoneInfo``; a zone name never starts with a sign.

    Raises one of ``TZ_READ_ERRORS`` when ``tz`` is neither. Their messages quote
    ``tz``, so a caller that shows the error must replace it.
    """
    if tz.startswith(("+", "-")):
        return datetime.strptime(tz, "%z").tzinfo
    return ZoneInfo(tz)


def _stored_tz(workout: Workout) -> str:
    """The ``tz`` value stored for ``workout``.

    It is the export's HKTimeZone when ``zone_from_tz`` can load it on this machine,
    and otherwise the offset in startDate, as when the export gives no HKTimeZone.
    Any app can write HKTimeZone (one might write "GMT+0200"), and a name this
    machine cannot load would stop every later summary; stored as the offset, the
    workout is still counted, placed in the exporting phone's zone. The name itself
    is kept in ``raw_metadata``.

    Postcondition: ``zone_from_tz`` accepts the result on this machine.
    """
    name = workout.metadata.get("HKTimeZone")
    if name and _is_readable_tz(name):
        return name
    return workout.start.strftime("%z")


def _is_readable_tz(tz: str) -> bool:
    try:
        zone_from_tz(tz)
    except TZ_READ_ERRORS:
        return False
    return True


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
    sql = _insert_or_ignore("workouts", _WORKOUT_COLUMNS)

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
    - ``start`` and ``end`` are timezone-aware datetimes (stored as UTC, like a
      workout's). ``end`` is expected to be at or after ``start``, as Apple writes
      them; this is not checked.
    - ``end`` is never ``None`` once built: left out, it defaults to ``start``, which
      is what an instant reading is. A span — one value Apple stores for several
      minutes — has a later ``end``.
    - ``value`` and ``unit`` may be ``None`` (the export marks them optional).
    - ``workout_id`` is the owning workout's id, or ``None`` if unlinked.
    """

    type: str
    start: datetime
    value: str | None
    unit: str | None
    source_name: str
    workout_id: str | None
    end: datetime | None = field(default=None)

    def __post_init__(self) -> None:
        if self.end is None:
            object.__setattr__(self, "end", self.start)  # frozen: set once, here


def sample_id(sample: Sample) -> str:
    """Return a stable, deterministic id for ``sample`` (same idea as ``workout_id``).

    Keyed on what identifies a reading — its type, start and end, value and source —
    so re-importing the same export is a no-op at the primary key. Times are taken in
    UTC, so the same moment written with a different offset (``+02:00`` in one export,
    ``+00:00`` in another) still gets the same id and is stored once. The end is part
    of the key so that a span and an instant reading that start together stay two
    readings. The ``workout_id`` link is derived, not part of identity.
    """
    start = sample.start.astimezone(UTC).isoformat()
    end = sample.end.astimezone(UTC).isoformat()
    key = f"{sample.type}|{start}|{end}|{sample.value}|{sample.source_name}"
    return hashlib.sha256(key.encode()).hexdigest()


def _sample_row_values(sample: Sample) -> tuple:
    """The column values for one sample, in ``_SAMPLE_COLUMNS`` order.

    Stores ``start`` and ``end`` as UTC ISO strings, matching how workouts store them.
    """
    return (
        sample_id(sample),
        sample.type,
        sample.start.astimezone(UTC).isoformat(),
        sample.end.astimezone(UTC).isoformat(),
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
    sql = _insert_or_ignore("samples", _SAMPLE_COLUMNS)

    inserted = 0
    for sample in samples:
        cursor = conn.execute(sql, _sample_row_values(sample))
        inserted += cursor.rowcount

    conn.commit()
    return inserted


def replace_workouts(conn: sqlite3.Connection, workouts: Iterable[Workout]) -> tuple[int, int]:
    """Make the workouts table hold exactly ``workouts``. Returns ``(inserted, removed)``.

    New workouts are inserted and ones already stored are kept, as in
    ``upsert_workouts``. Stored workouts that are not among ``workouts`` are then
    deleted: a workout deleted in Apple Health, or the old copy of one whose start,
    end, type or source changed (which gives it a new id).

    The mirror is of which workouts exist, decided by id. A workout already stored
    keeps the values it was first imported with, even if the export now gives
    another figure under the same type, start, end and source.

    Precondition: ``init_schema(conn)`` has been run, and ``workouts`` is a whole
    export's workouts, since anything left out is deleted (the import checks the
    file is an export, and refuses one with no workouts while the database holds
    some; see ``ingest``).

    Postcondition: the table's ids are exactly the ids of ``workouts``, committed in
    one transaction. If iterating ``workouts`` raises, nothing is committed and
    nothing has been deleted; closing the connection rolls back the inserts.
    """
    return _replace_rows(conn, "workouts", _WORKOUT_COLUMNS, map(_row_values, workouts))


def replace_samples(conn: sqlite3.Connection, samples: Iterable[Sample]) -> tuple[int, int]:
    """Make the samples table hold exactly ``samples``. Returns ``(inserted, removed)``.

    The same contract as ``replace_workouts``, for readings: those whose workout is
    gone, and those the newer export no longer has, are deleted. A reading already
    stored keeps the ``workout_id`` it was first linked to, even if that workout
    has since been replaced; select readings by time range, not by ``workout_id``.
    ``samples`` is streamed; the ids seen so far are kept in a temporary SQLite
    table, not in Python, so memory stays bounded.
    """
    return _replace_rows(conn, "samples", _SAMPLE_COLUMNS, map(_sample_row_values, samples))


def _replace_rows(
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    rows: Iterable[tuple],
) -> tuple[int, int]:
    """Insert ``rows`` (column values, id first) with ``INSERT OR IGNORE``, then
    delete every stored row whose id was not among them, and commit once.

    When no transaction is open, this one begins and commits its own with explicit
    ``BEGIN`` and ``COMMIT``, so the replacement is atomic and saved on any
    connection: Python's default, ``isolation_level=None``, and Python 3.12's
    ``autocommit=True`` (where ``conn.commit()`` does nothing). Inside a caller's open
    transaction it joins it and commits that, as ``conn.commit()`` does. ``table``
    and ``columns`` are this module's own constants, never caller input, so building
    the SQL from them is safe.
    """
    insert = _insert_or_ignore(table, columns)
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS seen_ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
    began_here = not conn.in_transaction
    if began_here:
        conn.execute("BEGIN")
    conn.execute("DELETE FROM temp.seen_ids")

    inserted = 0
    for values in rows:
        inserted += conn.execute(insert, values).rowcount
        conn.execute("INSERT OR IGNORE INTO temp.seen_ids (id) VALUES (?)", (values[0],))

    removed = conn.execute(
        f"DELETE FROM {table} WHERE id NOT IN (SELECT id FROM temp.seen_ids)"
    ).rowcount
    conn.execute("DELETE FROM temp.seen_ids")
    if began_here:
        conn.execute("COMMIT")
    else:
        conn.commit()
    return inserted, removed


def _insert_or_ignore(table: str, columns: tuple[str, ...]) -> str:
    """The ``INSERT OR IGNORE`` statement for ``table``'s ``columns``, one ``?`` each."""
    placeholders = ", ".join("?" for _ in columns)
    return f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
