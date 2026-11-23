"""Command-line import: read an Apple Health export into the local SQLite database.

This is the glue that joins the pieces already built — the streaming parser and the
storage layer — behind one command:

    uv run python -m apple_health_mcp.ingest exports/export.xml --db data/health.sqlite

It imports workouts, then the workout-adjacent heart-rate samples (option 1: only the
readings inside each workout's window plus a recovery tail, so the database stays
small and private). Malformed records are skipped and counted by default; ``--strict``
aborts on the first one instead.

``import_export`` imports the workouts; ``import_samples`` and the window-matching
import the heart-rate readings that belong to them.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from bisect import bisect_right
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apple_health_mcp.db import (
    OutdatedSchemaError,
    Sample,
    connect,
    init_schema,
    upsert_samples,
    upsert_workouts,
)
from apple_health_mcp.parse import MalformedRecordError, Record, iter_records, iter_workouts

# Heart-rate reading types we keep (option 1: workout-adjacent heart rate only).
HEART_RATE = "HKQuantityTypeIdentifierHeartRate"
HEART_RATE_RECOVERY = "HKQuantityTypeIdentifierHeartRateRecoveryOneMinute"
SAMPLE_TYPES = frozenset({HEART_RATE, HEART_RATE_RECOVERY})

# How far past a workout's end we still keep heart-rate readings, so the recovery
# curve (how fast HR falls after exercise) survives. See docs/architecture.md.
RECOVERY_WINDOW = timedelta(minutes=5)


def import_export(export_path: str | Path, db_path: str | Path) -> tuple[int, int]:
    """Import the workouts from ``export_path`` into the SQLite database at ``db_path``.

    Returns ``(inserted, seen)``:
    - ``seen``     — how many workouts were read from the export.
    - ``inserted`` — how many were newly stored. Because import is idempotent,
      re-running on the same export returns ``(0, seen)`` — the whole point.

    Contract:
    - Precondition: ``export_path`` is a readable Apple Health ``export.xml``.
    - Postcondition: every workout in the export is present in the database exactly
      once, and the connection is committed (``upsert_workouts`` commits for you).
    """
    conn = connect(db_path)
    init_schema(conn)
    workouts = list(iter_workouts(export_path))
    inserted = upsert_workouts(conn, workouts)
    return inserted, len(workouts)


def _windows_from_db(conn: sqlite3.Connection) -> list[tuple[datetime, datetime, str]]:
    """Each workout's keep-window ``[start, end + RECOVERY_WINDOW]``, sorted by start.

    Read from the database — the workouts are already imported — so the multi-GB
    export is not scanned a second time just to rebuild them. Each entry is
    ``(window_start, window_end, workout_id)``.
    """
    rows = conn.execute("SELECT id, start_utc, end_utc FROM workouts").fetchall()
    windows = [
        (
            datetime.fromisoformat(start),
            datetime.fromisoformat(end) + RECOVERY_WINDOW,
            workout_id,
        )
        for workout_id, start, end in rows
    ]
    windows.sort()
    return windows


def _samples_in_windows(
    windows: list[tuple[datetime, datetime, str]],
    records: Iterable[Record],
) -> Iterator[Sample]:
    """Yield a :class:`Sample` for each record that falls in some workout's window.

    Contract: a reading is kept if and only if at least one window contains it
    (``start <= reading <= end + RECOVERY_WINDOW``, both ends inclusive), and it is
    tagged with one of those workouts. Readings outside every window are dropped — we
    store only workout-adjacent heart rate (option 1), keeping the database small and
    private. ``windows`` must be sorted by start, as ``_windows_from_db`` returns them.

    Windows do overlap in real exports, mostly because one session is recorded twice
    by different sources (a third-party app and the Watch), with start and end a few
    seconds apart, so one record sits just inside the other. So the latest-starting
    window is tried first, and if it does not reach the reading, the window that
    reaches furthest among those started by then. Both
    lookups are a binary search plus a precomputed list, so millions of readings stay
    fast.

    Because overlapping workouts can share a reading, ``workout_id`` names *a* workout
    the reading falls in, not necessarily the only one, and ``INSERT OR IGNORE`` never
    re-tags a stored reading. Recovery analysis should therefore select readings by
    time range, not by ``workout_id``.
    """
    starts = [window[0] for window in windows]
    # furthest[i] = (end, workout_id) of the window, among windows[0..i], whose end is
    # latest: the fallback for a reading that the latest-starting window misses.
    furthest: list[tuple[datetime, str]] = []
    for _start, end, workout_id in windows:
        if not furthest or end > furthest[-1][0]:
            furthest.append((end, workout_id))
        else:
            furthest.append(furthest[-1])

    for record in records:
        moment = record.start.astimezone(UTC)
        index = bisect_right(starts, moment) - 1
        if index < 0:
            continue  # before the first workout
        _window_start, window_end, workout_id = windows[index]
        if moment > window_end:
            window_end, workout_id = furthest[index]
        if moment <= window_end:
            yield Sample(
                type=record.type,
                start=record.start,
                value=record.value,
                unit=record.unit,
                source_name=record.source_name,
                workout_id=workout_id,
                end=record.end,
            )


def import_samples(
    export_path: str | Path,
    db_path: str | Path,
    *,
    strict: bool = False,
) -> tuple[int, int, int]:
    """Import workout-adjacent heart-rate samples. Returns ``(inserted, kept, skipped)``.

    - ``kept``     — readings that fell in a workout window and were selected.
    - ``inserted`` — how many of those were newly stored (0 on a repeat import).
    - ``skipped``  — malformed records skipped. Always 0 when ``strict`` is True,
      which instead aborts on the first bad record.

    Precondition: the workouts are already imported (``import_export`` has run), since
    the windows are read from the workouts table. The whole thing streams — records
    are never all held in memory, so a 4.6M-row export stays bounded.
    """
    conn = connect(db_path)
    init_schema(conn)  # safe to repeat; ensures the samples table exists
    windows = _windows_from_db(conn)

    skipped = 0

    def count_skip(_record_type: str) -> None:
        nonlocal skipped
        skipped += 1

    on_skip = None if strict else count_skip
    records = iter_records(export_path, types=SAMPLE_TYPES, on_skip=on_skip)

    kept = 0

    def selected() -> Iterator[Sample]:
        nonlocal kept
        for sample in _samples_in_windows(windows, records):
            kept += 1
            yield sample

    try:
        inserted = upsert_samples(conn, selected())
    finally:
        conn.close()  # also on a strict-mode abort
    return inserted, kept, skipped


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the import, and print one summary line per table.

    Returns 0 on success, or 1 when ``--strict`` aborts on a malformed record (the
    workouts imported before the abort stay; re-running is safe because import is
    idempotent).
    """
    parser = argparse.ArgumentParser(
        prog="apple_health_mcp.ingest",
        description="Import an Apple Health export.xml into a local SQLite database.",
    )
    parser.add_argument("export", type=Path, help="path to the export.xml file")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/health.sqlite"),
        help="path to the SQLite database (created if missing; default: data/health.sqlite)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="abort on the first malformed record instead of skipping and counting it",
    )
    args = parser.parse_args(argv)

    # The database lives in a gitignored data/ dir; create it on first run.
    args.db.parent.mkdir(parents=True, exist_ok=True)

    try:
        w_inserted, w_seen = import_export(args.export, args.db)
    except OutdatedSchemaError as err:
        print(
            f"aborted: {args.db} was built by an older version of this tool "
            f"(schema {err.found}, this version needs {err.expected}). "
            "Delete it and run the import again; it is rebuilt from the export.",
            file=sys.stderr,
        )
        return 1
    print(f"imported {w_inserted} new of {w_seen} workouts into {args.db}")

    try:
        s_inserted, s_kept, s_skipped = import_samples(args.export, args.db, strict=args.strict)
    except MalformedRecordError as err:
        # The message names only the record type, never its raw value or timestamp.
        print(f"aborted: {err} (run without --strict to skip and count it)", file=sys.stderr)
        return 1
    skipped_note = f" (skipped {s_skipped} malformed)" if s_skipped else ""
    print(f"imported {s_inserted} new of {s_kept} heart-rate samples{skipped_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
