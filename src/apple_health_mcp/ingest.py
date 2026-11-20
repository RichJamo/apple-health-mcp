"""Command-line import: read an Apple Health export into the local SQLite database.

This is the glue that joins the pieces already built — the streaming parser and the
storage layer — behind one command:

    uv run python -m apple_health_mcp.ingest exports/export.xml --db data/health.sqlite

This first version imports workouts only (the tracer bullet: prove the whole path end
to end). Samples (heart rate, sleep) and the heart-rate-recovery tail come next, and
with them the skip-and-report error handling and a ``--strict`` flag.

Split: ``main`` and the CLI are scaffolded; ``import_export`` (the actual flow) is
yours to write — see its contract below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from apple_health_mcp.db import connect, init_schema, upsert_workouts
from apple_health_mcp.parse import iter_workouts

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

def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the import, and print a one-line summary."""
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
    args = parser.parse_args(argv)

    # The database lives in a gitignored data/ dir; create it on first run.
    args.db.parent.mkdir(parents=True, exist_ok=True)

    inserted, seen = import_export(args.export, args.db)
    print(f"imported {inserted} new of {seen} workouts into {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
