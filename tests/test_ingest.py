"""Tests for the CLI import flow (ingest.py).

Run against the synthetic fixture (tests/fixtures/sample_export.xml) — never real
health data. These are the spec for Rich's import_export: wiring the parser to the
storage layer, end to end, writing to a real SQLite file in a temp directory.

Until import_export is implemented it raises NotImplementedError, so these are
expected to be RED first and go green when the four steps are written.
"""

from pathlib import Path

from apple_health_mcp.db import connect
from apple_health_mcp.ingest import import_export

FIXTURE = Path(__file__).parent / "fixtures" / "sample_export.xml"


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
