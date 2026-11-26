"""Tests for db.connect_read_only, the connection the report and the MCP server use.

Synthetic fixture only (tests/fixtures/sample_export.xml), never real health data.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from apple_health_mcp.db import (
    SCHEMA_VERSION,
    SchemaVersionError,
    connect,
    connect_read_only,
    init_schema,
    upsert_workouts,
)
from apple_health_mcp.parse import iter_workouts

FIXTURE = Path(__file__).parent / "fixtures" / "sample_export.xml"


def _imported_database(tmp_path: Path) -> Path:
    """A database file at this code's schema version, holding the fixture's 3 workouts."""
    db = tmp_path / "health.sqlite"
    with closing(connect(db)) as conn:
        init_schema(conn)
        upsert_workouts(conn, iter_workouts(FIXTURE))
        conn.commit()
    return db


def test_it_reads_but_refuses_every_write(tmp_path):
    """No bug in the report or the MCP server can change the data: SQLite itself
    refuses the write, and the file's bytes are the same afterwards."""
    db = _imported_database(tmp_path)
    before = db.read_bytes()

    with closing(connect_read_only(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 3
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM workouts")

    assert db.read_bytes() == before


def test_it_refuses_a_missing_file_without_creating_it(tmp_path):
    """``connect`` would create an empty database here; this must not."""
    db = tmp_path / "missing.sqlite"

    with pytest.raises(FileNotFoundError):
        connect_read_only(db)

    assert not db.exists()


@pytest.mark.parametrize("version", [0, SCHEMA_VERSION - 1, SCHEMA_VERSION + 1])
def test_it_refuses_another_schema_version(tmp_path, version):
    """Version 0 is what an empty database file reports, so that is refused too."""
    db = _imported_database(tmp_path)
    with closing(connect(db)) as conn:
        conn.execute(f"PRAGMA user_version = {version}")

    with pytest.raises(SchemaVersionError) as caught:
        connect_read_only(db)

    assert (caught.value.found, caught.value.expected) == (version, SCHEMA_VERSION)


def test_it_opens_a_path_with_characters_a_uri_must_escape(tmp_path):
    """The file is opened through a ``file:`` URI, so a space, '?' or '#' in the path
    must be escaped rather than read as the start of URI options."""
    folder = tmp_path / "my data ?#%"
    folder.mkdir()
    db = _imported_database(folder)

    with closing(connect_read_only(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM workouts").fetchone()[0] == 3
