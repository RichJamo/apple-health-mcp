# apple-health-mcp

Ask an AI agent about your own Apple Health data, without that data leaving your machine.

apple-health-mcp imports an Apple Health export into a local SQLite database and,
next, will serve it to AI agents such as Claude Code through a read-only
[Model Context Protocol](https://modelcontextprotocol.io) (MCP) server. The first use
is training analysis: what did I do, how hard was it, and how quickly does my heart
rate recover afterwards?

**Status: Phase 1 of 5.** Import works end to end on a real multi-year export. The
training summary and the MCP server are next — see the [roadmap](#roadmap).

## What works today

- **Streaming import of the whole export.** Apple exports everything as one XML file
  that runs to gigabytes and millions of records. The parser streams it one element
  at a time and frees memory as it goes: a full 1.9 GB export imports in about 40
  seconds at about 32 MB peak memory.
- **Idempotent storage.** Each workout and each reading gets a stable id — a hash of
  what identifies it, taken in UTC — so importing the same export again never
  duplicates a row. The database mirrors which workouts and readings the last export
  holds: a newer export also removes what it no longer has, such as a workout you
  deleted in Apple Health. A row already stored keeps the figures it was first
  imported with. Import your latest full export; a file that is not an export, or
  has no workouts, is refused rather than emptying the database.
- **Only what the analysis needs.** Workouts, plus the heart-rate readings that fall
  inside a workout or the five minutes after it (the recovery curve). The millions of
  other readings, clinical records, GPS routes and personal characteristics such as
  date of birth are never imported.
- **Malformed records** are skipped and counted, or stop the import with `--strict`.
  Error messages name only the kind of record, never its values.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or later.

1. On your iPhone, open Health, tap your profile picture, then **Export All Health
   Data**. Move `export.zip` to your Mac (AirDrop keeps it off the internet).
2. Extract only `export.xml` into `exports/`, which is gitignored:

   ```sh
   unzip export.zip "apple_health_export/export.xml" -d exports/
   ```

3. Import it:

   ```sh
   uv sync
   uv run python -m apple_health_mcp.ingest exports/apple_health_export/export.xml
   ```

   ```text
   imported 812 new of 812 workouts into data/health.sqlite
   imported 70431 new of 70431 heart-rate samples
   ```

   Run it again and both lines report 0 new. Import a newer export and each line
   also says how many rows it removed, if any. (Counts are illustrative.)

## Privacy model

- The export and the database stay on your machine; both are gitignored, and nothing
  is uploaded anywhere.
- Only the selected fields above reach the database.
- The MCP server will be read-only, and an agent will only ever see the answers to the
  queries it is allowed to run — never the export or a whole table.
- Tests use synthetic data only.

## How it works

```text
export.xml ──stream──▶ parse.py ──▶ ingest.py ──▶ SQLite ──▶ MCP server (next) ──▶ Claude Code
                         records     windows +      data/
                         & workouts  idempotent     health.sqlite
                                     upsert
```

The design, the data model and the decisions behind them (including what real exports
taught us) are in [docs/architecture.md](docs/architecture.md).

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Export → streaming parser → SQLite → training summary | import done; summary next |
| 2 | Read-only MCP server for Claude Code | next |
| 3 | Native HealthKit iOS collector, so no manual export | later |
| 4 | End to end: Watch → iPhone → Mac → agent | later |
| 5 | Packaged open-source release | later |

## Development

```sh
uv run pytest           # tests, on synthetic fixtures only
uv run ruff check .     # lint
uv run ruff format .    # format
```

CI runs the format check, the linter and the tests on Python 3.11 and 3.12 for every
pull request.

## How this is built

This is a learning project: I'm using it to learn MCP and Python back-end work, and I
build it with [Claude Code](https://claude.com/claude-code) as a pair programmer. I
write core pieces myself — the record parser, the idempotent workout storage and the
import flow — and Claude writes others. Every change goes through tests, an
independent review pass and CI, and commits made with Claude carry a `Co-Authored-By`
line.

## License

[MIT](LICENSE).

Apple Health and HealthKit are trademarks of Apple Inc. This project is not affiliated
with or endorsed by Apple.
