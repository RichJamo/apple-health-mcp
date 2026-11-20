# CLAUDE.md — apple-health-mcp

## What this is
Local-first, privacy-conscious Apple Health integration for AI agents via MCP.
Python + SQLite now; a Swift/HealthKit iOS collector later. Also a learning
project — teach as we build, Rich writes parts; see the learning-platform memory.

## Priorities
Security > privacy > correctness > clarity > efficiency. Privacy is first-class
here because this repo handles sensitive health data. Minimal diffs, minimal
dependencies; preserve behaviour unless asked.

## Privacy rules (non-negotiable)
- Raw Apple Health exports never enter git (already in .gitignore) and never go
  to an AI provider.
- Only selected, derived fields reach the database; only query results reach an
  MCP client / AI. Design queries to minimise disclosure.
- Never import or log the `Me` block (DOB, sex, blood type) or anything not
  needed for the task at hand.
- Tests use synthetic fixtures only — never real health data.
- export_cda.xml (clinical) and workout-routes (GPS) stay unextracted unless a
  task explicitly needs them.

## How rules are enforced (ported from audit-toolkit)
A rule written only in prose is a suggestion. A rule that must not bend and would
fail silently belongs outside the model — a test, a CI check, or a hook.
- The test suite is the spec. CI (`.github/workflows/ci.yml`) runs it with ruff on
  every PR, and branch protection on main requires it to pass, so a red suite blocks merge.
- Don't weaken or delete a test to make code pass. If a test must change, the
  commit says why. If this recurs, it becomes a guard, not another line here.
- Don't claim a check passed unless it ran; if a tool is missing, fall back and say so.
- A green unit test on a tiny fixture proves correctness, not cost. Validate
  memory/time on the real export before trusting a streaming path.

## Architecture (detail in docs/architecture.md)
- Phase 1 (now): export.xml → streaming parser → SQLite → analysis/report.
- Phase 2: SQLite → read-only MCP server → Claude Code.
- Later: HealthKit iOS collector.
- Data model: `workouts` + `samples` (raw-ish), `workout_feedback` (decoupled,
  our schema), `ingestions` (provenance). Import is idempotent.
- Data facts (HealthKit Export v14): active energy comes from a `WorkoutStatistics`
  child (type=ActiveEnergyBurned, sum), not a Workout attribute; per-workout HR is
  computed from HeartRate records in the workout's time window when no
  WorkoutStatistics HR exists; timestamps carry UTC offsets (store UTC + original tz).

## Conventions
- uv-managed (pyproject.toml, uv.lock committed). Runtime deps minimal
  (stdlib xml/sqlite3; MCP SDK in Phase 2); dev: pytest. src/ layout.
- Run things via uv: `uv run pytest`, `uv run python -m ...`.
- Conventional Commits: type(scope): subject. Small, reviewable commits.
- MCP server is read-only; no tool that modifies or deletes source data.
- The global ~/.claude/CLAUDE.md still applies (Design by Contract, F.I.R.S.T.
  tests, meaningful names, "where a lesson goes").
