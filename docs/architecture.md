# Architecture — apple-health-mcp

> Living document. Captures the design and the decisions behind it, so we don't
> re-derive them. Last updated as of the first real export inspection (2026-11-19).

## 1. Goal and principles

A local-first, privacy-conscious Apple Health integration for AI agents, starting
with Claude Code over the Model Context Protocol (MCP). The first use case is
personal fitness analysis.

Principles (in priority order for this repo): **security > privacy > correctness >
clarity > efficiency.** Local-first, minimal external dependencies, no mandatory
cloud service, user keeps control of their data, modular and testable components.

## 2. Phases

| Phase | Scope | Status |
|---|---|---|
| 1 | Manual export → streaming parser → SQLite → analysis/report | in progress |
| 2 | SQLite → read-only MCP server → Claude Code | next |
| 3 | Native HealthKit iOS collector (incremental sync) | later |
| 4 | End-to-end: Watch → iOS app → Mac DB → MCP → agent | later |
| 5 | Open-source release (docs, CI, licence, sample data) | later |

Milestone 1: import selected workout data into SQLite and generate a meaningful
training summary. Milestone 2: query that DB through a working MCP server from
Claude Code. The iOS app does not start until both work.

## 3. Repository layout

```
apple-health-mcp/
  pyproject.toml              # uv-managed project; runtime deps minimal
  CLAUDE.md                   # enduring conventions (see also global ~/.claude)
  docs/
    architecture.md           # this file
    privacy.md                # what stays local vs what can reach the AI (Phase 5)
    export-guide.md           # getting the export off the iPhone (Phase 5)
  src/apple_health_mcp/
    parse.py                  # streaming XML -> normalized records
    db.py                     # SQLite schema + idempotent upsert
    ingest.py                 # parse -> db, CLI entry
    analyze.py                # training-summary functions
    report.py                 # renders the summary, CLI
    server.py                 # MCP server (Phase 2)
  tests/
    fixtures/                 # SYNTHETIC export.xml — committed, no real data
    test_parse.py  test_db.py  test_analyze.py
  exports/   data/            # gitignored; real export + health.sqlite live here
```

## 4. The source data (verified from a real export)

Apple Health "Export All Health Data" produces `export.zip` containing
`export.xml`, `export_cda.xml` (clinical) and `workout-routes/*.gpx` (GPS). For
this project we extract **only `export.xml`**; the clinical file and GPS routes are
left untouched inside the zip (sensitive, and not needed for fitness).

Observed in the first real export (HealthKit Export Version 14, ~2.0 GB
`export.xml`, 2019–2026): **1,448 workouts, ~4.6M records.** Because the file is
multi-GB, the parser **must stream** — never load it into memory.

Element shapes that matter (from the embedded DTD, which is the authoritative,
data-free schema):

- **`Workout`** — `workoutActivityType`, `duration`/`durationUnit`, start/end dates
  (with UTC offset), `sourceName`/`device`. May contain child **`WorkoutStatistics`**
  (`type`, `average`, `minimum`, `maximum`, `sum`, `unit`) and `MetadataEntry`.
- **`Record`** — the time-series row: `type`, `value`, `unit`, start/end dates. Heart
  rate is `type="HKQuantityTypeIdentifierHeartRate"`.
- **`ActivitySummary`** — daily rings (active energy, exercise minutes, stand hours).
- **`Me`** — DOB, sex, blood type, etc. **Not imported** (sensitive, unnecessary).

### Decisions forced by the real data

1. **Active energy comes from a child element, not an attribute.** Real workouts
   carry no `totalEnergyBurned` attribute; active energy is in
   `WorkoutStatistics[type=ActiveEnergyBurned].sum`. The parser reads it there, with
   the attribute as fallback only.
2. **Per-workout heart rate: Apple's statistic first; the readings hide the peak.**
   Apple's `WorkoutStatistics` of type `HeartRate` (average, maximum) exists only on
   workouts from late 2022 on, about half the sessions. Inside a workout Apple mostly
   stores heart rate as *spans*: one value covering several minutes, which is the
   average over that stretch: on sessions that have Apple's average, a span-weighted
   mean is a median 0.3 bpm away from it. So average heart rate falls back to that
   weighted mean when the statistic is absent (on older sessions, where it cannot be
   checked against Apple's figure), but peak heart rate comes **only** from the statistic: the
   highest span reads well below Apple's maximum. The parser stores average and
   maximum on the workout (schema version 2).
3. **Timestamps carry their UTC offset** (e.g. `+0200`) plus an `HKTimeZone`
   metadata entry. Store UTC and the original zone; never discard the offset.
   The offset comes from the exporting phone's zone, not the recording zone: on the
   real export, all of the two dozen or so workouts whose `HKTimeZone` had a
   different offset still carried the offset of the phone's zone in their
   `startDate`. So `HKTimeZone` is the only record of where a workout was recorded.
   About a fifth of workouts have none; their `tz` comes from the phone's zone at
   export, and an export made in another zone would shift their local times. (That
   phone's zone has no daylight saving, so the probe cannot say whether a zone that
   has it gives each workout that zone's offset on the workout's date or one offset
   for the whole export.) An `HKTimeZone` the importing machine cannot load is
   stored the same way, as the `startDate` offset, so one bad name from an app
   cannot stop the weekly summary; the name stays in `raw_metadata`.
4. **Apple duplicates some `MetadataEntry` lines.** De-dupe by key.
5. **Strength is logged as HIIT.** Rich records home dumbbell strength as
   `HighIntensityIntervalTraining`, same bucket as cardio HIIT. The workout *type*
   can't separate them; the `workout_feedback` layer (limiter = muscular vs cardio,
   weights noted) is how we'll distinguish them later.
6. **Duration leaves pauses out on Watch workouts.** On the real export, every Watch
   workout whose `duration` is more than a minute short of end minus start has pause
   events, and about three quarters of the Watch workouts with pause events are that
   short (the rest are within a minute of it). No other source recorded a pause
   event, and their durations match end minus start, so whether they would leave
   pauses out is unknown.

### Available for later (noted, not built yet)

- **Longitudinal fitness:** VO2 Max (~390 readings), resting HR (~2,688), HRV SDNN
  (~10,129) — enough for a 7-year "how regularly / how fit" trend.

### Heart-rate recovery (committed — capture now, compute at analysis)

How fast heart rate falls after a workout is a wanted feature, so the pipeline is
designed to preserve it from the start, even though the recovery *number* is computed
later. Two independent sources:

- **Apple's own figure:** 232 `HeartRateRecoveryOneMinute` records (2022+, the
  watchOS 9 feature — verified: none before 2022), each about three minutes after a
  workout ends. Their metadata marks them as predictions from a submaximal test, not
  measurements, so analysis reports them labelled as an estimate, never as ground truth.
- **Measured from samples:** the Watch keeps logging instant heart-rate readings in the
  background after a workout ends. "Heart rate at the end" is usually not measurable,
  because the last minutes inside the workout are a span. So measured recovery is
  Apple's maximum minus the instant reading nearest one minute after the end (within
  15 s either side): the drop from the session's peak, which equals a true one-minute
  recovery only when the peak came at the end. It exists for about half the sessions, those with Apple's maximum.
  If a workout is ended a while after the effort stopped, heart rate was already
  falling, so the figure covers more than a minute of recovery and reads high.

Two design consequences, to honour when we build ingest and analyze:

1. **Ingestion must keep a post-workout tail.** When selecting which `HeartRate` samples
   to import, use `[start, end + RECOVERY_WINDOW]` per workout (RECOVERY_WINDOW ≈ 5 min),
   not just `[start, end]`, or the very samples recovery needs get dropped. Also import
   the `HeartRateRecoveryOneMinute` records.
2. **Recovery is derived, not stored raw.** It is computed in `analyze.py`
   (`heart_rate_summary(workout, readings)`) from samples already in the DB. The four-table model
   needs no new column now; if we later cache the figure, it belongs in a derived
   `workout_metrics` table, never mixed into raw `samples`.

Timing: the *capture* half rides along with `ingest` (the next build step); the *compute*
half lands with the first training summary in `analyze.py`. So it is part of milestone 1,
not deferred to a later phase.

**Capture is implemented** (`ingest.py`). Only heart-rate readings inside some workout's
`[start, end + 5 min]` window are stored (Rich's choice: smaller and more private than
keeping every reading). Apple's `HeartRateRecoveryOneMinute` records go through the same
window, which on the first real export kept every one of them. Workouts overlap in real
exports, mostly because one session is recorded twice by different sources: a surf
tracking app or Strava, and the Watch's own workout, with start and end seconds apart.
So a reading is kept if *any* window holds it; its `workout_id` names one such workout,
and recovery analysis should select readings by time range rather than by `workout_id`.
These double recordings also mean the `workouts` table holds some sessions twice — see
"Source dedup" in §5.

### Malformed records (decided, issue #3)

Ingest skips a record it cannot parse and counts it, so one bad row in years of data
does not stop an import; the summary line reports how many were skipped. `--strict`
aborts on the first one instead and exits with code 1. Either way the output names only
the record type: Python's own parse errors quote the raw timestamp, so `iter_records`
raises `MalformedRecordError` in their place (`from None`), and the skip callback is
given the type alone. Workout parsing does not follow this contract yet: a malformed
workout still raises Python's own error, which can quote raw text (tracked in #3).

## 5. Data model (SQLite)

Four tables keep raw-ish records separate from derived summaries and keep subjective
feedback decoupled from Apple's schema.

- **`workouts`** — `id` (stable hash of source key), `source_uuid`, `type`,
  `start_utc`, `end_utc`, `duration_s`, `tz`, `active_energy_kcal`
  (from WorkoutStatistics), `heart_rate_average_bpm` and `heart_rate_maximum_bpm`
  (Apple's statistic, NULL when absent), `source_name`, `raw_metadata` (JSON).
- **`samples`** — time-series: `type`, `start_utc`, `end_utc` (equal to `start_utc`
  for an instant reading, later for a span), `value`, `unit`, `source_name`, nullable
  `workout_id`. Indexed on `(type, start_utc)`.
- **`workout_feedback`** — our own schema, FK to `workouts.id`:
  `difficulty_1_10`, `limiter` (muscular/cardio), `soreness`, `weights_note`,
  `video_url`, `notes`, `created_utc`.
- **`ingestions`** — provenance/dedup: `file_sha256`, `imported_utc`, counts.

**Idempotent import:** each row's `id` is a stable hash of its source key, inserted
with `ON CONFLICT DO NOTHING`, so re-importing the same or an overlapping export
never duplicates.

**Source dedup:** the same metric can appear from the Watch, the iPhone and
third-party apps. The rule for workouts is decided below, made explicit and
documented rather than chosen silently.

Seen in the first real export: at least 32 pairs of workouts whose start and end both
fall within 3 minutes of each other, and every such pair comes from two different
sources — mostly a surf-tracking app plus the Watch, a few Strava plus the Watch. Each
is one session stored twice, because `workout_id` includes the source. Heart rate is
unaffected (each reading is stored once), but session counts and weekly minutes would
count these twice.

**Decided (Rich):** keep both rows as recorded, and de-duplicate in `analyze.py`
(`distinct_sessions`). Two records from different sources are one session when the
time they share is positive and at least half the shorter recording, whatever
activity type each gave it (Strava can say Running where the Watch says Other).
Records are taken best first: the source that wrote the most heart rate, which is the
Watch, then the longer recording. Each is kept unless it is the same session as a
record already kept. The Watch is recognised by its heart rate, not by name, because
its source name is whatever the owner called the device. So a record is dropped only
for a direct duplicate that ranks above it, never through a chain of overlaps, and no
two kept records are the same session. Two Watch workouts are never merged with each
other, even when one app recording spans both. Import stays a faithful copy of the
export, so nothing the other app recorded is lost.

On the first real export this counts 1,448 stored workouts as 1,404 sessions: 44 app
records are dropped as second recordings, and no Watch record is dropped.

## 6. Privacy model

- Raw exports stay in gitignored `exports/`; never committed, never sent to an AI.
- Only selected fields reach SQLite; only query results reach the MCP client / AI.
- The MCP server is **read-only**; no tool can modify or delete source data.
- Tests use synthetic fixtures only. A fuller treatment lives in `docs/privacy.md`
  for the Phase 5 release.

## 7. Tooling

uv-managed (`pyproject.toml`, committed `uv.lock`). Core parser and storage use only
Python's standard library (`xml`, `sqlite3`); the MCP SDK is the one substantial
dependency, added in Phase 2. Tests: pytest. CI (GitHub Actions) runs ruff and the
suite on Python 3.11 and 3.12 for every pull request, and branch protection on `main`
makes a failure block the merge.
