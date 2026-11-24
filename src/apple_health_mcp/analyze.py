"""Training analysis over the local database: the pieces the summary is built from.

It starts with de-duplication (issue #7). Import keeps a faithful copy of the export,
so a session recorded by two sources (the Watch's own workout plus a surf-tracking app
or Strava) is stored twice. Session counts and weekly minutes must count it once, and
``distinct_sessions`` decides which record stands for it.

Public API:
- ``StoredWorkout``                         -> one row of the workouts table, typed
- ``load_workouts(conn)``                   -> every stored workout, in start order
- ``heart_rate_counts_by_source(conn)``     -> heart-rate readings stored per source
- ``is_same_session(a, b)``                 -> are two records one session?
- ``distinct_sessions(workouts, counts)``   -> one record per session

Read-only: nothing here writes to the database.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

from apple_health_mcp.ingest import HEART_RATE


@dataclass(frozen=True)
class StoredWorkout:
    """One workout as stored, read back with typed fields.

    Invariants:
    - ``start`` and ``end`` are timezone-aware datetimes in UTC.
    - ``tz`` is the zone it was recorded in: an IANA name such as
      "Africa/Johannesburg", or an offset such as "+0200" when the export gave none.
      It is what places a workout in a local day or week.
    - ``duration_s`` and ``active_energy_kcal`` may be ``None``, as in the table.
    """

    id: str
    activity_type: str
    start: datetime
    end: datetime
    duration_s: float | None
    active_energy_kcal: float | None
    source_name: str
    tz: str

    @property
    def recorded_seconds(self) -> float:
        """Seconds from start to end. Unlike ``duration_s``, never ``None``, and it
        includes any pauses, which is what overlap with another record is measured in."""
        return (self.end - self.start).total_seconds()


def load_workouts(conn: sqlite3.Connection) -> list[StoredWorkout]:
    """Every stored workout, sorted by start, then id.

    Precondition: the database was built by ``ingest`` (its schema is current).
    """
    rows = conn.execute(
        "SELECT id, activity_type, start_utc, end_utc, duration_s, active_energy_kcal,"
        " source_name, tz FROM workouts ORDER BY start_utc, id"
    )
    return [
        StoredWorkout(
            id=workout_id,
            activity_type=activity_type,
            start=datetime.fromisoformat(start_utc),
            end=datetime.fromisoformat(end_utc),
            duration_s=duration_s,
            active_energy_kcal=active_energy_kcal,
            source_name=source_name,
            tz=tz,
        )
        for (
            workout_id,
            activity_type,
            start_utc,
            end_utc,
            duration_s,
            active_energy_kcal,
            source_name,
            tz,
        ) in rows
    ]


def heart_rate_counts_by_source(conn: sqlite3.Connection) -> dict[str, int]:
    """How many heart-rate readings each source wrote, among those stored.

    Only workout-adjacent readings are stored, so this counts the heart rate recorded
    during and just after workouts. A source that wrote none is absent, not 0.
    ``distinct_sessions`` uses it to recognise the Watch without matching its device
    name, which is whatever the owner called it.
    """
    rows = conn.execute(
        "SELECT source_name, COUNT(*) FROM samples WHERE type = ? GROUP BY source_name",
        (HEART_RATE,),
    )
    return dict(rows.fetchall())


def is_same_session(a: StoredWorkout, b: StoredWorkout) -> bool:
    """True when ``a`` and ``b`` are one session recorded by two sources.

    The rule (Rich, issue #7): the sources differ, and the time the two records share
    is positive and at least half the shorter recording. Activity type is ignored,
    because two apps can label one session differently (Strava "Running" while the
    Watch says "Other").

    - Same source never matches: a source does not record one session twice.
    - Records that only touch, or a zero-length record, share no time and never match.
    - Symmetric: ``is_same_session(a, b) == is_same_session(b, a)``.
    """
    if a.source_name == b.source_name:
        return False
    shared = (min(a.end, b.end) - max(a.start, b.start)).total_seconds()
    shorter = min(a.recorded_seconds, b.recorded_seconds)
    return shared > 0 and 2 * shared >= shorter


def distinct_sessions(
    workouts: Iterable[StoredWorkout],
    heart_rate_counts: Mapping[str, int],
) -> list[StoredWorkout]:
    """Return one record per session, sorted by start, then id.

    Records are taken best first: the source that wrote the most heart rate
    (``heart_rate_counts``, from ``heart_rate_counts_by_source``), which is the Watch;
    then the longer recording; then the earlier start; then the smaller id. Each record
    is kept unless it is the same session (``is_same_session``) as a record already
    kept, in which case it is dropped as a second recording of that session.

    Precondition: ids are unique, as the table's primary key makes them.

    Postconditions:
    - No two kept records are the same session.
    - Every dropped record is the same session as a kept record that ranks above it.
      A record is never dropped through a chain of records it shares no time with.
    - The output is a subset of the input and does not depend on the input's order.

    Two records from one source are never the same session, so when an app records
    one long session while the Watch records two, both Watch workouts are kept and the
    app record is dropped; when an app splits one Watch session in two, both halves
    are dropped.
    """
    by_start = sorted(workouts, key=lambda w: (w.start, w.id))

    # Only records that share time can be the same session, so find each record's
    # overlapping neighbours once: in start order, a later record that starts after
    # this one ends cannot overlap it, and nor can any record after that.
    neighbours: dict[str, list[StoredWorkout]] = {w.id: [] for w in by_start}
    for index, earlier in enumerate(by_start):
        for later in by_start[index + 1 :]:
            if later.start >= earlier.end:
                break
            neighbours[earlier.id].append(later)
            neighbours[later.id].append(earlier)

    def preference(w: StoredWorkout) -> tuple:
        return (-heart_rate_counts.get(w.source_name, 0), -w.recorded_seconds, w.start, w.id)

    kept_ids: set[str] = set()
    for workout in sorted(by_start, key=preference):
        if not any(
            other.id in kept_ids and is_same_session(other, workout)
            for other in neighbours[workout.id]
        ):
            kept_ids.add(workout.id)
    return [w for w in by_start if w.id in kept_ids]
