"""Training analysis over the local database: the pieces the summary is built from.

De-duplication (issue #7): import keeps a faithful copy of the export, so a session
recorded by two sources (the Watch's own workout plus a surf-tracking app or Strava)
is stored twice. Session counts and weekly minutes must count it once, and
``distinct_sessions`` decides which record stands for it.

Heart rate: ``heart_rate_summary`` gives one session's average and peak heart rate,
its measured recovery (the drop from the session's peak to one minute after the end),
and Apple's own one-minute recovery figure, labelled as an estimate.

Public API:
- ``StoredWorkout``                         -> one row of the workouts table, typed
- ``load_workouts(conn)``                   -> every stored workout, in start order
- ``heart_rate_counts_by_source(conn)``     -> heart-rate readings stored per source
- ``is_same_session(a, b)``                 -> are two records one session?
- ``distinct_sessions(workouts, counts)``   -> one record per session
- ``Reading`` / ``readings_around(conn, w)`` -> a session's stored readings, typed
- ``HeartRateSummary`` / ``heart_rate_summary(w, readings)`` -> its heart-rate figures
- ``WeekSummary`` / ``weekly_summary(sessions)`` / ``weekly_summary_from_db(conn)``
                                           -> training per local week

Read-only: nothing here writes to the database.
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apple_health_mcp.ingest import HEART_RATE, HEART_RATE_RECOVERY, RECOVERY_WINDOW
from apple_health_mcp.parse import MalformedRecordError

# Measured recovery reads heart rate one minute after the workout ends, from the
# instant reading nearest that moment within this tolerance either side.
ONE_MINUTE_AFTER = timedelta(seconds=60)
ONE_MINUTE_TOLERANCE = timedelta(seconds=15)


@dataclass(frozen=True)
class StoredWorkout:
    """One workout as stored, read back with typed fields.

    Invariants:
    - ``start`` and ``end`` are timezone-aware datetimes in UTC.
    - ``tz`` is the export's HKTimeZone, the zone it was recorded in, such as
      "Africa/Johannesburg"; or, when the export gave none, the offset in its
      startDate, such as "+0200", which is the exporting phone's offset (see
      ``local_start``). It is what places a workout in a local day or week.
    - ``duration_s`` and ``active_energy_kcal`` may be ``None``, as in the table.
    - ``heart_rate_average_bpm`` and ``heart_rate_maximum_bpm`` are Apple's own
      figures for the whole workout, or ``None`` when the export had none.
    """

    id: str
    activity_type: str
    start: datetime
    end: datetime
    duration_s: float | None
    active_energy_kcal: float | None
    heart_rate_average_bpm: float | None
    heart_rate_maximum_bpm: float | None
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
        " heart_rate_average_bpm, heart_rate_maximum_bpm, source_name, tz"
        " FROM workouts ORDER BY start_utc, id"
    )
    return [
        StoredWorkout(
            id=workout_id,
            activity_type=activity_type,
            start=datetime.fromisoformat(start_utc),
            end=datetime.fromisoformat(end_utc),
            duration_s=duration_s,
            active_energy_kcal=active_energy_kcal,
            heart_rate_average_bpm=heart_rate_average_bpm,
            heart_rate_maximum_bpm=heart_rate_maximum_bpm,
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
            heart_rate_average_bpm,
            heart_rate_maximum_bpm,
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


@dataclass(frozen=True)
class Reading:
    """One stored reading of heart rate or of Apple's one-minute recovery figure.

    Invariants: ``start`` and ``end`` are timezone-aware UTC datetimes; ``end`` equals
    ``start`` for an instant reading and is later for a span (one value Apple stores
    as the average over several minutes); ``bpm`` is a number, never absent.
    """

    type: str
    start: datetime
    end: datetime
    bpm: float

    @property
    def is_instant(self) -> bool:
        return self.end == self.start


def readings_around(conn: sqlite3.Connection, workout: StoredWorkout) -> list[Reading]:
    """The heart-rate and recovery readings that start inside ``workout`` or in its
    recovery tail, ``[start, end + RECOVERY_WINDOW]``, sorted by start.

    Selected by time range, not by ``workout_id``, because a reading shared by two
    overlapping workouts is tagged with only one of them. A reading stored without a
    value is left out. Both bounds are inclusive and taken in UTC, as stored, whatever
    offset ``workout``'s times carry.

    Raises :class:`~apple_health_mcp.parse.MalformedRecordError` if a stored value is
    not a finite number ("NaN" and "inf" parse as floats but would read as real
    figures). Its message names only the reading's type, and no other error is attached
    to it: Python's own error would quote the raw value.
    """
    rows = conn.execute(
        "SELECT type, start_utc, end_utc, value FROM samples"
        " WHERE type IN (?, ?) AND start_utc >= ? AND start_utc <= ? AND value IS NOT NULL"
        " ORDER BY start_utc, type",
        (
            HEART_RATE,
            HEART_RATE_RECOVERY,
            workout.start.astimezone(UTC).isoformat(),
            (workout.end + RECOVERY_WINDOW).astimezone(UTC).isoformat(),
        ),
    )
    return [
        Reading(
            type=kind,
            start=datetime.fromisoformat(start_utc),
            end=datetime.fromisoformat(end_utc),
            bpm=_stored_bpm(kind, value),
        )
        for kind, start_utc, end_utc, value in rows
    ]


def _stored_bpm(kind: str, value: str) -> float:
    """``float(value)`` if it is a finite number; otherwise an error that does not
    quote the value.

    The error is raised after the ``except`` block has finished, so Python's own
    ValueError (which quotes the value) is not attached as its ``__context__`` either,
    for anything that walks the chain without honouring ``from None``.
    """
    try:
        bpm = float(value)
    except ValueError:
        bpm = math.nan  # handled below, outside the except block
    if not math.isfinite(bpm):
        raise MalformedRecordError(f"non-numeric {kind} value in the database")
    return bpm


@dataclass(frozen=True)
class HeartRateSummary:
    """One session's heart-rate figures, each ``None`` when there is nothing to base
    it on (never 0, which would read as a real heart rate).

    - ``average_bpm``: Apple's average, or else computed from the readings.
    - ``peak_bpm``: Apple's maximum only. Spans are averages and hide the peak, so
      the readings never stand in for it.
    - ``measured_recovery_bpm``: the drop from the session's peak to the heart rate
      one minute after the end. It equals a true one-minute recovery only when the
      peak came at the end; when the peak came earlier (intervals, surfing), or the
      workout was ended a while after the effort stopped, it covers more than a
      minute of recovery and reads high. It can be negative, when the reading after
      the end is above Apple's maximum.
    - ``apple_recovery_estimate_bpm``: Apple's one-minute recovery record. Apple can
      write it as a prediction rather than a measurement, and the import does not
      keep which, so it is an estimate, never ground truth.
    """

    average_bpm: float | None
    peak_bpm: float | None
    measured_recovery_bpm: float | None
    apple_recovery_estimate_bpm: float | None


def heart_rate_summary(workout: StoredWorkout, readings: Iterable[Reading]) -> HeartRateSummary:
    """Summarise one session's heart rate from Apple's statistic and its readings.

    Precondition: ``readings`` are this workout's, as ``readings_around`` returns them,
    in any order. Any iterable works, a generator included: it is read once.

    - Average: Apple's when present. Otherwise the mean of the heart-rate spans that
      start inside the workout, each weighted by the time it covers before the end;
      when no span covers any time inside, the plain mean of instant readings inside
      ``[start, end]``.
    - Measured recovery: ``peak_bpm`` minus the instant heart-rate reading nearest to
      ``end + ONE_MINUTE_AFTER``, within ``ONE_MINUTE_TOLERANCE`` either side (both
      bounds inclusive; a tie goes to the earlier reading). Spans are excluded.
    - Apple's estimate: the earliest recovery record in ``[end, end + RECOVERY_WINDOW]``.
    """
    readings = list(readings)  # read once: a generator would be empty the second time
    heart_rate = [r for r in readings if r.type == HEART_RATE]
    recovery_records = sorted(
        (r for r in readings if r.type == HEART_RATE_RECOVERY), key=lambda r: r.start
    )

    peak = workout.heart_rate_maximum_bpm
    average = workout.heart_rate_average_bpm
    if average is None:
        average = _average_from_readings(workout, heart_rate)

    target = workout.end + ONE_MINUTE_AFTER
    near_one_minute = [
        r for r in heart_rate if r.is_instant and abs(r.start - target) <= ONE_MINUTE_TOLERANCE
    ]
    measured_recovery = None
    if peak is not None and near_one_minute:
        nearest = min(near_one_minute, key=lambda r: (abs(r.start - target), r.start))
        measured_recovery = peak - nearest.bpm

    in_tail = [
        r for r in recovery_records if workout.end <= r.start <= workout.end + RECOVERY_WINDOW
    ]
    apple_estimate = in_tail[0].bpm if in_tail else None

    return HeartRateSummary(
        average_bpm=average,
        peak_bpm=peak,
        measured_recovery_bpm=measured_recovery,
        apple_recovery_estimate_bpm=apple_estimate,
    )


def _average_from_readings(workout: StoredWorkout, heart_rate: list[Reading]) -> float | None:
    """The fallback average when Apple gave none; ``None`` with nothing inside.

    A span whose end is before its start (never seen, and not checked at import)
    gets no weight rather than a negative one, so the mean stays within its inputs.
    """
    inside = [r for r in heart_rate if workout.start <= r.start <= workout.end]
    spans = [r for r in inside if not r.is_instant]
    if spans:
        weights = [max(0.0, (min(r.end, workout.end) - r.start).total_seconds()) for r in spans]
        if sum(weights) > 0:
            return sum(w * r.bpm for w, r in zip(weights, spans, strict=True)) / sum(weights)
    instants = [r.bpm for r in inside if r.is_instant]
    return sum(instants) / len(instants) if instants else None


# --- Weekly training summary ------------------------------------------------------------
#
# Weeks run Monday to Sunday in each session's local time (``local_start`` says what
# "local" can mean), because a Monday 00:30 run in Johannesburg is Sunday 22:30 in UTC.


@dataclass(frozen=True)
class WeekSummary:
    """One Monday-to-Sunday week of training, in local time.

    - ``week_start``: the Monday that starts the week, as a local calendar date.
    - ``sessions``: how many sessions started in the week (0 for an empty week).
    - ``minutes``: the sessions' durations added up, from ``duration_s`` (the
      duration its source recorded, which leaves pauses out on Apple Watch workouts),
      or from start to end when that is ``None``.
    - ``active_energy_kcal``: the known energy figures added up; ``None`` when no
      session in the week has one, which includes a week with no sessions.
    - ``sessions_by_type``: sessions per ``activity_type``; its counts add up to
      ``sessions``. Empty for an empty week.
    """

    week_start: date
    sessions: int
    minutes: float
    active_energy_kcal: float | None
    sessions_by_type: dict[str, int]


def local_start(workout: StoredWorkout) -> datetime:
    """``workout.start`` shown in the zone that ``workout.tz`` names.

    ``workout.tz`` is one of two things:
    - a zone name that ``zoneinfo.ZoneInfo`` accepts, such as "Africa/Johannesburg":
      the export's HKTimeZone, the zone the workout was recorded in;
    - an offset of the form "+HHMM" / "-HHMM", such as "+0200", when the export had
      no HKTimeZone. Apple writes every date in the zone the phone was in when the
      export was made, so this is the exporting phone's offset. For a workout
      recorded in another zone it is not the recording zone, and the export holds
      nothing better (docs/architecture.md, data fact 3).

    Postcondition: the result is the same moment as ``workout.start`` (equal when
    compared), only with the local offset, so ``.date()`` gives the local calendar day.

    Raises ``MalformedRecordError`` when ``tz`` is neither a zone this machine's time
    zone database knows nor a "+HHMM" offset. The importer stores HKTimeZone as the
    export gives it, so any app could have written it. The message names the
    workout's id, not the value, and is raised outside the ``except`` block, as in
    ``_stored_bpm``.
    """
    try:
        zone = _recorded_zone(workout.tz)
    except (ZoneInfoNotFoundError, ValueError):
        zone = None  # handled below, outside the except block
    if zone is None:
        raise MalformedRecordError(f"unrecognised time zone on stored workout {workout.id}")
    return workout.start.astimezone(zone)


def _recorded_zone(tz: str) -> tzinfo:
    """The zone a stored ``tz`` value names. When the export had no HKTimeZone, the
    importer wrote the offset with ``strftime("%z")``, so it is read back with
    ``strptime("%z")``. A zone name never starts with a sign."""
    if tz.startswith(("+", "-")):
        return datetime.strptime(tz, "%z").tzinfo
    return ZoneInfo(tz)


def weekly_summary(sessions: Iterable[StoredWorkout]) -> list[WeekSummary]:
    """One :class:`WeekSummary` per local week, from the first session's week to the
    last session's week, oldest first. An empty input gives ``[]``.

    Precondition: ``sessions`` are distinct sessions (from ``distinct_sessions``);
    this function does not de-duplicate.

    A session belongs to the week in which it *starts*, in local time
    (``local_start``), even if it runs past Sunday midnight.

    Postconditions (checked by the tests):
    - every ``week_start`` is a Monday, and each is seven days after the one before:
      weeks with no session in between are included, so gaps show in a trend;
    - the ``sessions`` counts add up to the number of input sessions;
    - the result does not depend on the input's order.

    Raises ``MalformedRecordError`` (from ``local_start``) when a session's time zone
    cannot be loaded.
    """
    # Sorting first makes the output identical whatever order the caller used, down to
    # the order of the keys in sessions_by_type, which a report prints.
    sessions_by_week: dict[date, list[StoredWorkout]] = defaultdict(list)
    for session in sorted(sessions, key=lambda s: (s.start, s.id)):
        sessions_by_week[_monday_of(local_start(session).date())].append(session)
    if not sessions_by_week:
        return []

    weeks = []
    week, last_week = min(sessions_by_week), max(sessions_by_week)
    while week <= last_week:
        weeks.append(_summarise_week(week, sessions_by_week.get(week, [])))
        week += timedelta(days=7)
    return weeks


def _monday_of(day: date) -> date:
    """The Monday on or before ``day``. ``weekday()`` is 0 on Monday, 6 on Sunday."""
    return day - timedelta(days=day.weekday())


def _summarise_week(week_start: date, sessions: list[StoredWorkout]) -> WeekSummary:
    """One week's figures from the sessions that started in it (empty for a gap week)."""
    known_energy = [s.active_energy_kcal for s in sessions if s.active_energy_kcal is not None]
    return WeekSummary(
        week_start=week_start,
        sessions=len(sessions),
        minutes=sum((_session_minutes(s) for s in sessions), start=0.0),
        active_energy_kcal=sum(known_energy) if known_energy else None,
        sessions_by_type=dict(Counter(s.activity_type for s in sessions)),
    )


def _session_minutes(session: StoredWorkout) -> float:
    """The duration its source recorded (pauses left out on Apple Watch workouts),
    else the recorded span from start to end."""
    seconds = session.duration_s if session.duration_s is not None else session.recorded_seconds
    return seconds / 60


def weekly_summary_from_db(conn: sqlite3.Connection) -> list[WeekSummary]:
    """The weekly summary of every stored session, read from the database.

    It must count a session recorded by two sources once: the workouts table keeps
    both records, so this starts from ``distinct_sessions`` (with the heart-rate
    counts it needs), never from the table's rows directly.

    Raises ``MalformedRecordError`` as ``weekly_summary`` does.
    """
    sessions = distinct_sessions(load_workouts(conn), heart_rate_counts_by_source(conn))
    return weekly_summary(sessions)
