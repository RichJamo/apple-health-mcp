"""Streaming parser for Apple Health ``export.xml``.

The real file is multiple gigabytes, so we must never build the whole XML tree in
memory. We use ``xml.etree.ElementTree.iterparse``, which hands us elements one at
a time as the parser reaches them, and we free each element (and the root's
accumulated children) as we go. See ``docs/architecture.md`` for the data facts.

Public API:
- ``iter_workouts(source)``  -> yields :class:`Workout`
- ``iter_records(source, types=None, *, on_skip=None)`` -> yields :class:`Record`
- ``MalformedRecordError`` -> raised by ``iter_records`` in strict mode (no ``on_skip``),
  and by ``analyze`` for a stored value it cannot read

Both take ``source`` = a path (str or ``os.PathLike``) or an open binary file.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime

# Apple writes timestamps as e.g. "2019-01-22 08:05:11 +0200": local time plus a
# UTC offset. %z parses the offset, so the resulting datetime is timezone-aware and
# we never lose which offset the reading was taken in.
_APPLE_DATETIME = "%Y-%m-%d %H:%M:%S %z"


class MalformedRecordError(ValueError):
    """A ``<Record>`` in the export, or a value stored in the database, could not be
    read. Its message names only the record type or the stored row's id, never a raw
    value or timestamp.

    Python's own parse errors quote the offending text (``strptime`` puts the raw
    timestamp in its message), and a traceback would print it. This error is raised
    in its place, with the quoting error detached (``from None``, or raised outside the
    ``except`` block), so no raw health value or timestamp reaches a terminal, a log,
    or an AI reading the output.
    """


@dataclass(frozen=True)
class Workout:
    """One workout session.

    Invariants:
    - ``start`` and ``end`` are timezone-aware datetimes (offset preserved).
    - ``duration_s`` is seconds, or ``None`` if the source omitted a duration.
    - ``active_energy_kcal`` comes from the ``ActiveEnergyBurned`` WorkoutStatistics
      child, NOT basal energy, and is ``None`` when the workout has no such child.
    - ``heart_rate_average_bpm`` and ``heart_rate_maximum_bpm`` are Apple's own
      figures for the whole workout, from the WorkoutStatistics child whose type is
      exactly ``HKQuantityTypeIdentifierHeartRate`` (not RestingHeartRate, which
      shares the suffix). Each is ``None`` when the workout has no such child.
    - ``metadata`` has one entry per key (Apple duplicates some; last value wins).
    """

    activity_type: str
    start: datetime
    end: datetime
    duration_s: float | None
    active_energy_kcal: float | None
    heart_rate_average_bpm: float | None
    heart_rate_maximum_bpm: float | None
    source_name: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class Record:
    """One time-series measurement (heart rate, resting HR, sleep, ...).

    Invariants:
    - ``start`` and ``end`` are timezone-aware datetimes.
    - ``value`` and ``unit`` may be ``None`` (the export marks them optional).
    """

    type: str
    start: datetime
    end: datetime
    value: str | None
    unit: str | None
    source_name: str


def _parse_apple_datetime(text: str) -> datetime:
    """Parse an Apple Health timestamp into a timezone-aware datetime.

    Precondition: ``text`` is in the form ``"YYYY-MM-DD HH:MM:SS +ZZZZ"``.
    Raises ``ValueError`` on any other shape rather than guessing.
    """
    return datetime.strptime(text, _APPLE_DATETIME)


def _duration_seconds(value: str | None, unit: str | None) -> float | None:
    """Convert a duration attribute to seconds.

    Returns ``None`` when no duration was given. Raises ``ValueError`` on an
    unrecognised unit rather than silently mis-scaling the number.
    """
    if value is None:
        return None
    seconds = float(value)
    if unit == "min":
        return seconds * 60
    if unit in (None, "s", "sec"):
        return seconds
    raise ValueError(f"unexpected durationUnit: {unit!r}")


def _iter_elements(source, tag: str) -> Iterator[ET.Element]:
    """Stream ``<tag>`` elements from an Apple Health export, freeing memory.

    This is the heart of the streaming discipline. After each matching element is
    yielded (and the caller has read it), we clear the root so finished elements do
    not pile up — without this, a multi-GB file would exhaust memory.
    """
    context = ET.iterparse(source, events=("start", "end"))
    # The first "start" event is the <HealthData> root; keep a handle to it so we
    # can drop its children as we finish with them. After consuming it, exactly one
    # element (the root) is open, so depth == 1.
    _event, root = next(context)
    depth = 1
    for event, elem in context:
        if event == "start":
            depth += 1
            continue
        # event == "end": an element just closed.
        depth -= 1
        if depth != 1:
            # A nested child (MetadataEntry, WorkoutStatistics, ...) closed; its
            # parent is still being built, so do nothing yet.
            continue
        # A direct child of <HealthData> just finished.
        if elem.tag == tag:
            yield elem
        # Clearing the root drops this element and every sibling accumulated so far,
        # whatever its tag. Doing it on EVERY top-level element (not just matches) is
        # what keeps memory flat on a multi-GB file with millions of dense records.
        root.clear()


def iter_workouts(source) -> Iterator[Workout]:
    """Yield a :class:`Workout` for every ``<Workout>`` in the export."""
    for elem in _iter_elements(source, "Workout"):
        yield _workout_from_element(elem)


# The one WorkoutStatistics type that holds the workout's own heart rate. Matched
# exactly: RestingHeartRate and others end in "HeartRate" too.
_HEART_RATE_STATISTIC = "HKQuantityTypeIdentifierHeartRate"


def _optional_float(text: str | None) -> float | None:
    """``float(text)``, or ``None`` when the attribute is absent."""
    return None if text is None else float(text)


def _workout_from_element(elem: ET.Element) -> Workout:
    """Build a :class:`Workout` from one ``<Workout>`` element and its children."""
    active_energy_kcal: float | None = None
    heart_rate_average_bpm: float | None = None
    heart_rate_maximum_bpm: float | None = None
    metadata: dict[str, str] = {}

    for child in elem:
        if child.tag == "WorkoutStatistics":
            stat_type = child.attrib.get("type", "")
            if stat_type.endswith("ActiveEnergyBurned"):
                total = child.attrib.get("sum")
                if total is not None:
                    active_energy_kcal = float(total)
            elif stat_type == _HEART_RATE_STATISTIC:
                heart_rate_average_bpm = _optional_float(child.attrib.get("average"))
                heart_rate_maximum_bpm = _optional_float(child.attrib.get("maximum"))
        elif child.tag == "MetadataEntry":
            # Later duplicates overwrite earlier ones, collapsing to one per key.
            metadata[child.attrib["key"]] = child.attrib["value"]

    return Workout(
        activity_type=elem.attrib["workoutActivityType"],
        start=_parse_apple_datetime(elem.attrib["startDate"]),
        end=_parse_apple_datetime(elem.attrib["endDate"]),
        duration_s=_duration_seconds(elem.attrib.get("duration"), elem.attrib.get("durationUnit")),
        active_energy_kcal=active_energy_kcal,
        heart_rate_average_bpm=heart_rate_average_bpm,
        heart_rate_maximum_bpm=heart_rate_maximum_bpm,
        source_name=elem.attrib.get("sourceName", ""),
        metadata=metadata,
    )


def iter_records(
    source,
    types: Iterable[str] | None = None,
    *,
    on_skip: Callable[[str], None] | None = None,
) -> Iterator[Record]:
    """Yield a :class:`Record` for every ``<Record>`` of a wanted type in the export.

    ``types`` restricts to those type identifiers (``None`` yields every record). With
    a filter, a ``<Record>`` that has no ``type`` at all is simply not wanted, so it is
    passed over like any other unwanted type.

    Error handling (the skip-and-report contract; docs/architecture.md, issue #3):
    - Without ``on_skip`` (strict), a record that fails to parse raises
      :class:`MalformedRecordError`, aborting the import. Its message names only the
      record type, and the original error is suppressed (``from None``), because that
      error quotes the raw timestamp.
    - With ``on_skip``, the bad record is skipped and ``on_skip`` is called with only
      its type identifier, never its raw value or timestamp.
    """
    wanted = set(types) if types is not None else None
    for elem in _iter_elements(source, "Record"):
        record_type = elem.attrib.get("type")
        if wanted is not None and record_type not in wanted:
            continue
        try:
            record = _record_from_element(elem)
        except (ValueError, KeyError):
            if on_skip is None:
                raise MalformedRecordError(f"malformed {record_type or 'untyped'} record") from None
            on_skip(record_type or "")
            continue
        yield record


def _record_from_element(elem: ET.Element) -> Record:
    """Build a :class:`Record` from one ``<Record>`` element."""
    return Record(
        type=elem.attrib["type"],
        start=_parse_apple_datetime(elem.attrib["startDate"]),
        end=_parse_apple_datetime(elem.attrib["endDate"]),
        value=elem.attrib.get("value"),
        unit=elem.attrib.get("unit"),
        source_name=elem.attrib.get("sourceName", ""),
    )
