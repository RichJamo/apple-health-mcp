"""Streaming parser for Apple Health ``export.xml``.

The real file is multiple gigabytes, so we must never build the whole XML tree in
memory. We use ``xml.etree.ElementTree.iterparse``, which hands us elements one at
a time as the parser reaches them, and we free each element (and the root's
accumulated children) as we go. See ``docs/architecture.md`` for the data facts.

Public API:
- ``iter_workouts(source)``  -> yields :class:`Workout`
- ``iter_records(source, types=None)`` -> yields :class:`Record`   (Rich's task)

Both take ``source`` = a path (str or ``os.PathLike``) or an open binary file.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET

# Apple writes timestamps as e.g. "2019-01-22 08:05:11 +0200": local time plus a
# UTC offset. %z parses the offset, so the resulting datetime is timezone-aware and
# we never lose which offset the reading was taken in.
_APPLE_DATETIME = "%Y-%m-%d %H:%M:%S %z"


@dataclass(frozen=True)
class Workout:
    """One workout session.

    Invariants:
    - ``start`` and ``end`` are timezone-aware datetimes (offset preserved).
    - ``duration_s`` is seconds, or ``None`` if the source omitted a duration.
    - ``active_energy_kcal`` comes from the ``ActiveEnergyBurned`` WorkoutStatistics
      child, NOT basal energy, and is ``None`` when the workout has no such child.
    - ``metadata`` has one entry per key (Apple duplicates some; last value wins).
    """

    activity_type: str
    start: datetime
    end: datetime
    duration_s: float | None
    active_energy_kcal: float | None
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


def _workout_from_element(elem: ET.Element) -> Workout:
    """Build a :class:`Workout` from one ``<Workout>`` element and its children."""
    active_energy_kcal: float | None = None
    metadata: dict[str, str] = {}

    for child in elem:
        if child.tag == "WorkoutStatistics":
            stat_type = child.attrib.get("type", "")
            if stat_type.endswith("ActiveEnergyBurned"):
                total = child.attrib.get("sum")
                if total is not None:
                    active_energy_kcal = float(total)
        elif child.tag == "MetadataEntry":
            # Later duplicates overwrite earlier ones, collapsing to one per key.
            metadata[child.attrib["key"]] = child.attrib["value"]

    return Workout(
        activity_type=elem.attrib["workoutActivityType"],
        start=_parse_apple_datetime(elem.attrib["startDate"]),
        end=_parse_apple_datetime(elem.attrib["endDate"]),
        duration_s=_duration_seconds(elem.attrib.get("duration"), elem.attrib.get("durationUnit")),
        active_energy_kcal=active_energy_kcal,
        source_name=elem.attrib.get("sourceName", ""),
        metadata=metadata,
    )


def iter_records(source, types: Iterable[str] | None =None) -> Iterator[Record]:
    wanted = set(types) if types is not None else None
    for elem in _iter_elements(source, "Record"):
        if wanted is None or elem.attrib["type"] in wanted:
            yield _record_from_element(elem)

def _record_from_element(elem: ET.Element) -> Record:
    """Build a :class:`Record` from one ``<Record>`` element."""
    return Record(
        type=elem.attrib["type"],
        start=_parse_apple_datetime(elem.attrib["startDate"]),
        end=_parse_apple_datetime(elem.attrib["endDate"]),
        value=elem.attrib.get("value"),
        unit=elem.attrib.get("unit"),
        source_name=elem.attrib["sourceName"],
    )