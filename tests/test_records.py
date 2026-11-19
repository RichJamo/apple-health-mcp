from datetime import datetime, timedelta, timezone
from pathlib import Path

from apple_health_mcp.parse import Record, iter_records

FIXTURE = Path(__file__).parent / "fixtures" / "sample_export.xml"
SAST = timezone(timedelta(hours=2))


def test_parses_every_record_when_unfiltered():
    # The fixture has 5 <Record> elements; <Workout>s are not records.
    assert len(list(iter_records(FIXTURE))) == 5


def test_types_filter_selects_only_matching_records():
    hr = list(iter_records(FIXTURE, types={"HKQuantityTypeIdentifierHeartRate"}))
    assert len(hr) == 3
    assert all(r.type == "HKQuantityTypeIdentifierHeartRate" for r in hr)


def test_record_fields_and_offset():
    first = next(iter_records(FIXTURE, types={"HKQuantityTypeIdentifierHeartRate"}))
    assert first.value == "92"
    assert first.unit == "count/min"
    assert first.source_name == "Test Watch"
    assert first.start == datetime(2019, 1, 22, 8, 10, 0, tzinfo=SAST)


def test_missing_unit_becomes_none():
    sleep = next(iter_records(FIXTURE, types={"HKCategoryTypeIdentifierSleepAnalysis"}))
    assert sleep.unit is None          # the sleep Record has no unit attribute
    assert sleep.value == "HKCategoryValueSleepAnalysisAsleepCore"
