"""
Regression tests for B15 — single source of truth for session bucketing.

Before this fix:
  * ``session_clock.active_market_session_key`` keyed off WITA (UTC+8).
  * ``dataset_logger._session(hour)`` rolled its own UTC-hour table with
    different cutoffs, so the same trade's ``TradeRecord.session`` and
    parquet row disagreed by ~1-2 hours.

After the fix, ``dataset_logger`` funnels session lookups through
``session_clock.active_market_session_key``.  These tests pin that
contract so a future refactor can't quietly reintroduce the drift.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.data.dataset_logger import _session_from_utc
from src.session_clock import WITA, active_market_session_key


# Spot-check at every UTC hour: dataset_logger and session_clock must agree.
@pytest.mark.parametrize("utc_hour", list(range(24)))
def test_session_lookup_matches_session_clock_at_every_hour(utc_hour: int):
    dt_utc = datetime(2026, 5, 22, utc_hour, 0, 0, tzinfo=timezone.utc)

    dataset_session = _session_from_utc(dt_utc)
    canonical_session = active_market_session_key(dt_utc.astimezone(WITA))

    assert dataset_session == canonical_session, (
        f"UTC {utc_hour:02d}:00 -> WITA {dt_utc.astimezone(WITA).hour:02d}:00; "
        f"dataset_logger says {dataset_session!r}, "
        f"session_clock says {canonical_session!r}"
    )


def test_session_lookup_handles_naive_datetime_as_utc():
    """A datetime without tzinfo must be treated as UTC, not local time.

    The ``_extract_signal_fields`` path always passes
    ``datetime.fromtimestamp(now, tz=timezone.utc)`` today, but a future
    caller could regress by dropping the tz arg.  Defaulting to UTC keeps
    the contract stable.
    """
    naive = datetime(2026, 5, 22, 6, 0, 0)  # would be WITA 14:00 -> "asia"
    aware = naive.replace(tzinfo=timezone.utc)
    assert _session_from_utc(naive) == _session_from_utc(aware)


def test_session_lookup_covers_all_four_buckets():
    """Sanity: every documented bucket label is reachable via UTC inputs.

    Catches an accidental rename of one of the four canonical keys.
    """
    seen: set[str] = set()
    for utc_hour in range(24):
        dt_utc = datetime(2026, 5, 22, utc_hour, 0, 0, tzinfo=timezone.utc)
        seen.add(_session_from_utc(dt_utc))

    assert seen == {"asia", "london", "ny", "overlap_london_ny"}
