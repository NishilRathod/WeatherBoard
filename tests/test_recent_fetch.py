"""The daily top-up: which endpoint, which days, and what a gap does.

Two endpoints have to cover one range because neither can do it alone -- ERA5
lags about five days, and the forecast endpoint only reaches 92 days back. The
seam between them is the interesting part: a day fetched from the wrong side of
it is either missing or silently attributed to the wrong date.
"""

from datetime import date, timedelta

import pytest

from app.services.city_index import CityRecord
from app.services.recent import Run, parse_runs
from scripts.fetch_recent_daily import (
    _backfill_gaps,
    _merge,
    _readings_by_date,
    _recency_gaps,
    _work_units,
    forecast_url,
    plan_windows,
)

TODAY = date(2026, 8, 17)
YESTERDAY = date(2026, 8, 16)


def _city(geonameid, population=200_000, row_index=0):
    return CityRecord(
        row_index=row_index,
        geonameid=geonameid,
        name=f"C{geonameid}",
        state="",
        country="XX",
        population=population,
        latitude=1.0,
        longitude=2.0,
    )


class TestWindowPlanning:
    def test_the_daily_top_up_is_one_forecast_call(self):
        """The normal case: yesterday only, which the archive does not have yet."""
        segments = plan_windows(YESTERDAY, YESTERDAY, TODAY)

        assert segments == [("forecast", YESTERDAY, YESTERDAY)]

    def test_a_range_straddling_the_lag_splits_at_it(self):
        """Days old enough for the archive come from the archive, the rest do not."""
        segments = plan_windows(date(2026, 8, 10), YESTERDAY, TODAY)

        assert segments == [
            ("archive", date(2026, 8, 10), date(2026, 8, 11)),
            ("forecast", date(2026, 8, 12), YESTERDAY),
        ]

    def test_a_backfill_uses_the_archive_then_the_forecast_tail(self):
        segments = plan_windows(date(2026, 1, 1), YESTERDAY, TODAY)

        assert [name for name, _, _ in segments] == ["archive", "forecast"]
        archive, forecast = segments
        # The two must meet exactly: a day in neither is lost, a day in both is
        # fetched twice.
        assert archive[2] + timedelta(days=1) == forecast[1]
        assert forecast[2] == YESTERDAY

    def test_the_archive_stops_short_of_the_reanalysis_lag(self):
        archive, _ = plan_windows(date(2026, 1, 1), YESTERDAY, TODAY)

        assert archive[2] == TODAY - timedelta(days=6)

    def test_the_forecast_tail_cannot_reach_past_its_limit(self):
        """Beyond 92 days the endpoint simply will not serve, so do not ask."""
        segments = plan_windows(date(2020, 1, 1), YESTERDAY, TODAY)

        _, forecast = segments
        assert forecast[1] >= TODAY - timedelta(days=92)

    def test_an_empty_range_asks_for_nothing(self):
        assert plan_windows(TODAY, YESTERDAY, TODAY) == []


class TestTodayIsExcluded:
    def test_the_forecast_call_asks_for_no_forecast_days(self):
        """Today's daily mean is a partial day and must never reach the cache.

        Scoring a part-day mean against a whole-day baseline is what produces
        five-sigma anomalies for whichever cities happen to be mid-afternoon.
        """
        url = forecast_url([_city(1)], 5)

        assert "forecast_days=0" in url
        assert "past_days=5" in url


class TestRecencyPhase:
    """Every city is brought to yesterday before any history is filled in.

    The board scores each city's most recent complete day and shows one date for
    the whole board, so a city filled oldest-first would rank on a months-old
    reading while looking exactly as current as its neighbours.
    """

    def test_a_city_is_asked_only_for_the_days_it_lacks(self):
        runs = {1: Run(start=date(2026, 8, 1), temps=[1.0] * 10, humidities=[1.0] * 10)}

        groups = _recency_gaps([_city(1)], runs, YESTERDAY, 7)

        assert list(groups) == [(date(2026, 8, 11), YESTERDAY)]

    def test_a_current_city_is_not_asked_at_all(self):
        runs = {1: Run(start=date(2026, 8, 1), temps=[1.0] * 16, humidities=[1.0] * 16)}

        assert _recency_gaps([_city(1)], runs, YESTERDAY, 7) == {}

    def test_a_long_gap_is_bounded_by_the_recency_window(self):
        """Catching up after weeks away must not cost weeks of quota up front."""
        runs = {1: Run(start=date(2026, 1, 1), temps=[1.0] * 31, humidities=[1.0] * 31)}

        ((start, end),) = _recency_gaps([_city(1)], runs, YESTERDAY, 7)

        assert (end - start).days + 1 == 7
        assert end == YESTERDAY

    def test_cities_wanting_the_same_span_share_one_group(self):
        groups = _recency_gaps([_city(1), _city(2)], {}, YESTERDAY, 7)

        assert len(groups) == 1
        assert len(next(iter(groups.values()))) == 2


class TestBackfillPhase:
    def test_history_is_extended_backwards_from_the_stored_run(self):
        runs = {1: Run(start=date(2026, 8, 10), temps=[1.0] * 7, humidities=[1.0] * 7)}

        ((start, end),) = _backfill_gaps([_city(1)], runs, date(2026, 1, 1), None)

        assert end == date(2026, 8, 9)
        assert start == date(2026, 1, 1)

    def test_limit_days_takes_the_newest_slice_first(self):
        """A capped backfill deepens the recent past before the distant past."""
        runs = {1: Run(start=date(2026, 8, 10), temps=[1.0] * 7, humidities=[1.0] * 7)}

        ((start, end),) = _backfill_gaps([_city(1)], runs, date(2026, 1, 1), 30)

        assert end == date(2026, 8, 9)
        assert (end - start).days + 1 == 30

    def test_a_city_with_no_run_is_left_to_the_recency_phase(self):
        assert _backfill_gaps([_city(1)], {}, date(2026, 1, 1), None) == {}

    def test_a_complete_history_asks_for_nothing(self):
        runs = {1: Run(start=date(2026, 1, 1), temps=[1.0] * 30, humidities=[1.0] * 30)}

        assert _backfill_gaps([_city(1)], runs, date(2026, 1, 1), None) == {}


class TestResponseAlignment:
    def test_readings_are_keyed_by_the_date_the_server_gave_them(self):
        readings = _readings_by_date(
            {
                "time": ["2026-08-15", "2026-08-16"],
                "temperature_2m_mean": [10.0, 11.0],
                "relative_humidity_2m_mean": [50.0, 55.0],
            }
        )

        assert readings[date(2026, 8, 16)] == (11.0, 55.0)

    def test_a_null_day_stays_null_rather_than_becoming_zero(self):
        readings = _readings_by_date(
            {
                "time": ["2026-08-16"],
                "temperature_2m_mean": [None],
                "relative_humidity_2m_mean": [None],
            }
        )

        assert readings[date(2026, 8, 16)] == (None, None)


class TestMerging:
    def test_new_days_extend_the_stored_run(self):
        run = Run(start=date(2026, 8, 14), temps=[10.0], humidities=[50.0])

        merged = _merge(run, {date(2026, 8, 15): (11.0, 55.0)})

        assert merged.start == date(2026, 8, 14)
        assert merged.temps == [10.0, 11.0]
        assert merged.end == date(2026, 8, 15)

    def test_a_missing_day_is_held_open_rather_than_closed_up(self):
        """A gap must not slide a later reading into an earlier day's slot."""
        run = Run(start=date(2026, 8, 14), temps=[10.0], humidities=[50.0])

        merged = _merge(run, {date(2026, 8, 16): (12.0, 60.0)})

        assert merged.temps == [10.0, None, 12.0]
        assert merged.start == date(2026, 8, 14)

    def test_the_backfill_extends_the_run_backwards(self):
        """The two phases grow the run from opposite ends; neither may truncate."""
        run = Run(start=date(2026, 8, 14), temps=[10.0], humidities=[50.0])

        merged = _merge(run, {date(2026, 8, 12): (8.0, 40.0)})

        assert merged.start == date(2026, 8, 12)
        assert merged.end == date(2026, 8, 14)
        assert merged.temps == [8.0, None, 10.0]

    def test_a_refetched_day_overwrites_rather_than_duplicates(self):
        run = Run(start=date(2026, 8, 14), temps=[10.0], humidities=[50.0])

        merged = _merge(run, {date(2026, 8, 14): (99.0, 9.0)})

        assert merged.temps == [99.0]

    def test_a_merged_run_survives_a_round_trip_through_the_file(self):
        from app.services.recent import record_line

        merged = _merge(
            Run(start=date(2026, 8, 14), temps=[10.0], humidities=[50.0]),
            {YESTERDAY: (11.0, 55.0)},
        )
        line = record_line(1, merged.start, merged.temps, merged.humidities)

        back = parse_runs([line])[1]
        assert back.start == merged.start
        assert back.end == YESTERDAY
        assert back.temps[-1] == pytest.approx(11.0, abs=0.05)


class TestProgressAccounting:
    """The denominator has to count whatever the numerator counts.

    A range straddling the archive/forecast seam is two requests for the same
    city, and the progress line ticks once per request as it lands. Totalling
    cities instead made a healthy run report ``7,620/3,810 cities`` -- a number
    that reads as a bug in the fetch itself rather than in the counting.
    """

    def test_a_city_spanning_both_endpoints_counts_twice(self):
        groups = {(date(2026, 1, 1), YESTERDAY): [_city(1)]}

        assert _work_units(groups, TODAY) == 2

    def test_a_city_inside_one_endpoint_counts_once(self):
        groups = {(date(2026, 8, 14), YESTERDAY): [_city(1)]}

        assert _work_units(groups, TODAY) == 1

    def test_every_city_in_a_group_carries_the_group_s_segments(self):
        groups = {(date(2026, 1, 1), YESTERDAY): [_city(1), _city(2)]}

        assert _work_units(groups, TODAY) == 4

    def test_the_total_matches_what_a_completed_phase_would_tick(self):
        """Walk the same loop the fetch walks, and count what it would report."""
        groups = {
            (date(2026, 1, 1), YESTERDAY): [_city(1), _city(2)],
            (date(2026, 8, 14), YESTERDAY): [_city(3)],
        }

        ticked = sum(
            len(cities)
            for (start, end), cities in groups.items()
            for _ in plan_windows(start, end, TODAY)
        )

        assert _work_units(groups, TODAY) == ticked


class TestPacingIsAnnounced:
    """A long pause has to say so.

    The gap between batches is the pacer's, not the server's, and it grows as
    throttling drives the rate down -- at the 5.0 floor a wide archive batch
    waits roughly 25 minutes. Silence that long is indistinguishable from a
    hung process, and has twice sent someone digging through netstat to find
    nothing wrong. A pause worth noticing announces itself.
    """

    def _run(self, monkeypatch, capsys, rate):
        import io
        import time as time_module

        from scripts import fetch_recent_daily as mod

        def fake_fetch_batch(batch, make_url, *, timeout):
            return [{} for _ in batch], False

        slept = []
        monkeypatch.setattr(mod, "fetch_batch", fake_fetch_batch)
        monkeypatch.setattr(time_module, "sleep", slept.append)

        start, end = date(2026, 1, 1), date(2026, 8, 10)
        groups = {(start, end): [_city(1)]}
        mod._run_phase(
            "backfill", groups, {}, TODAY, mod.Pacer(rate), 180.0, io.StringIO()
        )
        return slept, capsys.readouterr().out

    def test_a_long_pacing_sleep_is_logged(self, monkeypatch, capsys):
        slept, out = self._run(monkeypatch, capsys, rate=0.5)

        assert any(s > 30 for s in slept), f"expected a long sleep, got {slept}"
        assert "pacing" in out, f"long pause was silent; log was:\n{out}"

    def test_a_short_pacing_sleep_stays_quiet(self, monkeypatch, capsys):
        """One line per batch already exists; do not double the log for a blip."""
        _slept, out = self._run(monkeypatch, capsys, rate=600.0)

        assert "pacing" not in out, f"short pause should not log; log was:\n{out}"
