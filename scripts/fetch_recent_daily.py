"""Keep the running year's daily cache current, so the board never needs the network.

The baseline builder answers "what is normal here in August". This answers "what
happened here yesterday", for the same cities, into the file the app reads
directly. With both on disk the anomaly board is arithmetic over local data: no
upstream call sits between a page load and a ranking, and a restart cannot leave
the board empty.

Cost is the reason this is a separate job rather than part of the sweep.
Open-Meteo weights its quota by locations x days, so one day for every city in
the index is ~6,200 location-days -- about six cities' worth of the three-year
baseline fetch, which is to say nothing. The one-time backfill to January is the
only expensive part, and --limit-days exists to spread it over several runs so it
never starves the baseline fetch of a day's allowance.

Two endpoints, because neither alone can cover the range. ERA5 reanalysis lags
about five days, so the archive cannot reach the recent past; the forecast
endpoint serves up to 92 previous days but no further back. The split point is
handled per request rather than per city, so a city being backfilled from
January and a city needing only yesterday cost one request each.

Cities without a baseline are skipped: a reading that cannot be scored is quota
spent for nothing. Coverage is read from the artefact, so the set widens by
itself as the baseline fetch progresses.

    # daily top-up, the normal case
    python scripts/fetch_recent_daily.py

    # backfill a slice of the year, leaving the rest of the allowance alone
    python scripts/fetch_recent_daily.py --since 2026-01-01 --limit-days 60

Data: Open-Meteo (ERA5 archive + forecast), CC BY 4.0 -- see app/data/ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.city_index import CityRecord, city_records  # noqa: E402
from app.services.normals import NormalsUnavailableError, load_normals  # noqa: E402
from app.services.recent import RECENT_PATH, Run, parse_runs, record_line  # noqa: E402
from scripts.open_meteo_fetch import (  # noqa: E402
    ARCHIVE_URL,
    FORECAST_URL,
    DailyQuotaExhausted,
    Pacer,
    fetch_batch,
    split_to_limits,
)

# ERA5 is published on a delay, so the last few days are simply absent from the
# archive. Asking for them wastes a request and returns nulls that would be
# cached as gaps; the forecast endpoint covers that window instead.
ARCHIVE_LAG_DAYS = 6

# How far back the forecast endpoint will serve previous days.
MAX_PAST_DAYS = 92

# A day of readings per city, in the pacer's units. The pacer thinks in
# location-years because that is what the baseline fetch spends.
DAYS_PER_YEAR = 365.0

# Above this, a pacing pause gets a line of its own. Below it, the gap is too
# short to mistake for trouble and a second line per batch would only double
# the log.
PACING_ANNOUNCE_SECONDS = 30.0


def _daily_params() -> dict[str, str]:
    return {
        "daily": "temperature_2m_mean,relative_humidity_2m_mean",
        # Local days, so a reading is bucketed into the day it happened in
        # locally rather than in UTC -- near the dateline the two disagree.
        "timezone": "auto",
    }


def _coords(batch: Sequence[CityRecord]) -> dict[str, str]:
    return {
        "latitude": ",".join(str(city.latitude) for city in batch),
        "longitude": ",".join(str(city.longitude) for city in batch),
    }


def archive_url(batch: Sequence[CityRecord], start: date, end: date) -> str:
    query = urllib.parse.urlencode(
        {
            **_coords(batch),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            **_daily_params(),
        }
    )
    return f"{ARCHIVE_URL}?{query}"


def forecast_url(batch: Sequence[CityRecord], past_days: int) -> str:
    """Previous ``past_days`` days, ending yesterday.

    ``forecast_days=0`` keeps today out of the response entirely. Today's daily
    mean is a partial day, and caching it would mean the board scored a
    part-day mean against a whole-day baseline -- the error that manufactures
    multi-sigma anomalies tracking the clock rather than the weather.
    """
    query = urllib.parse.urlencode(
        {
            **_coords(batch),
            "past_days": str(past_days),
            "forecast_days": "0",
            **_daily_params(),
        }
    )
    return f"{FORECAST_URL}?{query}"


def plan_windows(start: date, end: date, today: date) -> list[tuple[str, date, date]]:
    """Split a wanted range into the endpoint that can actually serve each part.

    Returns ``(endpoint, start, end)`` segments in chronological order. A range
    entirely inside the last 92 days is one forecast call; a range reaching back
    to January is an archive call plus a forecast call for the tail the archive
    has not caught up with yet.
    """
    if start > end:
        return []

    archive_end = min(end, today - timedelta(days=ARCHIVE_LAG_DAYS))
    forecast_floor = today - timedelta(days=MAX_PAST_DAYS)

    segments: list[tuple[str, date, date]] = []
    if start <= archive_end:
        segments.append(("archive", start, archive_end))

    tail_start = max(start, archive_end + timedelta(days=1), forecast_floor)
    if tail_start <= end:
        segments.append(("forecast", tail_start, end))
    return segments


def _select_cities(min_population: int, only_covered: bool) -> list[CityRecord]:
    """Cities worth spending quota on: big enough to rank, and scoreable.

    A city with no baseline cannot be scored however current its reading is, so
    fetching one is quota spent for nothing. Coverage comes from the artefact,
    which the daily job repacks first, so this set widens on its own as the
    baseline fetch progresses.
    """
    cities = [city for city in city_records() if city.population >= min_population]
    if not only_covered:
        return cities

    try:
        normals = load_normals()
    except NormalsUnavailableError as exc:
        print(f"  no usable normals artefact ({exc}); covering every selected city", flush=True)
        return cities

    covered = set(normals.covered_row_indices)
    return [city for city in cities if city.row_index in covered]


def _recency_gaps(
    cities: list[CityRecord],
    runs: dict[int, Run],
    target_end: date,
    recency_days: int,
) -> dict[tuple[date, date], list[CityRecord]]:
    """Days needed to bring every city up to ``target_end``.

    This phase runs before any backfill, and the order is the whole point. The
    board scores each city's most recent complete day, so a city filled
    oldest-first would sit on the board with a reading from months ago while
    looking exactly as current as its neighbours -- one date per board hides it.
    Recency first means every city is either current or absent, never quietly
    stale.
    """
    floor = target_end - timedelta(days=recency_days - 1)
    groups: dict[tuple[date, date], list[CityRecord]] = {}
    for city in cities:
        run = runs.get(city.geonameid)
        start = max(run.end + timedelta(days=1), floor) if run else floor
        if start > target_end:
            continue
        groups.setdefault((start, target_end), []).append(city)
    return groups


def _backfill_gaps(
    cities: list[CityRecord],
    runs: dict[int, Run],
    since: date,
    limit_days: int | None,
) -> dict[tuple[date, date], list[CityRecord]]:
    """History still missing *behind* each city's run, newest slice first.

    Extending backwards rather than forwards is what lets a long backfill be
    spread over several days without ever costing the board its currency: the
    recent end is already there, and this only deepens the history behind it.
    """
    groups: dict[tuple[date, date], list[CityRecord]] = {}
    for city in cities:
        run = runs.get(city.geonameid)
        if run is None:
            continue  # nothing to extend yet; the recency phase seeds it
        end = run.start - timedelta(days=1)
        if end < since:
            continue
        start = since
        if limit_days is not None:
            start = max(start, end - timedelta(days=limit_days - 1))
        groups.setdefault((start, end), []).append(city)
    return groups


def _readings_by_date(daily: dict) -> dict[date, tuple[float | None, float | None]]:
    """One city's response as a date-keyed map.

    Keyed by the date the server labelled each value with, rather than by
    position: the two endpoints return different spans, and a positional join
    across a boundary would shift every reading onto the wrong day -- wrong in a
    way nothing downstream could detect.
    """
    times = daily.get("time") or []
    temps = daily.get("temperature_2m_mean") or []
    humidities = daily.get("relative_humidity_2m_mean") or []

    out: dict[date, tuple[float | None, float | None]] = {}
    for index, stamp in enumerate(times):
        try:
            day = date.fromisoformat(stamp)
        except (ValueError, TypeError):
            continue
        temperature = temps[index] if index < len(temps) else None
        humidity = humidities[index] if index < len(humidities) else None
        out[day] = (
            float(temperature) if temperature is not None else None,
            float(humidity) if humidity is not None else None,
        )
    return out


def _merge(run: Run | None, fetched: dict[date, tuple]) -> Run:
    """Combine a stored run with newly fetched days, filling any gap with None.

    Bounds come from the data rather than from the requested window, because the
    two phases grow the run from opposite ends: the recency phase extends it
    forward, the backfill extends it backward. Taking the union keeps either
    from truncating what the other added.

    Gaps are written rather than skipped so the run stays contiguous and every
    reading keeps its date. A missing day is a missing day; it must not slide a
    later reading into an earlier slot.
    """
    existing: dict[date, tuple] = {}
    if run:
        for offset, (temperature, humidity) in enumerate(
            zip(run.temps, run.humidities, strict=False)
        ):
            existing[run.start + timedelta(days=offset)] = (temperature, humidity)
    existing.update(fetched)
    if not existing:
        return run or Run(start=date.today(), temps=[], humidities=[])

    start, end = min(existing), max(existing)
    temps: list[float | None] = []
    humidities: list[float | None] = []
    day = start
    while day <= end:
        temperature, humidity = existing.get(day, (None, None))
        temps.append(temperature)
        humidities.append(humidity)
        day += timedelta(days=1)
    return Run(start=start, temps=temps, humidities=humidities)


def _compact(path: Path, runs: dict[int, Run]) -> None:
    """Rewrite the file as one line per city.

    The format is append-and-last-wins, which is what makes a daily top-up cheap,
    but left alone the file would grow by a full copy of every city every day.
    Rewriting through a temporary file keeps a crash mid-write from destroying
    the cache.
    """
    temporary = path.with_suffix(".compacting")
    with temporary.open("w", encoding="utf-8") as handle:
        for geonameid, run in runs.items():
            handle.write(record_line(geonameid, run.start, run.temps, run.humidities) + "\n")
    temporary.replace(path)


def _work_units(groups: dict[tuple[date, date], list[CityRecord]], today: date) -> int:
    """How many city-fetches a phase's groups amount to.

    A range straddling the archive/forecast seam is two requests for the same
    city, and progress ticks once per request as it lands. So the total has to
    count requests too: counting cities instead made a healthy run report
    ``7,620/3,810`` and look like it had lost track of its own workload.
    """
    return sum(
        len(cities) * len(plan_windows(start, end, today))
        for (start, end), cities in groups.items()
    )


def _run_phase(
    label: str,
    groups: dict[tuple[date, date], list[CityRecord]],
    runs: dict[int, Run],
    today: date,
    pacer: Pacer,
    timeout: float,
    handle,
) -> tuple[int, bool]:
    """Fetch one phase's groups into ``runs``, appending each batch as it lands.

    Returns ``(city_days_cached, exhausted)``. Exhaustion is returned rather
    than raised so the caller can still finish tidily: a run that stops halfway
    has done real work, and discarding it would mean paying for those days
    again tomorrow.

    Records are appended per batch rather than written once at the end. This job
    can sit in backoff for a long time behind the baseline fetch, and a run
    killed at minute nine having written nothing would have spent real quota for
    nothing. Append-and-last-wins is what makes that safe -- a later record for
    a city simply supersedes an earlier one, and the compaction at the end
    collapses the duplicates.
    """
    cities_pending = sum(len(group) for group in groups.values())
    if not cities_pending:
        print(f"{label}: nothing to fetch", flush=True)
        return 0, False

    # Two totals because they answer different questions: how much of the index
    # this phase touches, and how many requests that costs.
    pending = _work_units(groups, today)
    print(f"{label}: {cities_pending:,} cities, {pending:,} fetches", flush=True)
    started = time.time()
    done = 0
    cached_days = 0

    for (start, end), group_cities in groups.items():
        for endpoint, segment_start, segment_end in plan_windows(start, end, today):
            days = (segment_end - segment_start).days + 1
            past_days = (today - segment_start).days

            def make_url(
                batch: Sequence[CityRecord],
                _endpoint=endpoint,
                _start=segment_start,
                _end=segment_end,
                _past=past_days,
            ) -> str:
                if _endpoint == "archive":
                    return archive_url(batch, _start, _end)
                return forecast_url(batch, _past)

            for batch in split_to_limits(group_cities, days, make_url):
                batch_started = time.time()
                try:
                    results, throttled = fetch_batch(batch, make_url, timeout=timeout)
                except DailyQuotaExhausted as exc:
                    print(f"\n  daily quota exhausted: {exc}", flush=True)
                    print("  stopping here; re-run after it resets to continue", flush=True)
                    return cached_days, True

                if throttled:
                    pacer.on_throttled()
                else:
                    pacer.on_success()

                if results is None:
                    continue

                # Cities are matched to results by position, so a short or long
                # response would shift every city onto its neighbour's weather.
                if len(results) != len(batch):
                    print(
                        f"    length mismatch: sent {len(batch)}, got {len(results)}"
                        " -- batch dropped",
                        flush=True,
                    )
                    continue

                for city, result in zip(batch, results, strict=True):
                    daily = result.get("daily") if isinstance(result, dict) else None
                    if not daily:
                        continue
                    readings = _readings_by_date(daily)
                    if not readings:
                        continue
                    run = _merge(runs.get(city.geonameid), readings)
                    runs[city.geonameid] = run
                    handle.write(
                        record_line(city.geonameid, run.start, run.temps, run.humidities) + "\n"
                    )
                    cached_days += len(readings)
                handle.flush()

                done += len(batch)
                elapsed = time.time() - started
                print(
                    f"  {done:>6,}/{pending:,} fetches  "
                    f"{endpoint:<8} {segment_start}..{segment_end}  "
                    f"{done / elapsed * 60 if elapsed else 0:>5.0f} fetches/min",
                    flush=True,
                )

                remaining = pacer.seconds_between(len(batch) * days / DAYS_PER_YEAR) - (
                    time.time() - batch_started
                )
                if remaining > 0:
                    # This gap is the pacer's doing, not the server's. Once
                    # throttling has driven the rate to its floor the wait runs
                    # to tens of minutes, and an unexplained silence that long
                    # reads as a hung process -- it has twice sent someone
                    # through netstat to find nothing wrong.
                    if remaining >= PACING_ANNOUNCE_SECONDS:
                        print(
                            f"    pacing, sleeping {remaining:.0f}s "
                            f"({pacer.rate:.1f} location-years/min)",
                            flush=True,
                        )
                    time.sleep(remaining)

    return cached_days, False


def fetch(
    since: date,
    min_population: int,
    only_covered: bool,
    limit_days: int | None,
    recency_days: int,
    timeout: float,
    rate: float,
) -> None:
    today = datetime.now(UTC).date()
    # Yesterday, never today: today's daily mean is a partial day.
    target_end = today - timedelta(days=1)

    cities = _select_cities(min_population, only_covered)
    RECENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    runs: dict[int, Run] = {}
    if RECENT_PATH.exists():
        runs = parse_runs(RECENT_PATH.read_text(encoding="utf-8").splitlines())

    print(f"cities={len(cities):,} through {target_end}", flush=True)
    pacer = Pacer(rate)

    with RECENT_PATH.open("a", encoding="utf-8") as handle:
        # Recency before history, always. The board scores each city's most recent
        # complete day, so a city whose history is deep but whose tail is months old
        # would sit on the board looking exactly as current as the rest.
        cached_days, exhausted = _run_phase(
            "recent days",
            _recency_gaps(cities, runs, target_end, recency_days),
            runs,
            today,
            pacer,
            timeout,
            handle,
        )

        if not exhausted:
            backfilled, exhausted = _run_phase(
                "backfill",
                _backfill_gaps(cities, runs, since, limit_days),
                runs,
                today,
                pacer,
                timeout,
                handle,
            )
            cached_days += backfilled

    if not cached_days:
        # Same reasoning as the baseline builder: a run that got nothing must
        # not rewrite the file it reads from.
        print("\nno new days cached; leaving the recent cache as it is", flush=True)
        return

    _compact(RECENT_PATH, runs)
    scoreable = sum(1 for run in runs.values() if any(t is not None for t in run.temps))
    current = sum(1 for run in runs.values() if run.end >= target_end)
    print(f"\ncached {cached_days:,} new city-days")
    print(f"  {RECENT_PATH} -> {len(runs):,} cities, {scoreable:,} with a reading", flush=True)
    print(f"  {current:,} current through {target_end}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        default=date(datetime.now(UTC).year, 1, 1),
        help="first day to cache for a city with no record yet (default: Jan 1 this year)",
    )
    parser.add_argument(
        "--min-population",
        type=int,
        default=100_000,
        help="skip cities smaller than this (default: 100,000, matching the baseline build)",
    )
    parser.add_argument(
        "--all-cities",
        action="store_true",
        help=(
            "fetch cities with no baseline too. Off by default: their readings "
            "cannot be scored, so the quota buys nothing"
        ),
    )
    parser.add_argument(
        "--limit-days",
        type=int,
        default=None,
        help=(
            "cap the days fetched per city in this run, so a long backfill can be "
            "spread over several days without starving the baseline fetch"
        ),
    )
    parser.add_argument(
        "--recency-days",
        type=int,
        default=7,
        help=(
            "how far back the recency phase will reach to bring a city up to "
            "yesterday (default: 7). Bounds the cost of catching up after a gap"
        ),
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="per-request timeout seconds")
    parser.add_argument(
        "--rate",
        type=float,
        default=60.0,
        help="starting pace in location-years/min; adapts from there",
    )
    args = parser.parse_args()

    fetch(
        args.since,
        args.min_population,
        not args.all_cities,
        args.limit_days,
        args.recency_days,
        args.timeout,
        args.rate,
    )


if __name__ == "__main__":
    main()
