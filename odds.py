"""The Odds API: plan, discover, fetch, and audit point-in-time betting odds.

This is the only thing in the project that costs money (paid API credits). Fetches
are dry-run by default and cached under data/odds/raw/, so re-runs are free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
import pandas as pd

from data import load_results
from features import ODDS_FEATURES

API_ROOT = "https://api.the-odds-api.com/v4"

DATA_ROOT = Path("data/odds")
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
UNMATCHED_PATH = DATA_ROOT / "unmatched.csv"
FEATURES_PATH = DATA_ROOT / "features.csv"

# Deliberate equivalences, not fuzzy matches. Personal additions go in
# data/team_aliases.local.json (Git-ignored). Used by the HISTORICAL path
# (results-source name -> the API name used in that archive).
DEFAULT_TEAM_ALIASES = {
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "United States": "USA",
}

# Used by the UPCOMING path: current live-feed team name -> results.csv history
# name, for the cases where they differ (so fixtures inherit the right Elo state).
FEED_TO_RESULTS = {
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
}


@dataclass(frozen=True)
class Tournament:
    """A configured tournament archive: data-source name and API sport key."""

    name: str
    sport_key: str
    tournament: str
    start: str
    end: str


TOURNAMENTS = (
    # FIFA World Cup
    Tournament("world-cup-2022", "soccer_fifa_world_cup", "FIFA World Cup", "2022-11-20", "2022-12-18"),
    Tournament("world-cup-2026-group", "soccer_fifa_world_cup", "FIFA World Cup", "2026-06-11", "2026-06-28"),
    # UEFA Euro
    Tournament("euro-2020", "soccer_uefa_european_championship", "UEFA Euro", "2021-06-11", "2021-07-12"),
    Tournament("euro-2024", "soccer_uefa_european_championship", "UEFA Euro", "2024-06-14", "2024-07-14"),
    # Copa América
    Tournament("copa-america-2021", "soccer_conmebol_copa_america", "Copa América", "2021-06-13", "2021-07-11"),
    Tournament("copa-america-2024", "soccer_conmebol_copa_america", "Copa América", "2024-06-20", "2024-07-14"),
    # Africa Cup of Nations
    Tournament("afcon-2021", "soccer_africa_cup_of_nations", "African Cup of Nations", "2022-01-09", "2022-02-07"),
    Tournament("afcon-2023", "soccer_africa_cup_of_nations", "African Cup of Nations", "2024-01-13", "2024-02-11"),
    # CONCACAF Gold Cup
    Tournament("gold-cup-2021", "soccer_concacaf_gold_cup", "Gold Cup", "2021-07-10", "2021-08-02"),
    Tournament("gold-cup-2023", "soccer_concacaf_gold_cup", "Gold Cup", "2023-06-24", "2023-07-16"),
    Tournament("gold-cup-2025", "soccer_concacaf_gold_cup", "Gold Cup", "2025-06-14", "2025-07-07"),
    # UEFA Nations League finals
    Tournament("nations-league-finals-2023", "soccer_uefa_nations_league", "UEFA Nations League", "2023-06-14", "2023-06-18"),
    Tournament("nations-league-finals-2025", "soccer_uefa_nations_league", "UEFA Nations League", "2025-06-04", "2025-06-08"),
)


def selected_matches(tournaments: tuple[Tournament, ...] = TOURNAMENTS) -> dict[Tournament, pd.DataFrame]:
    """Return the played matches in each configured tournament archive."""
    matches = load_results()
    selections: dict[Tournament, pd.DataFrame] = {}
    for tournament in tournaments:
        mask = (
            matches["tournament"].eq(tournament.tournament)
            & matches["date"].between(tournament.start, tournament.end)
            & matches["outcome"].notna()
        )
        selections[tournament] = matches.loc[mask, ["date", "home_team", "away_team", "tournament"]]
    return selections


class OddsAPIError(RuntimeError):
    """An API error whose message never includes the secret key."""


@dataclass(frozen=True)
class Quota:
    """Quota headers returned by The Odds API."""

    remaining: int | None
    used: int | None
    last: int | None


@dataclass(frozen=True)
class APIResponse:
    """Decoded API response and sanitized usage metadata."""

    body: Any
    quota: Quota


class OddsAPIClient:
    """Small stdlib client that keeps credentials out of files and logs."""

    def __init__(self, api_key: str, timeout: int = 30):
        if not api_key.strip():
            raise ValueError("The Odds API key is empty")
        self._api_key = api_key.strip()
        self.timeout = timeout

    def get(self, path: str, **parameters: object) -> APIResponse:
        """GET and decode JSON without exposing the authenticated URL."""
        api_path = path.lstrip("/")
        path_parts = api_path.split("/")
        if (
            not api_path
            or any(part in {"", ".", ".."} for part in path_parts)
            or not re.fullmatch(r"[A-Za-z0-9_./-]+", api_path)
        ):
            raise ValueError("Unsafe API path")
        query = {key: value for key, value in parameters.items() if value is not None}
        query["apiKey"] = self._api_key
        url = f"{API_ROOT}/{api_path}?{urllib.parse.urlencode(query)}"
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != "api.the-odds-api.com":
            raise ValueError("API URL must use the configured HTTPS endpoint")
        request = urllib.request.Request(
            url, headers={"Accept": "application/json", "User-Agent": "football-backtest/1"}
        )
        try:
            # The scheme and host are fixed and validated immediately above.
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # nosec B310
                return APIResponse(body=json.loads(response.read()), quota=_quota(response.headers))
        except urllib.error.HTTPError as error:
            quota = _quota(error.headers)
            try:
                payload = json.loads(error.read())
                message = payload.get("message") or payload.get("error")
            except (json.JSONDecodeError, AttributeError):
                message = None
            safe_message = (
                str(message).replace(self._api_key, "[REDACTED]")
                if message
                else ""
            )
            detail = f": {safe_message}" if safe_message else ""
            raise OddsAPIError(
                f"The Odds API returned HTTP {error.code}{detail}; "
                f"last cost={quota.last}, remaining={quota.remaining}"
            ) from None
        except urllib.error.URLError as error:
            reason = str(error.reason).replace(self._api_key, "[REDACTED]")
            raise OddsAPIError(f"The Odds API request failed: {reason}") from None


def load_api_key(
    *,
    env_var: str | None = None,
    keychain_service: str | None = None,
    keychain_account: str | None = None,
) -> str:
    """Load a key from an explicitly named environment variable or Keychain item."""
    if env_var:
        value = os.environ.get(env_var)
        if value and value.strip():
            return value.strip()
    if bool(keychain_service) != bool(keychain_account):
        raise ValueError("Keychain service and account must be configured together")
    if keychain_service and keychain_account:
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-s",
                keychain_service,
                "-a",
                keychain_account,
                "-w",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if not result.returncode and result.stdout.strip():
            return result.stdout.strip()
    raise RuntimeError(
        "No API credential was found. Configure --api-key-env NAME or both "
        "--keychain-service and --keychain-account."
    )


def load_aliases(path: str | Path | None = None) -> dict[str, str]:
    """Load explicit result-name to API-name mappings."""
    aliases = dict(DEFAULT_TEAM_ALIASES)
    if path and Path(path).exists():
        with Path(path).open(encoding="utf-8") as handle:
            additions = json.load(handle)
        if not isinstance(additions, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in additions.items()
        ):
            raise ValueError("Team alias file must contain a JSON string-to-string object")
        aliases.update(additions)
    return aliases


def api_team(name: str, aliases: dict[str, str]) -> str:
    """Map a data-source team name to its explicit API equivalent."""
    return aliases.get(name, name)


def discover_events(
    *,
    client: OddsAPIClient,
    matches: pd.DataFrame,
    sport_key: str,
    aliases: dict[str, str],
    raw_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discover historical event IDs without spending quota where possible."""
    required = {"date", "home_team", "away_team"}
    missing = required.difference(matches.columns)
    if missing:
        raise ValueError(f"Matches are missing columns: {', '.join(sorted(missing))}")

    targets = matches.copy()
    targets["date"] = pd.to_datetime(targets["date"])
    targets["api_home"] = targets["home_team"].map(lambda value: api_team(value, aliases))
    targets["api_away"] = targets["away_team"].map(lambda value: api_team(value, aliases))
    discovered: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []
    cache: dict[str, list[dict[str, Any]]] = {}

    for row in targets.itertuples():
        match_day = pd.Timestamp(row.date).normalize().tz_localize("UTC")
        probes = (
            match_day - pd.Timedelta(hours=36),
            match_day - pd.Timedelta(hours=30),
            match_day - pd.Timedelta(hours=24),
            match_day - pd.Timedelta(hours=18),
        )
        candidates_by_id: dict[str, tuple[dict[str, Any], str]] = {}
        for probe in probes:
            probe_iso = _iso(probe)
            if probe_iso not in cache:
                parameters = {"date": probe_iso, "dateFormat": "iso"}
                response = load_raw_response(
                    raw_dir, kind="events", sport_key=sport_key, timestamp=probe_iso, parameters=parameters
                )
                if response is None:
                    response = client.get(f"historical/sports/{sport_key}/events", **parameters)
                    write_raw_once(
                        raw_dir,
                        kind="events",
                        sport_key=sport_key,
                        timestamp=probe_iso,
                        parameters=parameters,
                        body=response.body,
                        quota=response.quota,
                    )
                cache[probe_iso] = _response_data(response.body)
            for event in cache[probe_iso]:
                direct = event.get("home_team") == row.api_home and event.get("away_team") == row.api_away
                reversed_orientation = (
                    event.get("home_team") == row.api_away and event.get("away_team") == row.api_home
                )
                close_date = (
                    abs((pd.Timestamp(event["commence_time"]).date() - pd.Timestamp(row.date).date()).days) <= 1
                )
                if close_date and (direct or reversed_orientation):
                    candidates_by_id[event["id"]] = (event, "direct" if direct else "reversed")
            if len(candidates_by_id) == 1:
                break
        candidates = list(candidates_by_id.values())
        base = {
            "source_index": row.Index,
            "date": pd.Timestamp(row.date).date().isoformat(),
            "home_team": row.home_team,
            "away_team": row.away_team,
            "sport_key": sport_key,
        }
        if len(candidates) == 1:
            event, orientation = candidates[0]
            commence = pd.Timestamp(event["commence_time"])
            discovered.append(
                {
                    **base,
                    "event_id": event["id"],
                    "api_home_team": event["home_team"],
                    "api_away_team": event["away_team"],
                    "orientation": orientation,
                    "commence_time": _iso(commence),
                    "prediction_cutoff": _iso(commence - pd.Timedelta(hours=24)),
                }
            )
        else:
            unmatched.append(
                {
                    **base,
                    "reason": "no exact candidate" if not candidates else f"{len(candidates)} exact candidates",
                }
            )
    return pd.DataFrame(discovered), pd.DataFrame(unmatched)


def estimate_historical_credits(manifest: pd.DataFrame, *, markets: Iterable[str], regions: Iterable[str]) -> int:
    """Return a conservative upper bound, grouping simultaneous snapshots."""
    required = {"sport_key", "prediction_cutoff"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    requests = manifest.loc[:, list(required)].drop_duplicates().shape[0]
    return requests * 10 * len(tuple(markets)) * len(tuple(regions))


def fetch_historical_odds(
    *,
    client: OddsAPIClient,
    manifest: pd.DataFrame,
    markets: tuple[str, ...],
    regions: tuple[str, ...],
    raw_dir: str | Path,
    max_credits: int,
) -> tuple[pd.DataFrame, dict[str, int | None]]:
    """Fetch planned snapshots while enforcing a hard cumulative credit ceiling."""
    estimated = estimate_historical_credits(manifest, markets=markets, regions=regions)
    if estimated > max_credits:
        raise ValueError(f"Estimated upper bound {estimated} exceeds --max-credits {max_credits}")

    rows: list[dict[str, object]] = []
    spent = 0
    remaining: int | None = None
    grouped = manifest.groupby(["sport_key", "prediction_cutoff"], sort=True, dropna=False)
    for (sport_key, cutoff), group in grouped:
        if spent + 10 * len(markets) * len(regions) > max_credits:
            raise RuntimeError("Credit ceiling reached before the next API request")
        parameters = {
            "date": cutoff,
            "regions": list(regions),
            "markets": list(markets),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        response = load_raw_response(
            raw_dir, kind="odds", sport_key=str(sport_key), timestamp=str(cutoff), parameters=parameters
        )
        cache_hit = response is not None
        if response is None:
            response = client.get(
                f"historical/sports/{sport_key}/odds",
                date=cutoff,
                regions=",".join(regions),
                markets=",".join(markets),
                oddsFormat="decimal",
                dateFormat="iso",
            )
            write_raw_once(
                raw_dir,
                kind="odds",
                sport_key=str(sport_key),
                timestamp=str(cutoff),
                parameters=parameters,
                body=response.body,
                quota=response.quota,
            )
        cost = 0 if cache_hit else (response.quota.last or 0)
        spent += cost
        if not cache_hit:
            remaining = response.quota.remaining
        events = {event["id"]: event for event in _response_data(response.body)}
        observed_at = _response_timestamp(response.body, fallback=str(cutoff))
        for target in group.itertuples():
            event = events.get(target.event_id)
            if event is None:
                continue
            if target.orientation == "direct":
                target_api_home, target_api_away = target.api_home_team, target.api_away_team
            elif target.orientation == "reversed":
                target_api_home, target_api_away = target.api_away_team, target.api_home_team
            else:
                raise ValueError(f"Unknown orientation {target.orientation!r} for event {target.event_id}")
            direct_at_snapshot = (
                event.get("home_team") == target_api_home and event.get("away_team") == target_api_away
            )
            reversed_at_snapshot = (
                event.get("home_team") == target_api_away and event.get("away_team") == target_api_home
            )
            if not (direct_at_snapshot or reversed_at_snapshot):
                raise ValueError(
                    f"Team-pair mismatch for event {target.event_id}: "
                    f"expected {target_api_home} vs {target_api_away}, got "
                    f"{event.get('home_team')} vs {event.get('away_team')}"
                )
            consensus = consensus_h2h(event)
            if consensus is None:
                continue
            if reversed_at_snapshot:
                consensus["market_p_home"], consensus["market_p_away"] = (
                    consensus["market_p_away"],
                    consensus["market_p_home"],
                )
                consensus["market_home_std"], consensus["market_away_std"] = (
                    consensus["market_away_std"],
                    consensus["market_home_std"],
                )
            rows.append(
                {
                    "date": target.date,
                    "home_team": target.home_team,
                    "away_team": target.away_team,
                    "event_id": target.event_id,
                    "sport_key": sport_key,
                    "commence_time": target.commence_time,
                    "submission_cutoff": target.prediction_cutoff,
                    "observed_at": observed_at,
                    **consensus,
                }
            )
    return pd.DataFrame(rows), {
        "estimated_upper_bound": estimated,
        "actual_spent": spent,
        "remaining": remaining,
    }


def fetch_upcoming_odds(
    *,
    client: OddsAPIClient,
    sport_key: str,
    markets: tuple[str, ...],
    regions: tuple[str, ...],
    raw_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch CURRENT odds for ALL upcoming events of a sport and de-vig them.

    Uses the live ``sports/{key}/odds`` endpoint (~1 credit/region/market), not the
    10x historical one. The feed itself is the authoritative fixture list, so every
    event with a usable consensus is written (team names mapped back to data-source
    names via the reverse alias map).
    """
    parameters = {
        "regions": list(regions),
        "markets": list(markets),
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    # Cache at hour granularity so same-hour reruns do not re-spend.
    stamp = _iso(pd.Timestamp.now(tz="UTC"))[:13]
    response = load_raw_response(
        raw_dir, kind="upcoming", sport_key=sport_key, timestamp=stamp, parameters=parameters
    )
    if response is None:
        response = client.get(
            f"sports/{sport_key}/odds",
            regions=",".join(regions),
            markets=",".join(markets),
            oddsFormat="decimal",
            dateFormat="iso",
        )
        write_raw_once(
            raw_dir,
            kind="upcoming",
            sport_key=sport_key,
            timestamp=stamp,
            parameters=parameters,
            body=response.body,
            quota=response.quota,
        )
    observed_at = _iso(pd.Timestamp.now(tz="UTC"))
    events = _response_data(response.body)

    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for event in events:
        consensus = consensus_h2h(event)
        if consensus is None:
            skipped.append(f"{event.get('home_team')} vs {event.get('away_team')} (no h2h)")
            continue
        home = FEED_TO_RESULTS.get(event["home_team"], event["home_team"])
        away = FEED_TO_RESULTS.get(event["away_team"], event["away_team"])
        commence = _iso(pd.Timestamp(event["commence_time"]))
        rows.append(
            {
                "date": pd.Timestamp(event["commence_time"]).date().isoformat(),
                "home_team": home,
                "away_team": away,
                "event_id": event["id"],
                "sport_key": sport_key,
                "commence_time": commence,
                "submission_cutoff": commence,  # observed now (before kickoff) -> passes audit
                "observed_at": observed_at,
                **consensus,
            }
        )
    info = {
        "events": int(len(events)),
        "written": int(len(rows)),
        "skipped": skipped,
        "remaining": response.quota.remaining,
    }
    return pd.DataFrame(rows), info


def consensus_h2h(event: dict[str, Any]) -> dict[str, float | int] | None:
    """Create a robust median consensus after de-vigging each bookmaker."""
    home = event.get("home_team")
    away = event.get("away_team")
    per_book: list[tuple[float, float, float, float]] = []
    for bookmaker in event.get("bookmakers", []):
        markets = [m for m in bookmaker.get("markets", []) if m.get("key") == "h2h"]
        if len(markets) != 1:
            continue
        prices = {o.get("name"): o.get("price") for o in markets[0].get("outcomes", [])}
        try:
            decimal = [float(prices[home]), float(prices["Draw"]), float(prices[away])]
        except (KeyError, TypeError, ValueError):
            continue
        if any(not np.isfinite(price) or price <= 1.0 for price in decimal):
            continue
        implied = np.reciprocal(decimal)
        overround = float(implied.sum())
        per_book.append((*map(float, implied / overround), overround))
    if not per_book:
        return None

    values = np.asarray(per_book)
    consensus = np.median(values[:, :3], axis=0)
    consensus /= consensus.sum()
    return {
        "market_p_home": float(consensus[0]),
        "market_p_draw": float(consensus[1]),
        "market_p_away": float(consensus[2]),
        "market_home_std": float(values[:, 0].std()),
        "market_draw_std": float(values[:, 1].std()),
        "market_away_std": float(values[:, 2].std()),
        "market_overround": float(median(values[:, 3])),
        "book_count": int(len(values)),
    }


def audit_snapshots(snapshots: pd.DataFrame, manifest: pd.DataFrame | None = None) -> dict[str, object]:
    """Validate probabilities, timestamps, duplicates, and optional coverage."""
    required = {"date", "home_team", "away_team", "submission_cutoff", "observed_at", *ODDS_FEATURES}
    missing = required.difference(snapshots.columns)
    if missing:
        raise ValueError(f"Snapshots are missing columns: {', '.join(sorted(missing))}")
    keys = ["date", "home_team", "away_team"]
    duplicate_count = int(snapshots.duplicated(keys).sum())
    probabilities = snapshots.loc[:, ["market_p_home", "market_p_draw", "market_p_away"]].astype(float)
    invalid_probability_rows = int(
        (
            (~np.isfinite(probabilities)).any(axis=1)
            | (probabilities <= 0).any(axis=1)
            | (probabilities >= 1).any(axis=1)
            | ~np.isclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
        ).sum()
    )
    observed = pd.to_datetime(snapshots["observed_at"], utc=True)
    cutoff = pd.to_datetime(snapshots["submission_cutoff"], utc=True)
    future_snapshot_count = int((observed > cutoff).sum())
    if duplicate_count or invalid_probability_rows or future_snapshot_count:
        raise ValueError(
            "Odds audit failed: "
            f"duplicates={duplicate_count}, "
            f"invalid_probabilities={invalid_probability_rows}, "
            f"future_snapshots={future_snapshot_count}"
        )
    coverage: float | None = None
    if manifest is not None and len(manifest):
        covered = manifest.loc[:, keys].merge(
            snapshots.loc[:, keys].drop_duplicates(), on=keys, how="left", indicator=True
        )
        coverage = float(covered["_merge"].eq("both").mean())
    return {
        "rows": int(len(snapshots)),
        "coverage": coverage,
        "median_book_count": float(snapshots["book_count"].median()) if len(snapshots) else None,
        "max_snapshot_age_hours": (
            float(((cutoff - observed).dt.total_seconds() / 3600).max()) if len(snapshots) else None
        ),
    }


def write_raw_once(
    root: str | Path, *, kind: str, sport_key: str, timestamp: str, parameters: dict[str, object], body: Any, quota: Quota
) -> Path:
    """Write an immutable response envelope, never including the API key."""
    identity = _request_identity(kind=kind, sport_key=sport_key, timestamp=timestamp, parameters=parameters)
    path = _raw_path(root, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "request": json.loads(identity),
        "quota": {"remaining": quota.remaining, "used": quota.used, "last": quota.last},
        "response": body,
    }
    encoded = json.dumps(envelope, indent=2, sort_keys=True) + "\n"
    if path.exists():
        # Raw files are append-only evidence; never mutate the first capture.
        return path
    path.write_text(encoded, encoding="utf-8")
    return path


def load_raw_response(
    root: str | Path, *, kind: str, sport_key: str, timestamp: str, parameters: dict[str, object]
) -> APIResponse | None:
    """Return a previously captured response for this exact request identity."""
    identity = _request_identity(kind=kind, sport_key=sport_key, timestamp=timestamp, parameters=parameters)
    path = _raw_path(root, identity)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        envelope = json.load(handle)
    if envelope.get("request") != json.loads(identity):
        raise RuntimeError(f"Raw cache identity does not match its filename: {path}")
    quota = envelope.get("quota", {})
    return APIResponse(
        body=envelope["response"],
        quota=Quota(remaining=_optional_int(quota.get("remaining")), used=_optional_int(quota.get("used")), last=0),
    )


def _request_identity(*, kind: str, sport_key: str, timestamp: str, parameters: dict[str, object]) -> str:
    return json.dumps(
        {"kind": kind, "sport_key": sport_key, "timestamp": timestamp, "parameters": parameters}, sort_keys=True
    )


def _raw_path(root: str | Path, identity: str) -> Path:
    request = json.loads(identity)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    sport_key = _safe_path_component(request["sport_key"], "sport_key")
    kind = _safe_path_component(request["kind"], "kind")
    return Path(root) / "raw" / sport_key / f"{kind}_{digest}.json"


def _safe_path_component(value: object, label: str) -> str:
    text = str(value)
    if (
        text in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", text)
    ):
        raise ValueError(f"Unsafe {label}: expected letters, numbers, '.', '_', or '-'")
    return text


def _optional_int(value: object) -> int | None:
    return int(value) if value not in (None, "") else None


def _quota(headers: Any) -> Quota:
    def integer(name: str) -> int | None:
        value = headers.get(name) if headers else None
        return int(value) if value not in (None, "") else None

    return Quota(
        remaining=integer("x-requests-remaining"),
        used=integer("x-requests-used"),
        last=integer("x-requests-last"),
    )


def _response_data(body: Any) -> list[dict[str, Any]]:
    data = body.get("data", []) if isinstance(body, dict) else body
    if not isinstance(data, list):
        raise OddsAPIError("The Odds API returned an unexpected data shape")
    return data


def _response_timestamp(body: Any, fallback: str) -> str:
    if isinstance(body, dict) and body.get("timestamp"):
        return str(body["timestamp"])
    return fallback


def _iso(value: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--aliases", type=Path, default=Path("data/team_aliases.local.json"))
    credentials = parser.add_argument_group("credentials (network commands only)")
    credentials.add_argument(
        "--api-key-env",
        metavar="NAME",
        help="Read the API key from this environment variable",
    )
    credentials.add_argument("--keychain-service", help="macOS Keychain service name")
    credentials.add_argument("--keychain-account", help="macOS Keychain account name")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Show target matches and credit ceiling")
    _download_options(plan)

    upcoming = subparsers.add_parser(
        "upcoming", help="Fetch CURRENT odds for upcoming fixtures; append to features.csv"
    )
    upcoming.add_argument("--sport-key", default="soccer_fifa_world_cup")
    upcoming.add_argument("--output", type=Path, default=FEATURES_PATH)
    upcoming.add_argument("--markets", default="h2h")
    upcoming.add_argument("--regions", default="eu")
    upcoming.add_argument("--execute", action="store_true", help="Make the paid request (dry-run by default)")

    discover = subparsers.add_parser("discover", help="Discover exact historical event IDs (quota-light)")
    discover.add_argument("--output", type=Path, default=MANIFEST_PATH)
    discover.add_argument("--unmatched", type=Path, default=UNMATCHED_PATH)

    fetch = subparsers.add_parser("fetch-historical", help="Download kickoff-minus-24h snapshots; dry-run by default")
    fetch.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    fetch.add_argument("--output", type=Path, default=FEATURES_PATH)
    _download_options(fetch)
    fetch.add_argument("--max-credits", type=int, help="Required hard ceiling when --execute is supplied")
    fetch.add_argument("--execute", action="store_true", help="Actually make paid requests")

    audit = subparsers.add_parser("audit", help="Fail on leakage or malformed odds")
    audit.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    audit.add_argument("--features", type=Path, default=FEATURES_PATH)
    return parser.parse_args()


def _download_options(parser: argparse.ArgumentParser) -> None:
    # h2h + one region is all win/draw/loss needs, and ~6x cheaper than the
    # original h2h,spreads,totals x eu,us.
    parser.add_argument("--markets", default="h2h")
    parser.add_argument("--regions", default="eu")


def main() -> None:
    args = parse_args()
    markets = tuple(v.strip() for v in getattr(args, "markets", "").split(",") if v.strip())
    regions = tuple(v.strip() for v in getattr(args, "regions", "").split(",") if v.strip())

    if args.command == "plan":
        selections = selected_matches()
        count = sum(len(frame) for frame in selections.values())
        upper = count * 10 * len(markets) * len(regions)
        for tournament, frame in selections.items():
            print(f"{tournament.name}: {len(frame)} matches")
        print(f"Total: {count} matches")
        print(f"Conservative ceiling: {upper} credits ({len(markets)} markets x {len(regions)} regions)")
        return

    if args.command == "upcoming":
        print(
            f"Planned request: GET sports/{args.sport_key}/odds "
            f"markets={','.join(markets)} regions={','.join(regions)} "
            f"(current odds, ~{len(markets) * len(regions)} credits) -- the feed is the fixture list"
        )
        if not args.execute:
            print("Dry run only. Add --execute to make the paid request.")
            return
        client = OddsAPIClient(
            load_api_key(
                env_var=args.api_key_env,
                keychain_service=args.keychain_service,
                keychain_account=args.keychain_account,
            )
        )
        new_rows, info = fetch_upcoming_odds(
            client=client,
            sport_key=args.sport_key,
            markets=markets,
            regions=regions,
            raw_dir=args.data_root,
        )
        existing = pd.read_csv(args.output) if args.output.exists() else pd.DataFrame()
        combined = pd.concat([existing, new_rows], ignore_index=True) if len(existing) else new_rows
        # Dedup on event_id so a re-fetch replaces prior upcoming rows for the same game.
        combined = combined.drop_duplicates("event_id", keep="last")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.output, index=False)
        print(f"\nWrote {info['written']} upcoming events ({info['events']} in feed) -> {args.output} ({len(combined)} rows)")
        for pair in info["skipped"]:
            print(f"  skipped: {pair}")
        print(f"Credits remaining: {info['remaining']}")
        return

    if args.command == "discover":
        client = OddsAPIClient(
            load_api_key(
                env_var=args.api_key_env,
                keychain_service=args.keychain_service,
                keychain_account=args.keychain_account,
            )
        )
        aliases = load_aliases(args.aliases)
        manifests, unmatched = [], []
        for tournament, matches in selected_matches().items():
            found, missing = discover_events(
                client=client,
                matches=matches,
                sport_key=tournament.sport_key,
                aliases=aliases,
                raw_dir=args.data_root,
            )
            found["archive"] = tournament.name
            missing["archive"] = tournament.name
            manifests.append(found)
            unmatched.append(missing)
            print(f"{tournament.name}: matched {len(found)}/{len(matches)}, unmatched {len(missing)}")
        manifest = pd.concat(manifests, ignore_index=True)
        misses = pd.concat(unmatched, ignore_index=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(args.output, index=False)
        misses.to_csv(args.unmatched, index=False)
        print(f"Manifest: {args.output}")
        print(f"Unmatched report: {args.unmatched}")
        return

    if args.command == "fetch-historical":
        manifest = pd.read_csv(args.manifest)
        estimate = estimate_historical_credits(manifest, markets=markets, regions=regions)
        print(f"Requests: {manifest[['sport_key', 'prediction_cutoff']].drop_duplicates().shape[0]}")
        print(f"Estimated upper bound: {estimate} credits")
        if not args.execute:
            print("Dry run only. Add --execute --max-credits N after reviewing.")
            return
        if args.max_credits is None:
            raise ValueError("--execute requires an explicit --max-credits")
        features, quota = fetch_historical_odds(
            client=OddsAPIClient(
                load_api_key(
                    env_var=args.api_key_env,
                    keychain_service=args.keychain_service,
                    keychain_account=args.keychain_account,
                )
            ),
            manifest=manifest,
            markets=markets,
            regions=regions,
            raw_dir=args.data_root,
            max_credits=args.max_credits,
        )
        audit = audit_snapshots(features, manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(args.output, index=False)
        (args.output.parent / "fetch_report.json").write_text(
            json.dumps({"quota": quota, "audit": audit}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Features: {args.output}")
        print(json.dumps({"quota": quota, "audit": audit}, indent=2))
        return

    manifest = pd.read_csv(args.manifest)
    features = pd.read_csv(args.features)
    print(json.dumps(audit_snapshots(features, manifest), indent=2))


if __name__ == "__main__":
    main()
