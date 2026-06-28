from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


HOME_ADVANTAGE = 65.0
FEATURE_COLUMNS = (
    "elo_diff",
    "home_elo",
    "away_elo",
    "form5_diff",
    "form10_diff",
    "home_form5",
    "away_form5",
    "home_winrate",
    "away_winrate",
    "home_gf5",
    "away_gf5",
    "home_ga5",
    "away_ga5",
    "gd10_diff",
    "home_streak",
    "away_streak",
    "home_rest",
    "away_rest",
    "home_played",
    "away_played",
    "h2h_n",
    "h2h_home_winrate",
    "h2h_draw_rate",
    "h2h_gd",
    "neutral",
    "importance",
)


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "neutral",
        "importance",
    }
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise ValueError(f"Feature input is missing columns: {', '.join(missing)}")
    if not matches["date"].is_monotonic_increasing:
        raise ValueError("Feature input must be sorted chronologically")

    elo = defaultdict(lambda: 1500.0)
    team_results: dict[str, list[tuple[float, float, float, bool]]] = defaultdict(list)
    last_date: dict[str, pd.Timestamp] = {}
    head_to_head: dict[
        tuple[str, str], list[tuple[str, float, str]]
    ] = defaultdict(list)

    def team_state(team: str) -> tuple[float, ...]:
        history = team_results[team]
        if not history:
            return (elo[team], 1.3, 1.3, 0.33, 1.0, 1.0, 0.0, 0.0, 0)
        last_five = history[-5:]
        last_ten = history[-10:]
        streak = 0
        for points, *_ in reversed(history):
            if points != 3:
                break
            streak += 1
        return (
            elo[team],
            float(np.mean([row[0] for row in last_five])),
            float(np.mean([row[0] for row in last_ten])),
            float(np.mean([row[3] for row in last_ten])),
            float(np.mean([row[1] for row in last_five])),
            float(np.mean([row[2] for row in last_five])),
            float(np.mean([row[1] - row[2] for row in last_ten])),
            float(streak),
            float(len(history)),
        )

    def matchup_state(home: str, away: str) -> tuple[float, ...]:
        history = head_to_head[tuple(sorted((home, away)))]
        if not history:
            return 0.0, 0.5, 0.25, 0.0
        count = len(history)
        return (
            float(count),
            sum(winner == home for _, _, winner in history) / count,
            sum(winner == "draw" for _, _, winner in history) / count,
            float(
                np.mean(
                    [
                        goal_difference if historical_home == home else -goal_difference
                        for historical_home, goal_difference, _ in history
                    ]
                )
            ),
        )

    rows: list[dict[str, float]] = []
    for _, same_date in matches.groupby("date", sort=False):
        pending_updates = []
        for match in same_date.itertuples():
            home = match.home_team
            away = match.away_team
            home_advantage = HOME_ADVANTAGE * (1 - match.neutral)
            (
                home_elo,
                home_form_five,
                home_form_ten,
                home_winrate,
                home_goals_for,
                home_goals_against,
                home_goal_difference,
                home_streak,
                home_played,
            ) = team_state(home)
            (
                away_elo,
                away_form_five,
                away_form_ten,
                away_winrate,
                away_goals_for,
                away_goals_against,
                away_goal_difference,
                away_streak,
                away_played,
            ) = team_state(away)
            h2h_count, h2h_home_winrate, h2h_draw_rate, h2h_goal_difference = (
                matchup_state(home, away)
            )
            rows.append(
                {
                    "elo_diff": home_elo + home_advantage - away_elo,
                    "home_elo": home_elo,
                    "away_elo": away_elo,
                    "form5_diff": home_form_five - away_form_five,
                    "form10_diff": home_form_ten - away_form_ten,
                    "home_form5": home_form_five,
                    "away_form5": away_form_five,
                    "home_winrate": home_winrate,
                    "away_winrate": away_winrate,
                    "home_gf5": home_goals_for,
                    "away_gf5": away_goals_for,
                    "home_ga5": home_goals_against,
                    "away_ga5": away_goals_against,
                    "gd10_diff": home_goal_difference - away_goal_difference,
                    "home_streak": home_streak,
                    "away_streak": away_streak,
                    "home_rest": (
                        float(min((match.date - last_date[home]).days, 90))
                        if home in last_date
                        else 30.0
                    ),
                    "away_rest": (
                        float(min((match.date - last_date[away]).days, 90))
                        if away in last_date
                        else 30.0
                    ),
                    "home_played": home_played,
                    "away_played": away_played,
                    "h2h_n": h2h_count,
                    "h2h_home_winrate": h2h_home_winrate,
                    "h2h_draw_rate": h2h_draw_rate,
                    "h2h_gd": h2h_goal_difference,
                }
            )
            pending_updates.append((match, home_elo, away_elo, home_advantage))

        for match, home_elo, away_elo, home_advantage in pending_updates:
            if pd.isna(match.home_score) or pd.isna(match.away_score):
                continue
            home = match.home_team
            away = match.away_team
            goal_difference = float(match.home_score - match.away_score)
            expected_home = 1 / (
                1 + 10 ** ((away_elo - home_elo - home_advantage) / 400)
            )
            actual_home = (
                1.0 if goal_difference > 0 else (0.0 if goal_difference < 0 else 0.5)
            )
            margin = (
                1.0
                if abs(goal_difference) <= 1
                else (
                    1.5
                    if abs(goal_difference) == 2
                    else (11 + abs(goal_difference)) / 8
                )
            )
            change = match.importance * margin * (actual_home - expected_home)
            elo[home] += change
            elo[away] -= change
            team_results[home].append(
                (
                    3.0 if goal_difference > 0 else (1.0 if goal_difference == 0 else 0.0),
                    float(match.home_score),
                    float(match.away_score),
                    goal_difference > 0,
                )
            )
            team_results[away].append(
                (
                    3.0 if goal_difference < 0 else (1.0 if goal_difference == 0 else 0.0),
                    float(match.away_score),
                    float(match.home_score),
                    goal_difference < 0,
                )
            )
            last_date[home] = match.date
            last_date[away] = match.date
            head_to_head[tuple(sorted((home, away)))].append(
                (
                    home,
                    goal_difference,
                    (
                        home
                        if goal_difference > 0
                        else (away if goal_difference < 0 else "draw")
                    ),
                )
            )

    features = pd.DataFrame(rows, index=matches.index)
    return matches.join(features)
