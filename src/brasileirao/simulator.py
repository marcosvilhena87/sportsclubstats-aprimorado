"""Utilities for parsing match fixtures and running Monte Carlo simulations.

This module reads fixtures exported from SportsClubStats and provides
functions to compute league tables and run a SportsClubStats-style model to
project results.  It powers the public functions exposed in
``brasileirao.__init__``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------
SCORE_PATTERN = re.compile(
    r"(\d+/\d+/\d+)\s+(.+?)\s+(\d+)-(\d+)\s+(.+?)\s*(?:\(ID:.*)?$"
)
NOSCORE_PATTERN = re.compile(
    r"(\d+/\d+/\d+)\s+(.+?)\s{2,}(.+?)\s*(?:\(ID:.*)?$"
)


def _parse_date(date_str: str) -> pd.Timestamp:
    parts = date_str.split("/")
    year = parts[-1]
    if len(year) == 4:
        return pd.to_datetime(date_str, format="%d/%m/%Y")
    return pd.to_datetime(date_str, format="%m/%d/%y")


def parse_matches(path: str | Path) -> pd.DataFrame:
    """Return a DataFrame of fixtures and results."""
    rows: list[dict] = []
    in_games = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "GamesBegin":
                in_games = True
                continue
            if line.strip() == "GamesEnd":
                break
            if not in_games:
                continue
            line = line.rstrip("\n")
            m = SCORE_PATTERN.match(line)
            if m:
                date_str, home, hs, as_, away = m.groups()
                rows.append(
                    {
                        "date": _parse_date(date_str),
                        "home_team": home.strip(),
                        "away_team": away.strip(),
                        "home_score": int(hs),
                        "away_score": int(as_),
                    }
                )
                continue
            m = NOSCORE_PATTERN.match(line)
            if m:
                date_str, home, away = m.groups()
                rows.append(
                    {
                        "date": _parse_date(date_str),
                        "home_team": home.strip(),
                        "away_team": away.strip(),
                        "home_score": np.nan,
                        "away_score": np.nan,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table computation
# ---------------------------------------------------------------------------

def _head_to_head_points(matches: pd.DataFrame, teams: list[str]) -> Dict[str, int]:
    points = {t: 0 for t in teams}
    df = matches.dropna(subset=["home_score", "away_score"])
    df = df[df["home_team"].isin(teams) & df["away_team"].isin(teams)]
    for _, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        if hs > as_:
            points[ht] += 3
        elif hs < as_:
            points[at] += 3
        else:
            points[ht] += 1
            points[at] += 1
    return points


def league_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute league standings from played matches."""
    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    table: Dict[str, Dict[str, float]] = {
        t: {"team": t, "played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0}
        for t in teams
    }

    played = matches.dropna(subset=["home_score", "away_score"])
    for _, row in played.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        table[home]["played"] += 1
        table[away]["played"] += 1
        table[home]["gf"] += hs
        table[home]["ga"] += as_
        table[away]["gf"] += as_
        table[away]["ga"] += hs
        if hs > as_:
            table[home]["wins"] += 1
            table[home]["points"] = table[home].get("points", 0) + 3
            table[away]["losses"] += 1
            table[away].setdefault("points", 0)
        elif hs < as_:
            table[away]["wins"] += 1
            table[away]["points"] = table[away].get("points", 0) + 3
            table[home]["losses"] += 1
            table[home].setdefault("points", 0)
        else:
            table[home]["draws"] += 1
            table[away]["draws"] += 1
            table[home]["points"] = table[home].get("points", 0) + 1
            table[away]["points"] = table[away].get("points", 0) + 1

    for t in table.values():
        t.setdefault("points", 0)
        t["gd"] = t["gf"] - t["ga"]

    df = pd.DataFrame(table.values())
    df["head_to_head"] = 0
    for _, group in df.groupby(["points", "wins", "gd", "gf"]):
        if len(group) <= 1:
            continue
        teams_tied = group["team"].tolist()
        h2h = _head_to_head_points(played, teams_tied)
        for t, val in h2h.items():
            df.loc[df["team"] == t, "head_to_head"] = val

    df = df.sort_values(
        ["points", "wins", "gd", "gf", "head_to_head", "team"],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _simulate_table(
    played_df: pd.DataFrame,
    remaining: pd.DataFrame,
    rng: np.random.Generator,
    strengths: Dict[str, float] | None = None,
    *,
    expected_goals: float = 1.2,
) -> pd.DataFrame:
    """Simulate remaining fixtures using team strengths.

    Parameters
    ----------
    expected_goals:
        Average goals scored by each side in a match. This is used as the
        base Poisson mean when generating random scores.
    """
    sims: list[dict] = []
    for _, row in remaining.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if strengths is None:
            lam_home = expected_goals
            lam_away = expected_goals
        else:
            s_h = strengths.get(home, 1.0)
            s_a = strengths.get(away, 1.0)
            total = s_h + s_a
            if total == 0:
                lam_home = lam_away = expected_goals
            else:
                lam_home = expected_goals * (2 * s_h / total)
                lam_away = expected_goals * (2 * s_a / total)

        hs = int(rng.poisson(lam_home))
        as_ = int(rng.poisson(lam_away))
        sims.append(
            {
                "date": row["date"],
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": as_,
            }
        )
    all_matches = pd.concat([played_df, pd.DataFrame(sims)], ignore_index=True)
    return league_table(all_matches)


# ---------------------------------------------------------------------------
# Public simulation API
# ---------------------------------------------------------------------------

def simulate_chances(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    strengths: Dict[str, float] | None = None,
    expected_goals: float = 1.2,
) -> Dict[str, float]:
    """Return title probabilities using the given team strengths."""
    if rng is None:
        rng = np.random.default_rng()

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    champs = {t: 0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[matches["home_score"].isna() | matches["away_score"].isna()]

    for _ in range(iterations):
        table = _simulate_table(
            played_df,
            remaining,
            rng,
            strengths,
            expected_goals=expected_goals,
        )
        champs[table.iloc[0]["team"]] += 1

    for t in champs:
        champs[t] /= iterations
    return champs


def simulate_relegation_chances(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    strengths: Dict[str, float] | None = None,
    expected_goals: float = 1.2,
) -> Dict[str, float]:
    """Return probabilities of finishing in the bottom four."""
    if rng is None:
        rng = np.random.default_rng()

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    relegated = {t: 0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[matches["home_score"].isna() | matches["away_score"].isna()]

    for _ in range(iterations):
        table = _simulate_table(
            played_df,
            remaining,
            rng,
            strengths,
            expected_goals=expected_goals,
        )
        for team in table.tail(4)["team"]:
            relegated[team] += 1

    for t in relegated:
        relegated[t] /= iterations
    return relegated


def simulate_final_table(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    strengths: Dict[str, float] | None = None,
    expected_goals: float = 1.2,
) -> pd.DataFrame:
    """Project average finishing position and points."""
    if rng is None:
        rng = np.random.default_rng()

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    pos_totals = {t: 0.0 for t in teams}
    points_totals = {t: 0.0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[matches["home_score"].isna() | matches["away_score"].isna()]

    for _ in range(iterations):
        table = _simulate_table(
            played_df,
            remaining,
            rng,
            strengths,
            expected_goals=expected_goals,
        )
        for idx, row in table.iterrows():
            pos_totals[row["team"]] += idx + 1
            points_totals[row["team"]] += row["points"]

    results = []
    for team in teams:
        results.append(
            {
                "team": team,
                "position": pos_totals[team] / iterations,
                "points": points_totals[team] / iterations,
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("position").reset_index(drop=True)
    return df


def summary_table(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    strengths: Dict[str, float] | None = None,
    expected_goals: float = 1.2,
) -> pd.DataFrame:
    """Return a combined projection table."""
    chances = simulate_chances(
        matches,
        iterations=iterations,
        rng=rng,
        strengths=strengths,
        expected_goals=expected_goals,
    )
    relegation = simulate_relegation_chances(
        matches,
        iterations=iterations,
        rng=rng,
        strengths=strengths,
        expected_goals=expected_goals,
    )
    table = simulate_final_table(
        matches,
        iterations=iterations,
        rng=rng,
        strengths=strengths,
        expected_goals=expected_goals,
    )

    table = table.sort_values("position").reset_index(drop=True)
    table["position"] = range(1, len(table) + 1)
    table["points"] = table["points"].round().astype(int)
    table["title"] = table["team"].map(chances)
    table["relegation"] = table["team"].map(relegation)
    return table[["position", "team", "points", "title", "relegation"]]
