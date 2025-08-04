"""Automatic parameter estimation utilities."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from simulator import parse_matches


def estimate_parameters(
    paths: List[str], decay: float | None = None
) -> tuple[float, float]:
    """Estimate draw rate and home advantage from past seasons.

    Parameters
    ----------
    paths:
        A list of text file paths containing fixture results in the
        SportsClubStats format.  Files should be ordered from most recent to
        oldest when using ``decay``.
    decay:
        Optional exponential decay factor applied to older seasons. Games from
        the ``n``th file in ``paths`` are weighted by ``decay**n``. Use ``None``
        to give all seasons equal weight.

    Returns
    -------
    tuple[float, float]
        The ``(tie_percent, home_advantage)`` calculated from the data.
    """

    total_games = 0.0
    home_wins = 0.0
    away_wins = 0.0
    draws = 0.0

    for n, path in enumerate(paths):
        weight = 1.0 if decay is None else decay**n
        if weight == 0:
            continue
        df = parse_matches(path)
        played = df.dropna(subset=["home_score", "away_score"])
        total_games += len(played) * weight
        home_wins += (played["home_score"] > played["away_score"]).sum() * weight
        away_wins += (played["home_score"] < played["away_score"]).sum() * weight
        draws += (played["home_score"] == played["away_score"]).sum() * weight

    if total_games == 0:
        raise ValueError("No played games found in provided paths")

    tie_percent = float(100.0 * draws / total_games)
    if away_wins == 0:
        home_advantage = 1.0
    else:
        home_advantage = float(home_wins / away_wins)

    return tie_percent, home_advantage


def estimate_goal_means(
    paths: List[str], decay: float | None = None
) -> tuple[float, float]:
    """Estimate average goals scored by home and away teams.

    Parameters
    ----------
    paths:
        A list of text file paths containing fixture results in the
        SportsClubStats format.  Files should be ordered from most recent to
        oldest when using ``decay``.
    decay:
        Optional exponential decay factor applied to older seasons. Games from
        the ``n``th file in ``paths`` are weighted by ``decay**n``. Use ``None``
        to give all seasons equal weight.

    Returns
    -------
    tuple[float, float]
        The ``(home_goals_mean, away_goals_mean)`` calculated from the data.
    """

    total_games = 0.0
    total_home_goals = 0.0
    total_away_goals = 0.0

    for n, path in enumerate(paths):
        weight = 1.0 if decay is None else decay**n
        if weight == 0:
            continue
        df = parse_matches(path)
        played = df.dropna(subset=["home_score", "away_score"])
        total_games += len(played) * weight
        total_home_goals += played["home_score"].sum() * weight
        total_away_goals += played["away_score"].sum() * weight

    if total_games == 0:
        raise ValueError("No played games found in provided paths")

    home_goals_mean = float(total_home_goals / total_games)
    away_goals_mean = float(total_away_goals / total_games)
    return home_goals_mean, away_goals_mean


def estimate_team_strengths(
    paths: List[str], decay: float | None = None
) -> Dict[str, tuple[float, float]]:
    """Estimate attack and defense multipliers for each team.

    The returned values are normalized so that ``1.0`` represents league
    average performance.  Values greater than 1.0 indicate stronger attack or
    weaker defense respectively.

    Parameters
    ----------
    paths:
        A list of text file paths containing historical fixtures with results.
        Files should be ordered from most recent to oldest when using
        ``decay``.
    decay:
        Optional exponential decay factor applied to older seasons.  Games from
        the ``n``th file in ``paths`` are weighted by ``decay**n``.  Use
        ``None`` to give all seasons equal weight.

    Returns
    -------
    Dict[str, tuple[float, float]]
        Mapping of team name to ``(attack, defense)`` multipliers.
    """

    stats: Dict[str, Dict[str, float]] = {}

    for n, path in enumerate(paths):
        weight = 1.0 if decay is None else decay**n
        if weight == 0:
            continue
        df = parse_matches(path)
        played = df.dropna(subset=["home_score", "away_score"])

        teams = pd.unique(played[["home_team", "away_team"]].values.ravel())
        for t in teams:
            stats.setdefault(t, {"gf": 0.0, "ga": 0.0, "w": 0.0})

        for _, row in played.iterrows():
            ht = row["home_team"]
            at = row["away_team"]
            hs = float(row["home_score"])
            as_ = float(row["away_score"])
            stats[ht]["gf"] += hs * weight
            stats[ht]["ga"] += as_ * weight
            stats[ht]["w"] += weight
            stats[at]["gf"] += as_ * weight
            stats[at]["ga"] += hs * weight
            stats[at]["w"] += weight

    if not stats:
        return {}

    total_goals = sum(s["gf"] for s in stats.values())
    total_weight = sum(s["w"] for s in stats.values())
    if total_weight == 0:
        raise ValueError("No played games found in provided paths")

    avg_goals = total_goals / total_weight

    strengths: Dict[str, tuple[float, float]] = {}
    for team, s in stats.items():
        if s["w"] == 0:
            continue
        attack = (s["gf"] / s["w"]) / avg_goals
        defense = (s["ga"] / s["w"]) / avg_goals
        strengths[team] = (attack, defense)

    return strengths
