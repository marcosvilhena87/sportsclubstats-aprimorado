"""Convenience exports for the simulator package."""

from .simulator import (
    league_table,
    parse_matches,
    reset_results_from,
    home_goal_rate,
    away_goal_rate,
    estimate_home_field_advantage,
    DEFAULT_TIE_PERCENT,
    simulate_chances,
    simulate_relegation_chances,
    simulate_final_table,
    summary_table,
)

__all__ = [
    "parse_matches",
    "league_table",
    "reset_results_from",
    "home_goal_rate",
    "away_goal_rate",
    "estimate_home_field_advantage",
    "simulate_chances",
    "simulate_relegation_chances",
    "simulate_final_table",
    "summary_table",
    "DEFAULT_TIE_PERCENT",
]
