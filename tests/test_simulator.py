import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pandas as pd
import numpy as np
from brasileirao import parse_matches, league_table, simulate_chances
from brasileirao import simulator


def test_parse_matches():
    df = parse_matches('data/Brasileirao2025A.txt')
    assert len(df) == 380
    assert {'home_team', 'away_team', 'home_score', 'away_score'}.issubset(df.columns)


def test_league_table():
    df = parse_matches('data/Brasileirao2025A.txt')
    table = league_table(df)
    assert 'points' in table.columns
    assert table['played'].max() > 0


def test_league_table_deterministic_sorting():
    data = [
        {'date': '2025-01-01', 'home_team': 'Alpha', 'away_team': 'Beta', 'home_score': 1, 'away_score': 0},
        {'date': '2025-01-02', 'home_team': 'Beta', 'away_team': 'Gamma', 'home_score': 1, 'away_score': 0},
        {'date': '2025-01-03', 'home_team': 'Gamma', 'away_team': 'Alpha', 'home_score': 1, 'away_score': 0},
    ]
    df = pd.DataFrame(data)
    table = league_table(df)
    assert list(table.team) == sorted(table.team)


def test_simulate_chances_sum_to_one():
    df = parse_matches('data/Brasileirao2025A.txt')
    chances = simulate_chances(df, iterations=10)
    assert abs(sum(chances.values()) - 1.0) < 1e-6


def test_simulate_chances_seed_repeatability():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(1234)
    chances1 = simulate_chances(df, iterations=5, rng=rng)
    rng = np.random.default_rng(1234)
    chances2 = simulate_chances(df, iterations=5, rng=rng)
    assert chances1 == chances2




def test_simulate_relegation_chances_sum_to_four():
    df = parse_matches('data/Brasileirao2025A.txt')
    probs = simulator.simulate_relegation_chances(df, iterations=10)
    assert abs(sum(probs.values()) - 4.0) < 1e-6


def test_simulate_relegation_chances_seed_repeatability():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(123)
    first = simulator.simulate_relegation_chances(df, iterations=5, rng=rng)
    rng = np.random.default_rng(123)
    second = simulator.simulate_relegation_chances(df, iterations=5, rng=rng)
    assert first == second


def test_simulate_final_table_deterministic():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(1)
    table1 = simulator.simulate_final_table(df, iterations=5, rng=rng)
    rng = np.random.default_rng(1)
    table2 = simulator.simulate_final_table(df, iterations=5, rng=rng)
    pd.testing.assert_frame_equal(table1, table2)
    assert {"team", "position", "points"}.issubset(table1.columns)


def test_summary_table_deterministic():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(5)
    table1 = simulator.summary_table(df, iterations=5, rng=rng)
    rng = np.random.default_rng(5)
    table2 = simulator.summary_table(df, iterations=5, rng=rng)
    pd.testing.assert_frame_equal(table1, table2)
    assert {"position", "team", "points", "title", "relegation"}.issubset(table1.columns)


def _strengths_from_df(df: pd.DataFrame) -> dict:
    teams = sorted(pd.unique(df[["home_team", "away_team"]].values.ravel()))
    return {t: i + 1 for i, t in enumerate(teams)}


def test_simulate_chances_strengths_seed_repeatability():
    df = parse_matches("data/Brasileirao2025A.txt")
    strengths = _strengths_from_df(df)
    rng = np.random.default_rng(7)
    first = simulate_chances(df, iterations=5, rng=rng, strengths=strengths)
    rng = np.random.default_rng(7)
    second = simulate_chances(df, iterations=5, rng=rng, strengths=strengths)
    assert first == second


def test_simulate_relegation_chances_strengths_seed_repeatability():
    df = parse_matches("data/Brasileirao2025A.txt")
    strengths = _strengths_from_df(df)
    rng = np.random.default_rng(8)
    first = simulator.simulate_relegation_chances(
        df, iterations=5, rng=rng, strengths=strengths
    )
    rng = np.random.default_rng(8)
    second = simulator.simulate_relegation_chances(
        df, iterations=5, rng=rng, strengths=strengths
    )
    assert first == second


def test_simulate_final_table_strengths_deterministic():
    df = parse_matches("data/Brasileirao2025A.txt")
    strengths = _strengths_from_df(df)
    rng = np.random.default_rng(9)
    table1 = simulator.simulate_final_table(
        df, iterations=5, rng=rng, strengths=strengths
    )
    rng = np.random.default_rng(9)
    table2 = simulator.simulate_final_table(
        df, iterations=5, rng=rng, strengths=strengths
    )
    pd.testing.assert_frame_equal(table1, table2)


def test_summary_table_strengths_deterministic():
    df = parse_matches("data/Brasileirao2025A.txt")
    strengths = _strengths_from_df(df)
    rng = np.random.default_rng(10)
    table1 = simulator.summary_table(
        df, iterations=5, rng=rng, strengths=strengths
    )
    rng = np.random.default_rng(10)
    table2 = simulator.summary_table(
        df, iterations=5, rng=rng, strengths=strengths
    )
    pd.testing.assert_frame_equal(table1, table2)


def test_league_table_tiebreakers():
    data = [
        {"date": "2025-01-01", "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 2},
        {"date": "2025-01-02", "home_team": "A", "away_team": "C", "home_score": 1, "away_score": 0},
        {"date": "2025-01-03", "home_team": "C", "away_team": "A", "home_score": 0, "away_score": 1},
        {"date": "2025-01-04", "home_team": "B", "away_team": "C", "home_score": 3, "away_score": 0},
    ]
    df = pd.DataFrame(data)
    table = league_table(df)
    assert list(table.team[:2]) == ["B", "A"]


def test_simulate_final_table_zero_expected_goals_draws():
    df = pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
            },
            {
                "date": "2025-01-02",
                "home_team": "B",
                "away_team": "A",
                "home_score": np.nan,
                "away_score": np.nan,
            },
        ]
    )
    rng = np.random.default_rng(42)
    table = simulator.simulate_final_table(
        df, iterations=1, rng=rng, expected_goals=0
    )
    points = dict(zip(table.team, table.points.round().astype(int)))
    assert points["A"] == 4
    assert points["B"] == 1
