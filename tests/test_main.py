import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import main


def test_auto_team_strengths_runs_by_default(monkeypatch):
    called = {"value": False}

    def fake_estimate_team_strengths(files):
        called["value"] = True
        return {}

    monkeypatch.setattr(main, "estimate_team_strengths", fake_estimate_team_strengths)
    monkeypatch.setattr(main, "parse_matches", lambda path: pd.DataFrame())
    monkeypatch.setattr(main, "summary_table", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(sys, "argv", ["main.py", "--simulations", "1", "--html-output", ""])
    main.main()
    assert called["value"]
