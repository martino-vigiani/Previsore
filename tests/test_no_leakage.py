"""Garanzie di onesta: il fit non deve usare partite >= ref_date (no leakage)."""
import pandas as pd

from previsore import metrics
from previsore.model import DixonColes


def _synth() -> pd.DataFrame:
    teams = ["AAA", "BBB", "CCC", "DDD"]
    rows = []
    for i, dt in enumerate(pd.date_range("2020-01-01", "2023-12-01", periods=240)):
        rows.append((dt, teams[i % 4], teams[(i + 1) % 4], i % 4, (i + 2) % 3, "Friendly", False))
    # 'ZZZ' compare SOLO dopo il cutoff: non deve finire nel modello
    for i, dt in enumerate(pd.date_range("2024-02-01", "2024-12-01", periods=40)):
        rows.append((dt, "ZZZ", teams[i % 4], 1, 0, "Friendly", False))
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                       "home_score", "away_score", "tournament", "neutral"])


def test_fit_excludes_future_team():
    df = _synth()
    m = DixonColes.fit(df, ref_date=pd.Timestamp("2024-01-01"), min_matches=1, reg=0.0, verbose=False)
    assert m.ref_date == "2024-01-01"
    assert "ZZZ" not in m.attack      # gioca solo dopo il cutoff -> escluso
    assert "AAA" in m.attack


def test_metrics_sane():
    assert metrics.rps([1, 0, 0], 0) == 0.0
    assert metrics.brier([1, 0, 0], 0) == 0.0
    assert metrics.log_loss([0.99, 0.005, 0.005], 0) < 0.05
    assert metrics.log_loss([0.005, 0.005, 0.99], 0) > 3.0   # previsione sbagliata = penalita alta
