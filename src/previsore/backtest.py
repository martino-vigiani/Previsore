"""Backtest temporale: addestra sul passato, valuta sul futuro.

Metriche: RPS (Ranked Probability Score, piu basso = meglio), accuratezza 1X2,
hit-rate risultato esatto. Confronto Dixon-Coles vs Elo.

Nota MVP: split singolo, modello fittato una volta (niente walk-forward refit per
ogni partita). Leggermente ottimista ma rapido e onesto come ordine di grandezza.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import elo
from .model import DixonColes


def _rps(p, outcome_idx: int) -> float:
    """RPS per 3 esiti ordinati [H, D, A]."""
    o = [0.0, 0.0, 0.0]
    o[outcome_idx] = 1.0
    cp = np.cumsum(p)
    co = np.cumsum(o)
    return float(((cp - co)[:2] ** 2).sum() / 2.0)


def run(played: pd.DataFrame, cutoff: str, **fit_kw) -> dict:
    cut = pd.Timestamp(cutoff)
    train = played[played["date"] < cut]
    test = played[(played["date"] >= cut) & played["home_score"].notna()].copy()
    test = test.dropna(subset=["home_score", "away_score"])

    model = DixonColes.fit(train, ref_date=cut, **fit_kw)
    R, _ = elo.compute(train)

    rows = []
    for r in test.itertuples(index=False):
        if r.home_team not in model.attack or r.away_team not in model.attack:
            continue
        gd = int(r.home_score) - int(r.away_score)
        oc = 0 if gd > 0 else (1 if gd == 0 else 2)
        pred = model.predict(r.home_team, r.away_team, bool(r.neutral))
        p_dc = [pred["p_home"], pred["p_draw"], pred["p_away"]]
        rh = R.get(r.home_team, 1500.0)
        ra = R.get(r.away_team, 1500.0)
        p_el = list(elo.probs(rh, ra, bool(r.neutral)))
        rows.append({
            "rps_dc": _rps(p_dc, oc), "rps_elo": _rps(p_el, oc),
            "hit_dc": int(np.argmax(p_dc) == oc),
            "hit_elo": int(np.argmax(p_el) == oc),
            "exact_dc": int(pred["exact"] == (int(r.home_score), int(r.away_score))),
        })

    d = pd.DataFrame(rows)
    if d.empty:
        return {"n": 0}
    return {
        "n": int(len(d)),
        "rps_dc": float(d["rps_dc"].mean()),
        "rps_elo": float(d["rps_elo"].mean()),
        "acc1x2_dc": float(d["hit_dc"].mean()),
        "acc1x2_elo": float(d["hit_elo"].mean()),
        "exact_dc": float(d["exact_dc"].mean()),
    }
