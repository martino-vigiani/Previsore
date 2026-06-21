"""Marcatori probabili (euristica gratis, senza formazioni live).

Idea: i gol attesi di squadra (lambda dal modello Dixon-Coles) vengono distribuiti
tra i giocatori in base alla loro quota storica di gol della nazionale, pesata per
recency (decadimento temporale). Poi prob. "marcatore in qualsiasi momento" per
giocatore p = 1 - exp(-lambda_giocatore), con lambda_giocatore = lambda_squadra * quota.

Limite: senza la formazione titolare (esce ~1h pre-partita) e senza filtrare i
ritirati, e una stima a livello di rosa-recente, non di XI confermato.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def player_shares(goals: pd.DataFrame, team: str, ref_date,
                  half_life_days: float = 1095.0, recency_years: int = 5) -> dict:
    """Quota storica di gol per giocatore della nazionale `team` (somma = 1)."""
    ref = pd.Timestamp(ref_date)
    cutoff = ref - pd.DateOffset(years=recency_years)
    g = goals[(goals["team"] == team) & (goals["date"] >= cutoff) & (goals["date"] <= ref)]
    if "own_goal" in g.columns:
        g = g[~g["own_goal"]]              # gli autogol non contano per il marcatore
    g = g.dropna(subset=["scorer"])
    if g.empty:
        return {}
    xi = math.log(2) / half_life_days
    age = (ref - g["date"]).dt.days.to_numpy().astype(float)
    w = np.exp(-xi * np.clip(age, 0, None))
    s = pd.Series(w, index=g["scorer"].to_numpy()).groupby(level=0).sum()
    return (s / s.sum()).to_dict()


def predict_scorers(team_lambda: float, shares: dict, topn: int = 5):
    """Lista (giocatore, prob_marcatore, gol_attesi) ordinata per probabilita."""
    out = []
    for player, share in shares.items():
        lam_p = team_lambda * share
        out.append((player, 1.0 - math.exp(-lam_p), lam_p))
    out.sort(key=lambda t: t[1], reverse=True)
    return out[:topn]
