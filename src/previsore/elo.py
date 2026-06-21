"""Elo per nazionali (stile World Football Elo) + modello di pareggio Davidson.

Usato come baseline indipendente e per confronto nel backtest. Il modello
principale di previsione resta Dixon-Coles (model.py).
"""
from __future__ import annotations

import math

import pandas as pd


def _k(tournament: str) -> float:
    """Peso K per importanza torneo (stile eloratings.net, semplificato)."""
    t = str(tournament).lower()
    if "friendly" in t:
        return 20.0
    if "world cup" in t and "qual" not in t:
        return 60.0
    if "qualif" in t:
        return 40.0
    if any(s in t for s in ("euro", "copa", "african cup", "asian cup",
                            "gold cup", "nations league", "confederations")):
        return 50.0
    return 30.0


def _g(goal_diff: int) -> float:
    """Moltiplicatore margine di vittoria."""
    d = abs(goal_diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11.0 + d) / 8.0


def compute(played: pd.DataFrame, hfa: float = 65.0, init: float = 1500.0):
    """Calcola Elo iterando in ordine di data.

    Ritorna (ratings_finali: dict, pre_match: list[(R_home, R_away)]).
    Una passata lineare: pochi secondi anche su 49k partite.
    """
    df = played.dropna(subset=["home_score", "away_score"]).sort_values("date")
    R: dict[str, float] = {}
    pre: list[tuple[float, float]] = []
    for row in df.itertuples(index=False):
        rh = R.get(row.home_team, init)
        ra = R.get(row.away_team, init)
        adv = 0.0 if row.neutral else hfa
        eh = 1.0 / (1.0 + 10 ** (-(rh + adv - ra) / 400.0))
        gd = int(row.home_score) - int(row.away_score)
        wh = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        k = _k(row.tournament) * _g(gd)
        pre.append((rh, ra))
        R[row.home_team] = rh + k * (wh - eh)
        R[row.away_team] = ra + k * ((1.0 - wh) - (1.0 - eh))
    return R, pre


def probs(rh: float, ra: float, neutral: bool = False, hfa: float = 65.0, nu: float = 1.15):
    """Probabilita 1X2 dal modello di pareggio di Davidson.

    nu controlla la frequenza dei pareggi. Ritorna (pH, pD, pA).
    """
    adv = 0.0 if neutral else hfa
    a = 10 ** ((rh + adv) / 400.0)
    b = 10 ** (ra / 400.0)
    g = nu * math.sqrt(a * b)
    denom = a + b + g
    return a / denom, g / denom, b / denom
