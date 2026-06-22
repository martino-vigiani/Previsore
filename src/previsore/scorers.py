"""Marcatori probabili: quota gol storica + gate rosa reale + rigorista.

I gol attesi di squadra (lambda dal modello) si dividono tra i giocatori per
quota storica di gol nel gioco aperto; la frazione di rigori va al rigorista
designato. Con la rosa reale (squads.py) si filtrano i non convocati/ritirati.
"""
from __future__ import annotations

import math
import unicodedata

import numpy as np
import pandas as pd


def _norm_name(name: str) -> str:
    """Minuscolo senza accenti (per match e per deduplicare 'Álvarez'/'Alvarez')."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.casefold().strip()


def _tokens(name: str) -> set:
    return {t for t in _norm_name(name).split() if len(t) > 2}


def _recent(goals, team, ref, years):
    cutoff = ref - pd.DateOffset(years=years)
    g = goals[(goals["team"] == team) & (goals["date"] >= cutoff) & (goals["date"] <= ref)]
    if "own_goal" in g.columns:
        g = g[~g["own_goal"]]
    return g.dropna(subset=["scorer"]).copy()


def penalty_fraction(goals, team, ref_date, years: int = 4) -> float:
    g = _recent(goals, team, pd.Timestamp(ref_date), years)
    if g.empty or "penalty" not in g.columns:
        return 0.0
    return float(min(g["penalty"].mean(), 0.25))


def penalty_taker(goals, team, ref_date, years: int = 4, squad_tokens=None):
    g = _recent(goals, team, pd.Timestamp(ref_date), years)
    if g.empty or "penalty" not in g.columns:
        return None
    pk = g[g["penalty"]]
    if pk.empty:
        return None
    taker = pk["scorer"].value_counts().idxmax()
    if squad_tokens and not (_tokens(taker) & squad_tokens):
        return None
    return taker


def player_shares(goals, team, ref_date, half_life_days: float = 730.0,
                  recency_years: int = 4, gate_months: int = 30,
                  squad_tokens=None, include_penalties: bool = True) -> dict:
    """Quota di gol per giocatore (somma 1). `squad_tokens`: se dato, tiene solo
    chi combacia con la rosa reale; `gate_months`: scarta gli inattivi (ritirati)."""
    ref = pd.Timestamp(ref_date)
    g = _recent(goals, team, ref, recency_years)
    if not include_penalties and "penalty" in g.columns:
        g = g[~g["penalty"]]
    if g.empty:
        return {}
    # gate attivita: solo chi ha segnato negli ultimi gate_months
    active = set(g[g["date"] >= ref - pd.DateOffset(months=gate_months)]["scorer"].unique())
    g = g[g["scorer"].isin(active)]
    # gate rosa reale (se disponibile); fallback a non-filtrato se azzera tutto
    if squad_tokens:
        gg = g[g["scorer"].map(lambda s: bool(_tokens(s) & squad_tokens))]
        if not gg.empty:
            g = gg
    if g.empty:
        return {}
    xi = math.log(2) / half_life_days
    g = g.assign(w=np.exp(-xi * np.clip((ref - g["date"]).dt.days.to_numpy().astype(float), 0, None)),
                 key=g["scorer"].map(_norm_name))
    # somma pesi per nome normalizzato; etichetta = grafia col peso maggiore
    by_key = g.groupby("key")["w"].sum()
    label = g.sort_values("w").drop_duplicates("key", keep="last").set_index("key")["scorer"]
    s = pd.Series({label[k]: v for k, v in by_key.items()})
    return (s / s.sum()).to_dict()


def predict_scorers(team_lambda: float, shares: dict, topn: int = 5):
    """Compat: lista (giocatore, prob_marcatore, gol_attesi) senza rigorista."""
    out = [(p, 1.0 - math.exp(-team_lambda * sh), team_lambda * sh) for p, sh in shares.items()]
    out.sort(key=lambda t: t[1], reverse=True)
    return out[:topn]


def scorer_probs(goals, team, ref_date, team_lambda: float, squad_tokens=None, topn: int = 4):
    """Lista (giocatore, prob_marcatore, e_rigorista) con quota gioco aperto +
    massa rigori instradata al rigorista."""
    pf = penalty_fraction(goals, team, ref_date)
    open_sh = player_shares(goals, team, ref_date, squad_tokens=squad_tokens, include_penalties=False)
    taker = penalty_taker(goals, team, ref_date, squad_tokens=squad_tokens)
    lam = {p: sh * (1.0 - pf) * team_lambda for p, sh in open_sh.items()}
    if taker and pf > 0:
        lam[taker] = lam.get(taker, 0.0) + pf * team_lambda
    out = [(p, 1.0 - math.exp(-l), p == taker) for p, l in lam.items()]
    out.sort(key=lambda t: t[1], reverse=True)
    return out[:topn]
