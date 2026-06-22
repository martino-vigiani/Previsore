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


def _accent_richness(name: str) -> int:
    """Numero di segni diacritici: per scegliere la grafia canonica (con accenti)."""
    nf = unicodedata.normalize("NFKD", str(name))
    return sum(1 for c in nf if unicodedata.combining(c))


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


def penalty_taker(goals, team, ref_date, years: int = 4, gate_months: int = 30, squad_tokens=None):
    ref = pd.Timestamp(ref_date)
    g = _recent(goals, team, ref, years)
    if g.empty or "penalty" not in g.columns:
        return None
    # stesso gate attivita di player_shares: niente rigoristi non piu attivi
    pk = g[g["penalty"] & (g["date"] >= ref - pd.DateOffset(months=gate_months))]
    if pk.empty:
        return None
    taker = pk["scorer"].value_counts().idxmax()
    if squad_tokens and not (_tokens(taker) & squad_tokens):
        return None
    return taker


def player_shares(goals, team, ref_date, half_life_days: float = 730.0,
                  recency_years: int = 4, gate_months: int = 30,
                  squad_tokens=None, include_penalties: bool = True, cap: float = 0.50) -> dict:
    """Frazione dei gol di squadra per giocatore. NON rinormalizzata: il
    denominatore e l'intero monte-gol recente, cosi scartare gli inattivi/non
    convocati lascia massa NON attribuita (profondita rosa) invece di gonfiare i
    superstiti. `cap`: tetto per singolo giocatore. Somma <= 1."""
    ref = pd.Timestamp(ref_date)
    g = _recent(goals, team, ref, recency_years)
    if not include_penalties and "penalty" in g.columns:
        g = g[~g["penalty"]]
    if g.empty:
        return {}
    xi = math.log(2) / half_life_days
    g = g.assign(w=np.exp(-xi * np.clip((ref - g["date"]).dt.days.to_numpy().astype(float), 0, None)),
                 key=g["scorer"].map(_norm_name))
    total = g["w"].sum()                      # denominatore = TUTTO il pool recente
    # gate attivita: solo chi ha segnato negli ultimi gate_months
    active = set(g[g["date"] >= ref - pd.DateOffset(months=gate_months)]["scorer"].unique())
    gk = g[g["scorer"].isin(active)]
    if squad_tokens:                          # gate rosa reale (fallback se azzera tutto)
        gg = gk[gk["scorer"].map(lambda s: bool(_tokens(s) & squad_tokens))]
        if not gg.empty:
            gk = gg
    if gk.empty or total <= 0:
        return {}
    by_key = gk.groupby("key")["w"].sum()
    # etichetta = grafia con piu accenti (a parita, peso maggiore)
    gk = gk.assign(acc=gk["scorer"].map(_accent_richness))
    label = (gk.sort_values(["acc", "w"]).drop_duplicates("key", keep="last")
             .set_index("key")["scorer"])
    return {label[k]: min(float(v / total), cap) for k, v in by_key.items()}


def scorer_probs(goals, team, ref_date, team_lambda: float, squad_tokens=None, topn: int = 4):
    """Lista (giocatore, prob_marcatore, e_rigorista) con quota gioco aperto +
    massa rigori instradata al rigorista attivo. Conserva i gol attesi totali:
    senza un rigorista valido la frazione rigori rientra nel gioco aperto."""
    taker = penalty_taker(goals, team, ref_date, squad_tokens=squad_tokens)
    pf = penalty_fraction(goals, team, ref_date) if taker else 0.0
    # se nessun rigorista valido: includi i rigori nel calcolo generale (no mass loss)
    open_sh = player_shares(goals, team, ref_date, squad_tokens=squad_tokens,
                            include_penalties=(taker is None))
    lam = {p: sh * (1.0 - pf) * team_lambda for p, sh in open_sh.items()}
    if taker and pf > 0:
        norm_map = {_norm_name(p): p for p in lam}     # accoppia per nome normalizzato
        key = norm_map.get(_norm_name(taker), taker)
        lam[key] = lam.get(key, 0.0) + pf * team_lambda
    tk = _norm_name(taker) if taker else None
    out = [(p, 1.0 - math.exp(-l), _norm_name(p) == tk) for p, l in lam.items()]
    out.sort(key=lambda t: t[1], reverse=True)
    return out[:topn]
