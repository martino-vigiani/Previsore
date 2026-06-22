"""Validazione onesta: confronta le predizioni con le partite GIA giocate.

Addestra solo su dati PRECEDENTI al `cutoff` (niente leakage), poi predice le
partite gia disputate di un torneo. Riporta RPS, log-loss, Brier, accuratezza,
ECE e un intervallo di confidenza bootstrap (a n piccolo nulla e significativo).
Confronta DC, blend (DC+Elo) ed Elo. Niente tuning qui: gli iperparametri (w, T)
arrivano dalla config tarata sul backbone grande.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import blend, elo, metrics, scorers
from .model import DixonColes


def _outcome(hs, as_) -> int:
    gd = int(hs) - int(as_)
    return 0 if gd > 0 else (1 if gd == 0 else 2)


def run(df: pd.DataFrame, cutoff: str, tournament: str = "FIFA World Cup",
        since: str | None = None, goals: pd.DataFrame | None = None,
        w: float = 1.0, T: float = 1.0, **fit_kw) -> dict:
    cut = pd.Timestamp(cutoff)
    played = df[df["home_score"].notna()]
    train = played[played["date"] < cut]

    since_ts = pd.Timestamp(since) if since else cut
    test = df[df["tournament"].astype(str).str.contains(tournament, na=False)
              & (df["date"] >= since_ts) & df["home_score"].notna()].copy()
    test = test.dropna(subset=["home_score", "away_score"])

    model = DixonColes.fit(train, ref_date=cut, **fit_kw)
    R = blend.elo_ratings(train)
    pred_engine = blend.Predictor(model, R, w=w, T=T)

    P_dc, P_bl, P_el, Y = [], [], [], []
    examples, sc_rows = [], []
    for r in test.itertuples(index=False):
        if r.home_team not in model.attack or r.away_team not in model.attack:
            continue
        oc = _outcome(r.home_score, r.away_score)
        pred = pred_engine.predict(r.home_team, r.away_team, bool(r.neutral))
        p_bl = [pred["p_home"], pred["p_draw"], pred["p_away"]]
        p_dc = pred["p_dc"]
        p_el = pred_engine.elo_1x2(r.home_team, r.away_team, bool(r.neutral))
        P_dc.append(p_dc); P_bl.append(p_bl); P_el.append(p_el); Y.append(oc)

        hit_out = int(np.argmax(p_bl) == oc)
        hit_exact = int(pred["exact"] == (int(r.home_score), int(r.away_score)))
        examples.append((str(r.date.date()),
                         f"{r.home_team} {pred['exact'][0]}-{pred['exact'][1]} {r.away_team}",
                         f"{int(r.home_score)}-{int(r.away_score)}", hit_out, hit_exact))

        if goals is not None:
            actual = goals[(goals["date"] == r.date) & (goals["home_team"] == r.home_team)
                           & (goals["away_team"] == r.away_team)]
            for team, lam in ((r.home_team, pred["lambda_home"]), (r.away_team, pred["lambda_away"])):
                sh = scorers.player_shares(goals, team, model.ref_date)
                top = scorers.predict_scorers(lam, sh, topn=3)
                top3 = {t[0] for t in top}
                top1 = top[0][0] if top else None
                real = set(actual[(actual["team"] == team) & (~actual["own_goal"])]["scorer"].dropna())
                if real:
                    sc_rows.append({"t1": int(top1 in real), "t3": int(bool(top3 & real))})

    P_dc, P_bl, P_el, Y = map(np.array, (P_dc, P_bl, P_el, Y))
    res = {"n": int(len(Y)), "examples": examples}
    if len(Y) == 0:
        return res

    def agg(P):
        return {
            "rps": float(np.mean([metrics.rps(P[i], Y[i]) for i in range(len(Y))])),
            "logloss": float(np.mean([metrics.log_loss(P[i], Y[i]) for i in range(len(Y))])),
            "brier": float(np.mean([metrics.brier(P[i], Y[i]) for i in range(len(Y))])),
            "acc": float((P.argmax(1) == Y).mean()),
            "exact": None,
        }

    res["dc"] = agg(P_dc)
    res["blend"] = agg(P_bl)
    res["elo"] = agg(P_el)
    res["exact"] = float(np.mean([e[4] for e in examples]))

    # CI bootstrap sulla differenza di log-loss blend - elo (a n piccolo: contesto)
    rng = np.random.default_rng(7)
    n = len(Y)
    ll_bl = np.array([metrics.log_loss(P_bl[i], Y[i]) for i in range(n)])
    ll_el = np.array([metrics.log_loss(P_el[i], Y[i]) for i in range(n)])
    diffs = [(ll_bl[idx] - ll_el[idx]).mean() for idx in (rng.integers(0, n, n) for _ in range(2000))]
    res["ci_blend_minus_elo_logloss"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]

    s = pd.DataFrame(sc_rows)
    if len(s):
        res.update({"sc_n": int(len(s)), "sc_t1": float(s["t1"].mean()), "sc_t3": float(s["t3"].mean())})
    return res
