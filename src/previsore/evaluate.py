"""Validazione onesta: confronta le predizioni con le partite GIA giocate.

Addestra il modello solo su dati PRECEDENTI al `cutoff` (niente leakage), poi
predice le partite di un torneo gia disputate e misura accuratezza, RPS e
hit-rate marcatori. Confronto con baseline Elo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import elo, scorers
from .model import DixonColes


def _rps(p, o: int) -> float:
    oo = [0.0, 0.0, 0.0]
    oo[o] = 1.0
    return float(((np.cumsum(p) - np.cumsum(oo))[:2] ** 2).sum() / 2.0)


def run(df: pd.DataFrame, cutoff: str, tournament: str = "FIFA World Cup",
        since: str | None = None, goals: pd.DataFrame | None = None, **fit_kw) -> dict:
    cut = pd.Timestamp(cutoff)
    played = df[df["home_score"].notna()]
    train = played[played["date"] < cut]

    since_ts = pd.Timestamp(since) if since else cut
    test = df[df["tournament"].astype(str).str.contains(tournament, na=False)
              & (df["date"] >= since_ts) & df["home_score"].notna()].copy()
    test = test.dropna(subset=["home_score", "away_score"])

    model = DixonColes.fit(train, ref_date=cut, **fit_kw)
    R, _ = elo.compute(train)

    rows, sc_rows, examples = [], [], []
    for r in test.itertuples(index=False):
        if r.home_team not in model.attack or r.away_team not in model.attack:
            continue
        gd = int(r.home_score) - int(r.away_score)
        oc = 0 if gd > 0 else (1 if gd == 0 else 2)
        pred = model.predict(r.home_team, r.away_team, bool(r.neutral))
        p = [pred["p_home"], pred["p_draw"], pred["p_away"]]
        rh = R.get(r.home_team, 1500.0)
        ra = R.get(r.away_team, 1500.0)
        pe = list(elo.probs(rh, ra, bool(r.neutral)))
        hit_out = int(np.argmax(p) == oc)
        hit_exact = int(pred["exact"] == (int(r.home_score), int(r.away_score)))
        rows.append({"out": hit_out, "exact": hit_exact, "rps": _rps(p, oc),
                     "out_elo": int(np.argmax(pe) == oc), "rps_elo": _rps(pe, oc)})
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

    t = pd.DataFrame(rows)
    res = {"n": int(len(t)), "examples": examples}
    if len(t):
        res.update({
            "out": float(t["out"].mean()), "exact": float(t["exact"].mean()),
            "rps": float(t["rps"].mean()),
            "out_elo": float(t["out_elo"].mean()), "rps_elo": float(t["rps_elo"].mean()),
        })
    s = pd.DataFrame(sc_rows)
    if len(s):
        res.update({"sc_n": int(len(s)), "sc_t1": float(s["t1"].mean()), "sc_t3": float(s["t3"].mean())})
    return res
