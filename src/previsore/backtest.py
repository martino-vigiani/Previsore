"""Backtest temporale e walk-forward, con metriche probabilistiche e CI bootstrap.

`run`         = split singolo (rapido, leggermente ottimista).
`walk_forward`= refit a finestra espandente (uno per anno): valutazione onesta
                out-of-sample su un grande backbone, con log-loss/Brier/RPS e
                intervalli di confidenza bootstrap sulle differenze tra modelli.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import blend, elo, metrics
from .model import DixonColes


def _outcome(hs, as_) -> int:
    gd = int(hs) - int(as_)
    return 0 if gd > 0 else (1 if gd == 0 else 2)


# ---------------------------------------------------------------- split singolo
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
        oc = _outcome(r.home_score, r.away_score)
        pred = model.predict(r.home_team, r.away_team, bool(r.neutral))
        p_dc = [pred["p_home"], pred["p_draw"], pred["p_away"]]
        p_el = list(elo.probs(R.get(r.home_team, 1500.0), R.get(r.away_team, 1500.0), bool(r.neutral)))
        rows.append({
            "rps_dc": metrics.rps(p_dc, oc), "rps_elo": metrics.rps(p_el, oc),
            "hit_dc": int(np.argmax(p_dc) == oc), "hit_elo": int(np.argmax(p_el) == oc),
            "exact_dc": int(pred["exact"] == (int(r.home_score), int(r.away_score))),
        })
    d = pd.DataFrame(rows)
    if d.empty:
        return {"n": 0}
    return {
        "n": int(len(d)),
        "rps_dc": float(d["rps_dc"].mean()), "rps_elo": float(d["rps_elo"].mean()),
        "acc1x2_dc": float(d["hit_dc"].mean()), "acc1x2_elo": float(d["hit_elo"].mean()),
        "exact_dc": float(d["exact_dc"].mean()),
    }


# --------------------------------------------------------------- walk-forward
def walk_forward(played: pd.DataFrame, start: str = "2018-01-01", end: str = "2026-06-11",
                 w: float = 1.0, T: float = 1.0, **fit_kw) -> dict:
    """Refit annuale a finestra espandente; valuta DC, Elo e blend."""
    played = played.dropna(subset=["home_score", "away_score"])
    y0 = pd.Timestamp(start).year
    y1 = pd.Timestamp(end).year
    P_dc, P_el, P_bl, Y = [], [], [], []
    for yr in range(y0, y1 + 1):
        cut = pd.Timestamp(f"{yr}-01-01")
        nxt = pd.Timestamp(f"{yr + 1}-01-01")
        train = played[played["date"] < cut]
        batch = played[(played["date"] >= cut) & (played["date"] < min(nxt, pd.Timestamp(end)))]
        if len(train) < 500 or batch.empty:
            continue
        m = DixonColes.fit(train, ref_date=cut, verbose=False, **fit_kw)
        R, _ = elo.compute(train)
        for r in batch.itertuples(index=False):
            if r.home_team not in m.attack or r.away_team not in m.attack:
                continue
            oc = _outcome(r.home_score, r.away_score)
            pr = m.predict(r.home_team, r.away_team, bool(r.neutral))
            p_dc = [pr["p_home"], pr["p_draw"], pr["p_away"]]
            p_el = list(elo.probs(R.get(r.home_team, 1500.0), R.get(r.away_team, 1500.0), bool(r.neutral)))
            P_dc.append(p_dc)
            P_el.append(p_el)
            P_bl.append(list(blend.blend_1x2(p_dc, p_el, w, T)))
            Y.append(oc)

    P_dc, P_el, P_bl, Y = map(np.array, (P_dc, P_el, P_bl, Y))
    if len(Y) == 0:
        return {"n": 0}

    def agg(P):
        ll = np.array([metrics.log_loss(P[i], Y[i]) for i in range(len(Y))])
        rp = np.array([metrics.rps(P[i], Y[i]) for i in range(len(Y))])
        br = np.array([metrics.brier(P[i], Y[i]) for i in range(len(Y))])
        acc = (P.argmax(1) == Y).mean()
        return ll, rp, br, acc

    ll_dc, rp_dc, br_dc, acc_dc = agg(P_dc)
    ll_el, rp_el, br_el, acc_el = agg(P_el)
    ll_bl, rp_bl, br_bl, acc_bl = agg(P_bl)

    # bootstrap CI sulla differenza di log-loss (blend - elo) e (dc - elo)
    rng = np.random.default_rng(7)
    n = len(Y)
    def boot_ci(a, b, reps=2000):
        diffs = []
        for _ in range(reps):
            idx = rng.integers(0, n, n)
            diffs.append((a[idx] - b[idx]).mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return float(lo), float(hi)

    return {
        "n": int(n),
        "logloss": {"dc": float(ll_dc.mean()), "elo": float(ll_el.mean()), "blend": float(ll_bl.mean())},
        "rps": {"dc": float(rp_dc.mean()), "elo": float(rp_el.mean()), "blend": float(rp_bl.mean())},
        "brier": {"dc": float(br_dc.mean()), "elo": float(br_el.mean()), "blend": float(br_bl.mean())},
        "acc": {"dc": float(acc_dc), "elo": float(acc_el), "blend": float(acc_bl)},
        "ci_logloss_blend_minus_elo": boot_ci(ll_bl, ll_el),
        "ci_logloss_dc_minus_elo": boot_ci(ll_dc, ll_el),
        "ece": {"dc": metrics.ece(P_dc, Y), "elo": metrics.ece(P_el, Y), "blend": metrics.ece(P_bl, Y)},
    }
