"""Ensemble DC + Elo sul vettore 1X2, con temperature scaling.

Il risultato esatto e i marcatori restano dal modello Dixon-Coles (che produce
la matrice dei punteggi); il blend migliora solo le probabilita 1X2 calibrate.
I pesi (w, T) vanno tunati OUT-OF-SAMPLE su un grande backbone, mai sulle poche
partite del Mondiale.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import elo, metrics
from .model import DixonColes


def elo_ratings(played: pd.DataFrame) -> dict:
    R, _ = elo.compute(played)
    return R


def blend_1x2(p_dc, p_elo, w: float, T: float) -> np.ndarray:
    p = w * np.asarray(p_dc, dtype=float) + (1.0 - w) * np.asarray(p_elo, dtype=float)
    p = p / p.sum()
    return metrics.temperature(p, T)


def tune(played: pd.DataFrame, val_cutoff: str = "2023-01-01", **fit_kw) -> dict:
    """Sceglie (ensemble_w, temperature) minimizzando il log-loss su una finestra
    di validazione ampia (tutte le partite dal val_cutoff in poi)."""
    cut = pd.Timestamp(val_cutoff)
    train = played[played["date"] < cut]
    val = played[played["date"] >= cut].dropna(subset=["home_score", "away_score"])
    m = DixonColes.fit(train, ref_date=cut, verbose=False, **fit_kw)
    R = elo_ratings(train)

    samples = []
    for r in val.itertuples(index=False):
        if r.home_team not in m.attack or r.away_team not in m.attack:
            continue
        gd = int(r.home_score) - int(r.away_score)
        o = 0 if gd > 0 else (1 if gd == 0 else 2)
        pr = m.predict(r.home_team, r.away_team, bool(r.neutral))
        p_dc = [pr["p_home"], pr["p_draw"], pr["p_away"]]
        rh = R.get(r.home_team, 1500.0)
        ra = R.get(r.away_team, 1500.0)
        p_el = list(elo.probs(rh, ra, bool(r.neutral)))
        samples.append((p_dc, p_el, o))

    best = None
    for w in np.linspace(0.0, 1.0, 21):
        for T in np.linspace(0.8, 1.6, 17):
            ll = np.mean([metrics.log_loss(blend_1x2(pd_, pe_, w, T), o) for pd_, pe_, o in samples])
            if best is None or ll < best[0]:
                best = (ll, float(round(w, 3)), float(round(T, 3)))
    return {"ensemble_w": best[1], "temperature": best[2],
            "val_logloss": float(best[0]), "val_n": len(samples), "val_cutoff": val_cutoff}


class Predictor:
    """Avvolge il modello DC + rating Elo + config di blend."""

    def __init__(self, dc: DixonColes, ratings: dict, w: float = 1.0, T: float = 1.0):
        self.dc = dc
        self.R = ratings
        self.w = w
        self.T = T

    def elo_1x2(self, home, away, neutral=False):
        rh = self.R.get(home, 1500.0)
        ra = self.R.get(away, 1500.0)
        return list(elo.probs(rh, ra, neutral))

    def predict(self, home, away, neutral=False, topn=5) -> dict:
        pred = self.dc.predict(home, away, neutral, topn=topn)
        p_dc = [pred["p_home"], pred["p_draw"], pred["p_away"]]
        p_el = self.elo_1x2(home, away, neutral)
        pb = blend_1x2(p_dc, p_el, self.w, self.T)
        pred["p_home"], pred["p_draw"], pred["p_away"] = float(pb[0]), float(pb[1]), float(pb[2])
        pred["p_dc"] = p_dc          # 1X2 grezzo del solo DC (per confronto)
        return pred


def load_config(path) -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {"ensemble_w": 1.0, "temperature": 1.0}


def save_config(path, cfg: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg, indent=2))
