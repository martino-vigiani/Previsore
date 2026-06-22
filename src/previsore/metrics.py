"""Metriche di valutazione probabilistica per esiti 1X2 ordinati [H, D, A]."""
from __future__ import annotations

import numpy as np


def rps(p, o: int) -> float:
    """Ranked Probability Score (ordinale, piu basso = meglio)."""
    oo = [0.0, 0.0, 0.0]
    oo[o] = 1.0
    return float(((np.cumsum(p) - np.cumsum(oo))[:2] ** 2).sum() / 2.0)


def log_loss(p, o: int, eps: float = 1e-15) -> float:
    """Log-loss sull'esito realizzato (discrimina meglio dell'RPS)."""
    return float(-np.log(min(max(p[o], eps), 1.0)))


def brier(p, o: int) -> float:
    """Brier score multiclasse."""
    oo = np.zeros(3)
    oo[o] = 1.0
    return float(((np.asarray(p, dtype=float) - oo) ** 2).sum())


def temperature(p, T: float) -> np.ndarray:
    """Temperature scaling: T>1 ammorbidisce, T<1 rende piu netto."""
    lp = np.log(np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)) / T
    e = np.exp(lp - lp.max())
    return e / e.sum()


def ece(probs, outcomes, bins: int = 10) -> float:
    """Expected Calibration Error medio sui tre esiti (binning per classe)."""
    probs = np.asarray(probs, dtype=float)         # (N, 3)
    y = np.zeros_like(probs)
    y[np.arange(len(outcomes)), outcomes] = 1.0
    total = 0.0
    for k in range(3):
        conf = probs[:, k]
        hit = y[:, k]
        edges = np.linspace(0, 1, bins + 1)
        e = 0.0
        for i in range(bins):
            m = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
            if m.sum() == 0:
                continue
            e += (m.sum() / len(conf)) * abs(conf[m].mean() - hit[m].mean())
        total += e
    return total / 3.0
