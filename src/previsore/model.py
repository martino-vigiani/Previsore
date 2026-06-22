"""Modello Dixon-Coles: Poisson bivariata con correzione bassi punteggi.

Per la partita (casa i, trasferta j):
    lambda = exp(attacco_i - difesa_j + gamma * campo_casa)
    mu     = exp(attacco_j - difesa_i)
con gamma = vantaggio campo (annullato su campo neutro, tipico al Mondiale),
correzione Dixon-Coles rho sui punteggi 0-0/1-0/0-1/1-1, decadimento temporale
e peso per importanza torneo. Identificabilita: sum(attacco) = 0.

Fit via massima verosimiglianza pesata (scipy L-BFGS-B). Solo CPU: secondi/minuti.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import confed


def _importance(tournament: str) -> float:
    """Peso partita: le amichevoli pesano meno dei tornei veri."""
    t = str(tournament).lower()
    if "friendly" in t:
        return 0.5
    if "world cup" in t and "qual" not in t:
        return 1.0
    if "qualif" in t:
        return 0.8
    if any(s in t for s in ("euro", "copa", "african cup", "asian cup",
                            "gold cup", "nations league", "confederations")):
        return 0.9
    return 0.7


_LOGFACT: np.ndarray | None = None


def _logfact(n: int) -> np.ndarray:
    global _LOGFACT
    if _LOGFACT is None or len(_LOGFACT) < n + 1:
        _LOGFACT = np.array([math.lgamma(k + 1) for k in range(n + 1)])
    return _LOGFACT


def _dc_tau(h, a, lam, mu, rho):
    """Correzione Dixon-Coles sui punteggi bassi (vettoriale)."""
    tau = np.ones_like(lam, dtype=float)
    m = (h == 0) & (a == 0); tau[m] = 1.0 - lam[m] * mu[m] * rho
    m = (h == 0) & (a == 1); tau[m] = 1.0 + lam[m] * rho
    m = (h == 1) & (a == 0); tau[m] = 1.0 + mu[m] * rho
    m = (h == 1) & (a == 1); tau[m] = 1.0 - rho
    return tau


@dataclass
class DixonColes:
    teams: list = field(default_factory=list)
    attack: dict = field(default_factory=dict)
    defence: dict = field(default_factory=dict)
    gamma: float = 0.0      # vantaggio campo (scala log)
    rho: float = 0.0        # correzione Dixon-Coles
    conf: dict = field(default_factory=dict)   # offset forza per confederazione
    half_life_days: float = 730.0
    window_years: int = 12
    min_matches: int = 25
    ref_date: str = ""

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, played: pd.DataFrame, ref_date=None, half_life_days: float = 1095.0,
            window_years: int = 12, min_matches: int = 15, reg: float = 1.0,
            maxiter: int = 300, verbose: bool = True) -> "DixonColes":
        df = played.dropna(subset=["home_score", "away_score"]).copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        ref = pd.Timestamp(ref_date) if ref_date is not None else df["date"].max()
        cutoff = ref - pd.DateOffset(years=window_years)
        df = df[(df["date"] >= cutoff) & (df["date"] <= ref)]

        # tieni solo squadre con abbastanza partite nella finestra
        counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        keep = set(counts[counts >= min_matches].index)
        df = df[df["home_team"].isin(keep) & df["away_team"].isin(keep)]
        if df.empty:
            raise ValueError("Nessuna partita dopo i filtri: allenta window/min_matches.")

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        N = len(teams)

        h = df["home_team"].map(idx).to_numpy()
        a = df["away_team"].map(idx).to_numpy()
        hg = df["home_score"].to_numpy()
        ag = df["away_score"].to_numpy()
        home_flag = (~df["neutral"].to_numpy().astype(bool)).astype(float)

        # indice confederazione per squadra (6 = sconosciuta -> effetto 0)
        NC = len(confed.CONFEDERATIONS)
        cmap = {c: i for i, c in enumerate(confed.CONFEDERATIONS)}
        team_conf = {t: cmap.get(confed.conf_of(t), NC) for t in teams}
        ch = np.array([team_conf[t] for t in df["home_team"]])
        ca = np.array([team_conf[t] for t in df["away_team"]])

        age_days = (ref - df["date"]).dt.days.to_numpy().astype(float)
        xi = math.log(2) / half_life_days
        w = np.exp(-xi * age_days) * df["tournament"].map(_importance).to_numpy()

        # parametri: attacco[0..N-2] liberi (attacco[N-1] = -somma), difesa[0..N-1],
        # gamma, rho, confederazione[0..NC-2] liberi (conf[NC-1] = -somma)
        def unpack(p):
            atk = np.empty(N)
            atk[:N - 1] = p[:N - 1]
            atk[N - 1] = -p[:N - 1].sum()
            dfc = p[N - 1:2 * N - 1]
            gamma, rho = p[2 * N - 1], p[2 * N]
            cf = np.empty(NC)
            cf[:NC - 1] = p[2 * N + 1:2 * N + NC]
            cf[NC - 1] = -p[2 * N + 1:2 * N + NC].sum()
            cf_full = np.append(cf, 0.0)        # indice NC = sconosciuta -> 0
            return atk, dfc, gamma, rho, cf, cf_full

        def nll(p):
            atk, dfc, gamma, rho, cf, cf_full = unpack(p)
            conf_term = cf_full[ch] - cf_full[ca]   # 0 entro la stessa confederazione
            lam = np.clip(np.exp(atk[h] - dfc[a] + gamma * home_flag + conf_term), 1e-8, 50.0)
            mu = np.clip(np.exp(atk[a] - dfc[h] - conf_term), 1e-8, 50.0)
            tau = np.clip(_dc_tau(hg, ag, lam, mu, rho), 1e-10, None)
            ll = w * (hg * np.log(lam) - lam + ag * np.log(mu) - mu + np.log(tau))
            # ridge (L2): pooling parziale verso la media, modella anche le minnow
            penalty = reg * (np.sum(atk ** 2) + np.sum((dfc - dfc.mean()) ** 2) + np.sum(cf ** 2))
            return -ll.sum() + penalty

        x0 = np.zeros(2 * N + NC)
        x0[2 * N - 1] = 0.25   # gamma iniziale
        x0[2 * N] = -0.05      # rho iniziale
        bounds = ([(-3, 3)] * (N - 1) + [(-3, 3)] * N + [(-0.5, 1.5), (-0.2, 0.2)]
                  + [(-2, 2)] * (NC - 1))

        t0 = time.time()
        # La verosimiglianza e quasi piatta vicino all'ottimo (molte squadre con
        # pochi dati), quindi il solver tipicamente si ferma per budget invece che
        # per gradiente: e benigno, le stime sono stabili. Manteniamo il fit rapido.
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "maxfun": maxiter * 60,
                                "ftol": 1e-8, "gtol": 1e-6})
        atk, dfc, gamma, rho, cf, _ = unpack(res.x)
        if verbose:
            print(f"fit: {len(df)} partite, {N} squadre, nll={res.fun:.1f}, "
                  f"gamma={gamma:.3f}, rho={rho:.3f}, iter={res.nit}, {time.time() - t0:.1f}s")

        return cls(
            teams=teams,
            attack={t: float(atk[i]) for t, i in idx.items()},
            defence={t: float(dfc[i]) for t, i in idx.items()},
            gamma=float(gamma), rho=float(rho),
            conf={confed.CONFEDERATIONS[i]: float(cf[i]) for i in range(NC)},
            half_life_days=half_life_days, window_years=window_years,
            min_matches=min_matches, ref_date=str(ref.date()),
        )

    # -------------------------------------------------------------- predict
    def rates(self, home: str, away: str, neutral: bool = False):
        for t in (home, away):
            if t not in self.attack:
                raise ValueError(
                    f"Squadra '{t}' assente dal modello (servono >= {self.min_matches} "
                    f"partite negli ultimi {self.window_years} anni).")
        hf = 0.0 if neutral else 1.0
        ct = self.conf.get(confed.conf_of(home), 0.0) - self.conf.get(confed.conf_of(away), 0.0)
        lam = math.exp(self.attack[home] - self.defence[away] + self.gamma * hf + ct)
        mu = math.exp(self.attack[away] - self.defence[home] - ct)
        # cap per evitare lambda assurdi (es. vs minnow) che la griglia troncherebbe
        return min(lam, 12.0), min(mu, 12.0)

    def score_matrix(self, home: str, away: str, neutral: bool = False, maxgoals: int | None = None):
        lam, mu = self.rates(home, away, neutral)
        if maxgoals is None:  # griglia adattiva: copre la coda anche per punteggi alti
            top = max(lam, mu)
            maxgoals = int(max(10, min(40, math.ceil(top + 6 * math.sqrt(top)))))
        lf = _logfact(maxgoals)[:maxgoals + 1]
        ks = np.arange(maxgoals + 1)
        ph = np.exp(-lam + ks * math.log(lam) - lf)
        pa = np.exp(-mu + ks * math.log(mu) - lf)
        M = np.outer(ph, pa)            # M[x, y] = P(casa x, ospite y)
        M[0, 0] *= 1 - lam * mu * self.rho
        M[0, 1] *= 1 + lam * self.rho
        M[1, 0] *= 1 + mu * self.rho
        M[1, 1] *= 1 - self.rho
        M = np.clip(M, 0.0, None)
        M /= M.sum()
        return M, lam, mu

    def predict(self, home: str, away: str, neutral: bool = False,
                maxgoals: int | None = None, topn: int = 5) -> dict:
        M, lam, mu = self.score_matrix(home, away, neutral, maxgoals)
        p_home = float(np.tril(M, -1).sum())   # casa > ospite
        p_draw = float(np.trace(M))
        p_away = float(np.triu(M, 1).sum())     # casa < ospite
        order = np.argsort(M.ravel())[::-1]
        xy = np.dstack(np.unravel_index(order, M.shape))[0]
        top = [((int(x), int(y)), float(M[x, y])) for x, y in xy[:topn]]
        bx, by = top[0][0]
        # mercati derivati (gratis, dalla stessa matrice)
        gx = np.arange(M.shape[0])
        totals = np.add.outer(gx, gx)
        over25 = float(M[totals >= 3].sum())
        btts = float(M[1:, 1:].sum())          # entrambe segnano
        markets = {
            "over25": over25, "under25": 1.0 - over25,
            "btts": btts, "btts_no": 1.0 - btts,
            "clean_home": float(M[:, 0].sum()),   # ospite non segna
            "clean_away": float(M[0, :].sum()),   # casa non segna
            "dc_1x": p_home + p_draw, "dc_12": p_home + p_away, "dc_x2": p_draw + p_away,
            "exp_total": lam + mu,
        }
        return {
            "home": home, "away": away, "neutral": neutral,
            "lambda_home": lam, "lambda_away": mu,
            "exact": (bx, by), "exact_prob": top[0][1],
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "top_scores": top, "markets": markets,
        }

    # ---------------------------------------------------------- persistence
    def to_json(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "teams": self.teams, "attack": self.attack, "defence": self.defence,
            "gamma": self.gamma, "rho": self.rho, "conf": self.conf,
            "half_life_days": self.half_life_days,
            "window_years": self.window_years, "min_matches": self.min_matches,
            "ref_date": self.ref_date,
        }))

    @classmethod
    def from_json(cls, path) -> "DixonColes":
        return cls(**json.loads(Path(path).read_text()))
