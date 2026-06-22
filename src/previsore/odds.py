"""Quote bookmaker: Shin de-vig + ancoraggio al mercato (opt-in).

Catena di fallback (degrada senza mai crashare):
  1. PREVISORE_ODDS_API_KEY impostata -> fetch live the-odds-api, cache in data/odds.csv
  2. altrimenti data/odds.csv presente -> usa quello
  3. altrimenti nessuna quota -> peso mercato 0 -> solo modello (resta offline)

Solo il vettore 1X2 viene ancorato al mercato; matrice punteggi, top risultati e
marcatori restano 100% Dixon-Coles.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq

from .data import DATA_DIR

API_URL = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"
ODDS_CSV = DATA_DIR / "odds.csv"

# nomi feed quote -> nomi martj42
_ALIAS = {
    "usa": "United States", "united states of america": "United States",
    "korea republic": "South Korea", "south korea": "South Korea",
    "korea dpr": "North Korea", "ir iran": "Iran", "iran": "Iran",
    "china pr": "China", "ivory coast": "Ivory Coast", "cote d'ivoire": "Ivory Coast",
    "czechia": "Czech Republic", "cape verde islands": "Cape Verde",
    "turkiye": "Turkey", "türkiye": "Turkey",
}


def _norm(s: str) -> str:
    return _ALIAS.get(str(s).strip().lower(), str(s).strip())


# --------------------------------------------------------------- Shin de-vig
def shin_devig(odds) -> list | None:
    """Rimuove il margine bookmaker da 3 quote decimali -> prob. eque [H,D,A]."""
    o = np.asarray(odds, dtype=float)
    if o.shape != (3,) or not np.all(np.isfinite(o)) or np.any(o <= 1.0):
        return None
    pi = 1.0 / o
    B = pi.sum()
    if B <= 1.0 + 1e-12:
        return (pi / B).tolist()

    def p_of_z(z):
        return (np.sqrt(z * z + 4 * (1 - z) * pi * pi / B) - z) / (2 * (1 - z))

    def f(z):
        return p_of_z(z).sum() - 1.0

    hi = 0.999999
    try:
        if f(0.0) * f(hi) > 0:
            return (pi / B).tolist()
        z = brentq(f, 0.0, hi, xtol=1e-12, maxiter=200)
        p = p_of_z(z)
    except Exception:
        return (pi / B).tolist()
    if not np.all(np.isfinite(p)) or p.sum() <= 0:
        return (pi / B).tolist()
    return (p / p.sum()).tolist()


def overround(odds) -> float:
    o = np.asarray(odds, dtype=float)
    return float((1.0 / o).sum())


# ------------------------------------------------------------- sorgente quote
def fetch_odds(regions: str = "eu", verbose: bool = True) -> list[dict]:
    key = os.environ.get("PREVISORE_ODDS_API_KEY")
    if not key:
        return []
    r = requests.get(API_URL, params={"regions": regions, "markets": "h2h",
                                      "oddsFormat": "decimal", "dateFormat": "iso",
                                      "apiKey": key}, timeout=20)
    r.raise_for_status()
    if verbose and "x-requests-remaining" in r.headers:
        print(f"the-odds-api: {r.headers['x-requests-remaining']} crediti residui")
    rows = []
    for ev in r.json():
        h, a = ev.get("home_team"), ev.get("away_team")
        cols = {"home": [], "draw": [], "away": []}
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                d = {o["name"]: o["price"] for o in mk.get("outcomes", [])}
                if h in d and a in d and "Draw" in d:
                    cols["home"].append(d[h]); cols["draw"].append(d["Draw"]); cols["away"].append(d[a])
        if cols["home"]:
            rows.append({"home_team": h, "away_team": a,
                         "commence_time": ev.get("commence_time", ""),
                         "odds_home": float(np.median(cols["home"])),
                         "odds_draw": float(np.median(cols["draw"])),
                         "odds_away": float(np.median(cols["away"])),
                         "n_books": len(cols["home"])})
    return rows


def get_odds_table(refresh: bool = True, verbose: bool = True) -> pd.DataFrame | None:
    """Tabella quote secondo la catena di fallback. None se non disponibili."""
    if refresh and os.environ.get("PREVISORE_ODDS_API_KEY"):
        try:
            rows = fetch_odds(verbose=verbose)
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(ODDS_CSV, index=False)
                return df
        except Exception as e:  # rete giù, 401/429, quota finita...
            if verbose:
                print(f"odds: fetch fallito ({type(e).__name__}), provo la cache")
    if ODDS_CSV.exists():
        return pd.read_csv(ODDS_CSV)
    return None


def lookup_market(table: pd.DataFrame | None, home: str, away: str) -> list | None:
    """Prob. di mercato de-viggate per la partita, o None se assente."""
    if table is None or len(table) == 0:
        return None
    h, a = _norm(home), _norm(away)
    for r in table.itertuples(index=False):
        if _norm(r.home_team) == h and _norm(r.away_team) == a:
            return shin_devig([r.odds_home, r.odds_draw, r.odds_away])
    return None
