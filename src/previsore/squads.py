"""Rose 26 reali del Mondiale 2026 da Wikipedia, per filtrare i marcatori.

Scarica e associa ogni tabella-rosa alla nazionale dall'intestazione precedente.
Se il parsing rende troppe poche squadre, tiene la cache esistente (robusto ai
cambi di markup durante il torneo). Tutto opt-in: senza cache, nessun gate.
"""
from __future__ import annotations

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .data import DATA_DIR
from .scorers import _norm_name, _tokens

WIKI_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
SQUADS_CSV = DATA_DIR / "squads.csv"
_HEADERS = {"User-Agent": "Previsore/0.1 (research; football model)"}

# differenze nome heading-Wikipedia -> nome martj42 (lato normalizzato)
_ALIAS = {"czechia": "czech republic", "korea republic": "south korea",
          "ir iran": "iran", "turkiye": "turkey", "türkiye": "turkey"}


def fetch_squads(verbose: bool = True) -> pd.DataFrame:
    r = requests.get(WIKI_URL, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    rows, team = [], None
    for el in soup.select("h2, h3, table.wikitable"):
        if el.name in ("h2", "h3"):
            team = re.sub(r"\[edit\]", "", el.get_text(" ", strip=True)).strip()
        elif el.name == "table" and team:
            header = el.find("tr")
            if header is None:
                continue
            ths = [th.get_text(strip=True) for th in header.find_all(["th", "td"])]
            pidx = next((i for i, c in enumerate(ths) if "Player" in c), None)
            if pidx is None:
                continue
            for tr in el.find_all("tr")[1:]:
                cells = tr.find_all(["td", "th"])
                if len(cells) <= pidx:
                    continue
                name = re.sub(r"\(.*?\)", "", cells[pidx].get_text(" ", strip=True)).strip()
                if name:
                    rows.append((team, name))
    df = pd.DataFrame(rows, columns=["team", "player"]).drop_duplicates()
    if verbose:
        print(f"rose: {df['team'].nunique()} nazionali, {len(df)} giocatori")
    return df


def update_squads(verbose: bool = True) -> pd.DataFrame:
    df = fetch_squads(verbose=verbose)
    if df["team"].nunique() < 24 and SQUADS_CSV.exists():
        if verbose:
            print("parsing scarso: tengo la cache precedente")
        return pd.read_csv(SQUADS_CSV)
    SQUADS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SQUADS_CSV, index=False)
    return df


def load_squads() -> pd.DataFrame | None:
    return pd.read_csv(SQUADS_CSV) if SQUADS_CSV.exists() else None


def tokens_by_team(df: pd.DataFrame | None = None) -> dict:
    """{nome_nazionale_martj42_normalizzato -> set di token cognome}.

    Le chiavi sono gia normalizzate verso la grafia martj42 (via _ALIAS sul nome
    Wikipedia), cosi il lookup da cli usa direttamente il nome martj42.
    """
    if df is None:
        df = load_squads()
    if df is None or len(df) == 0:
        return {}
    out: dict = {}
    for team, grp in df.groupby("team"):
        wiki_key = _norm_name(team)
        key = _ALIAS.get(wiki_key, wiki_key)   # heading Wikipedia -> nome martj42
        toks = out.setdefault(key, set())
        for p in grp["player"]:
            toks |= _tokens(p)
    return out


def squad_tokens_for(team: str, tmap: dict):
    """Token-rosa per una nazionale martj42 (chiavi gia canonicalizzate). None se assente."""
    return tmap.get(_norm_name(team))
