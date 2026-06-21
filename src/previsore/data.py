"""Download e caricamento dati sorgente (martj42/international_results, CC0).

Una sola sorgente fa sia da training (storico) sia da elenco fixture:
le partite future hanno score = "NA" (caricate come NaN).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
GOALS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"

# data/ nella root del progetto, override con env PREVISORE_DATA
DATA_DIR = Path(os.environ.get("PREVISORE_DATA", Path(__file__).resolve().parents[2] / "data"))


def _download(url: str, dest: Path, timeout: int = 30) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    dest.write_bytes(r.content)


def update(verbose: bool = True) -> None:
    """Scarica/aggiorna i CSV sorgente. Un solo `git`-style pull, zero API key."""
    for url, name in ((RESULTS_URL, "results.csv"), (GOALS_URL, "goalscorers.csv")):
        dest = DATA_DIR / name
        if verbose:
            print(f"↓ {url}\n  → {dest}")
        _download(url, dest)
    if verbose:
        print("OK")


def load_results(auto_update: bool = True) -> pd.DataFrame:
    """Carica results.csv (scaricandolo alla prima esecuzione)."""
    path = DATA_DIR / "results.csv"
    if not path.exists():
        if not auto_update:
            raise FileNotFoundError(f"{path} assente. Esegui `previsore update`.")
        update(verbose=True)
    df = pd.read_csv(path, parse_dates=["date"])
    for c in ("home_score", "away_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")  # "NA" -> NaN
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    return df


def load_goalscorers(auto_update: bool = True) -> pd.DataFrame:
    """Carica goalscorers.csv (chi ha segnato, minuto, autogol, rigore)."""
    path = DATA_DIR / "goalscorers.csv"
    if not path.exists():
        if not auto_update:
            raise FileNotFoundError(f"{path} assente. Esegui `previsore update`.")
        update(verbose=True)
    g = pd.read_csv(path, parse_dates=["date"])
    for c in ("own_goal", "penalty"):
        if c in g.columns:
            g[c] = g[c].astype(str).str.upper().eq("TRUE")
    return g


def played(df: pd.DataFrame) -> pd.DataFrame:
    """Partite gia disputate (score noto)."""
    return df[df["home_score"].notna()].copy()


def future(df: pd.DataFrame) -> pd.DataFrame:
    """Fixture future (score = NA), incluse le partite del Mondiale 2026."""
    return df[df["home_score"].isna()].copy()
