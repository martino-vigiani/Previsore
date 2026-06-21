"""Previsore — predittore risultato esatto + 1X2 per il Mondiale 2026.

Modello base: Dixon-Coles (Poisson bivariata con correzione bassi punteggi),
forza attacco/difesa per squadra, vantaggio campo, decadimento temporale.
Baseline/confronto: Elo (stile World Football Elo) + modello di pareggio Davidson.

Tutto su CPU, dati CC0 (martj42/international_results).
"""

__version__ = "0.1.0"
