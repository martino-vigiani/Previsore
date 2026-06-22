# Previsore

Predittore **risultato esatto + 1X2 + marcatori** per partite di calcio
internazionali (Mondiale 2026 e oltre). **Ensemble** Dixon-Coles (Poisson bivariata)
+ Elo, con ridge-shrinkage e calibrazione a temperatura, e **ancoraggio opzionale al
mercato** (quote bookmaker, Shin de-vig). Output `minimal-swiss` monocromo + card SVG
condivisibile. Tutto su **CPU**, dati **CC0**; **offline di default** (quote e rose
reali sono opt-in). Nessuna API key richiesta per il funzionamento base.

> Onestà: il calcio è basso punteggio e ad alta varianza. Il risultato esatto si
> azzecca **~1 volta su 8** anche coi modelli migliori. L'output va letto come
> *distribuzione di probabilità calibrata + risultato più probabile*, non come "il
> risultato giusto". Vedi i numeri di backtest qui sotto.

## Architettura

```
[dati martj42/international_results, CC0]   ← una sola sorgente: storico + fixture future (score=NA)
        │
        ▼
[feature: forza attacco/difesa per squadra, vantaggio campo, decadimento temporale (~3 anni), peso torneo]
        │
        ├─► [Dixon-Coles + ridge]  →  matrice 11×11 dei punteggi ──► risultato esatto + top-N + marcatori
        │            │ 1X2
        │            ▼
        └─► [Elo]    [ ensemble  w·DC + (1−w)·Elo ]  ──► [temperature scaling] ──► 1X2 calibrato
```

Pesi `w` (≈0.7) e temperatura `T` (≈0.85) tarati **out-of-sample** su ~3500 partite
(2023→) per log-loss, mai sulle poche partite del Mondiale. Il risultato esatto e i
marcatori restano dal Dixon-Coles; l'ensemble migliora solo le probabilità 1X2.

Scelta deliberata (vedi feasibility): il modello base è statistico/calibrato, **non**
un LLM che indovina i numeri (gli LLM sono mal-calibrati: in un test giugno 2026
Copilot ha fatto 0/4 sui risultati). Un eventuale layer LLM va aggiunto solo per
aggiustare gli input last-minute (formazioni, infortuni) e scrivere la spiegazione.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Uso

```bash
previsore update                                  # scarica/aggiorna i dati (CC0)
previsore fit                                     # addestra + tara il blend (~17s)
previsore squads                                  # (opt-in) rose 26 reali da Wikipedia
previsore odds                                    # (opt-in) quote, serve PREVISORE_ODDS_API_KEY
previsore predict --home Spain --away Germany --neutral --scorers
previsore predict --home Spain --away Germany --neutral --odds      # ancora al mercato
previsore predict --home Spain --away Germany --neutral --scorers --card card.svg
previsore predict --upcoming --limit 8 --scorers  # solo fixture da oggi in poi
previsore evaluate --scorers                       # predizioni vs partite GIA giocate
previsore walkforward                              # validazione onesta su ~7800 partite
previsore backtest --cutoff 2024-01-01             # backtest split singolo (rapido)
```

Output (`minimal-swiss`, monocromo, NO_COLOR-safe):

```
  ────────────────────────────────────────────────────────────

  SPAIN  ·  GERMANY                      neutral · group stage
  FIFA World Cup 2026                            blend + market

  ────────────────────────────────────────────────────────────

  the line               1             X              2
                       48.5%         26.9%          24.6%
                  █████████████████████▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░

  expected goals  1.76  ·  1.22
  likely score    1–1                            ~11% · 1 su 9

  also            2–1    9.6   1–0    8.4

  goals o/u 2.5   over 51%  ·  under 49%
  both score      yes 58%  ·  no 42%
  double chance   1X 78%   12 73%   X2 50%
  clean sheet     spain 30%   germany 22%

  scorers · spain                   scorers · germany
    Mikel Oyarzabal  (p)  28%       Kai Havertz      (p)  19%
```

Il punteggio è etichettato `likely score ~11% · 1 su 9`: è il singolo esito più
probabile, non una certezza (il calcio è troppo vario perché lo sia). I **mercati
derivati** (O/U, BTTS, doppia chance, clean sheet) vengono dalla stessa matrice,
gratis, e sono ben calibrati.

`evaluate`/`walkforward` riaddestrano solo su dati precedenti (niente leakage).

Esempio output:

```
  Spain 1-1 France  (campo neutro)
  gol attesi:  1.34 - 1.05
  1X2:         1=42.6%   X=28.8%   2=28.6%
  esatto piu probabile: 1-1 (13.6%)
  top risultati: 1-1 13.6%, 1-0 11.6%, 0-0 9.9%, 0-1 8.9%, 2-1 8.6%
```

## Validazione (numeri reali, fuori campione)

**Walk-forward, refit annuale, 7875 partite internazionali (2018→2026):**

| Metrica            | **Blend** | Dixon-Coles | Elo    |
|--------------------|-----------|-------------|--------|
| Accuratezza 1X2    | **60.4%** | 60.3%       | 58.5%  |
| RPS (↓ meglio)     | **0.167** | 0.167       | 0.175  |
| log-loss (↓)       | **0.859** | 0.862       | 0.898  |
| Brier (↓)          | **0.505** | 0.507       | 0.530  |
| ECE calibrazione (↓)| **1.86%**| 2.24%       | 6.21%  |

(con effetti di confederazione: vs senza, log-loss 0.867 → 0.859, RPS 0.169 → 0.167.)

CI95 log-loss (blend − Elo) = **[−0.035, −0.026]**, interamente sotto zero → il blend
è **significativamente** meglio. ECE 1.72% < 5% = ben calibrato (riferimento Hicruben
WC2026: RPS 0.175, log-loss 0.89, ECE 2.3% su 763 partite — qui su 10× più partite).

**Diagnostico sul Mondiale 2026** (36 partite già giocate, `previsore evaluate`):
Acc 1X2 63.9% (blend), risultato esatto ~11–14%, marcatori top-1 ~28%, top-3 ~57%.
Il CI a n=36 è ampio: questo è un controllo, non una prova — la prova è il walk-forward.

> Cosa NON è stato aggiunto e perché: forma recente, giorni di riposo, congestione
> calendario. Una replica su ~8000 partite mostra che danno ~0 guadagno di RPS una
> volta modellata la forza squadra. Meglio non aggiungere rumore.

## Autonomia (cron)

`scripts/daily.sh` fa pull dati → riaddestra → previsioni. Pianifica con crontab:

```
0 8 * * * /Users/martinovigiani/lab/Previsore/scripts/daily.sh >> /tmp/previsore.log 2>&1
```

## Prestazioni su Apple Silicon (M5)

Compute **non** è il collo di bottiglia: fit + taratura blend (~11k partite, ~234
squadre) in **~17 secondi su CPU**; il walk-forward su 7875 partite (8 refit) in
**~50 secondi**. Niente GPU, RAM trascurabile. Il vero limite è la qualità dei dati
(formazioni/infortuni last-minute), non la potenza di calcolo.

## Marcatori (`--scorers`)

Euristica gratis: i gol attesi di squadra (lambda dal modello) vengono distribuiti
tra i giocatori per quota storica di gol (pesata per recency), poi
`p_marcatore = 1 - exp(-lambda_giocatore)`. Esempio:

```
  marcatori casa (Spain): Mikel Oyarzabal 19%, Mikel Merino 15%, Ferran Torres 14%, ...
  marcatori ospite (France): Kylian Mbappé 31%, Randal Kolo Muani 8%, Adrien Rabiot 7%, ...
```

## Già implementato

- Ensemble DC + Elo sul 1X2, peso tarato out-of-sample (CI sotto zero = significativo).
- Half-life ~3 anni + ridge-shrinkage → modella anche le minnow (217 → 234 squadre).
- Temperature scaling → ECE 6.2% (Elo) → 1.7% (blend).
- **Ancoraggio al mercato** (`--odds`): Shin de-vig delle quote + pool lineare; opt-in
  via `PREVISORE_ODDS_API_KEY` o `data/odds.csv`, fallback model-only.
- **Gate rosa-26 reale** (`previsore squads`, Wikipedia) + **rigorista** instradato:
  marcatori dalla rosa convocata, non più dai ritirati.
- Nomi marcatori normalizzati per accenti (`Álvarez`/`Alvarez` deduplicati).
- Output `minimal-swiss` monocromo + export `--card` SVG; `walkforward`/`evaluate`
  con log-loss/Brier/ECE + CI bootstrap; test (`tests/`).
- **Marcatori calibrati**: quote per giocatore senza rinormalizzare sui superstiti
  del gate (la massa non attribuita = profondità rosa) → massa totale 71 vs 72 reali
  su WC2026, ECE 1.2% (prima la fascia 20-35% era gonfiata 2×).
- **Mercati derivati** (O/U 2.5, BTTS, doppia chance, clean sheet) dalla matrice.
- Punteggio riformulato come "1 su N" (non più falsa certezza); griglia adattiva
  + clip λ per i blowout; accenti canonicalizzati; rose con Caps/Goals.
- **Effetti di confederazione** (UEFA/CONMEBOL/CONCACAF/CAF/AFC/OFC): offset di
  forza stimato dalle partite cross-confederation (tutte quelle del Mondiale lo sono).
  Offset sensati (CONMEBOL +0.73, UEFA +0.51, OFC −0.93) e log-loss 0.867 → 0.859.

## Limiti noti / prossimi passi

- **Peso mercato** non tarabile su storico (non esistono quote internazionali storiche
  gratis): default fisso `w=0.5`, da rivedere se si fornisce un CSV storico.
- Marcatore esatto resta ≈ fortuna; il gate rosa è a livello di convocati, non di XI.
- Layer LLM per aggiustamenti last-minute + spiegazione: non ancora presente.

## Disclaimer

Strumento a **scopo educativo e di intrattenimento**. **Non** è un consiglio di
scommessa né una garanzia di risultato: il calcio è ad alta varianza e il risultato
esatto si azzecca circa 1 volta su 7 anche coi modelli migliori. Le previsioni sono
distribuzioni di probabilità, non certezze. Se giochi, fallo responsabilmente e a tuo
rischio. Gli autori non sono responsabili di perdite derivanti dall'uso del software.

## Dati & licenza

- Codice: **MIT** (vedi [LICENSE](LICENSE)).
- Risultati/marcatori: [martj42/international_results](https://github.com/martj42/international_results) — CC0.
- Calendario/sedi: [openfootball](https://github.com/openfootball) — CC0.
- Rose 2026: Wikipedia — **CC-BY-SA** (attribuzione dovuta; scaricate con User-Agent
  descrittivo e cache, senza martellare il sito).
- Quote (opzionale): [the-odds-api](https://the-odds-api.com) con chiave propria.

Le quote/rose sono **opt-in**; di default l'app usa solo dati CC0 e gira offline.
