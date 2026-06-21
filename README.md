# Previsore

Predittore **risultato esatto + 1X2** per partite di calcio internazionali
(Mondiale 2026 e oltre). Modello statistico calibrato **Dixon-Coles** (Poisson
bivariata) con baseline **Elo**. Tutto su **CPU**, dati **CC0**, gira **offline**
dopo il primo download. Nessuna API key.

> Onestà: il calcio è basso punteggio e ad alta varianza. Il risultato esatto si
> azzecca **~1 volta su 8** anche coi modelli migliori. L'output va letto come
> *distribuzione di probabilità calibrata + risultato più probabile*, non come "il
> risultato giusto". Vedi i numeri di backtest qui sotto.

## Architettura

```
[dati martj42/international_results, CC0]   ← una sola sorgente: storico + fixture future (score=NA)
        │
        ▼
[feature: forza attacco/difesa per squadra, vantaggio campo, decadimento temporale, peso torneo]
        │
        ▼
[modello base Dixon-Coles]  →  matrice 11×11 delle probabilità di ogni risultato
        │
        ├─ risultato esatto più probabile  (argmax)
        ├─ 1 / X / 2  (somma triangoli della matrice)
        └─ top-N risultati con probabilità
```

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
previsore fit                                     # addestra e salva il modello (~6s)
previsore predict --home Spain --away France --neutral
previsore predict --home Spain --away France --neutral --scorers   # + marcatori probabili
previsore predict --upcoming --limit 8            # prossime fixture del Mondiale
previsore backtest --cutoff 2024-01-01            # validazione temporale
```

Esempio output:

```
  Spain 1-1 France  (campo neutro)
  gol attesi:  1.34 - 1.05
  1X2:         1=42.6%   X=28.8%   2=28.6%
  esatto piu probabile: 1-1 (13.6%)
  top risultati: 1-1 13.6%, 1-0 11.6%, 0-0 9.9%, 0-1 8.9%, 2-1 8.6%
```

## Backtest (numeri reali, fuori campione)

Split temporale dal 2024-01-01, 2507 partite di test:

| Metrica            | Dixon-Coles | Elo    | Riferimento        |
|--------------------|-------------|--------|--------------------|
| RPS (↓ meglio)     | **0.168**   | 0.173  | bookmaker ~0.20    |
| Accuratezza 1X2    | **59.2%**   | 58.7%  | tetto ~52–58%      |
| Risultato esatto   | **12.8%**   | —      | tetto reale ~9–15% |

Il modello batte la baseline Elo su tutte le metriche e sta dentro i tetti teorici.

## Autonomia (cron)

`scripts/daily.sh` fa pull dati → riaddestra → previsioni. Pianifica con crontab:

```
0 8 * * * /Users/martinovigiani/lab/Previsore/scripts/daily.sh >> /tmp/previsore.log 2>&1
```

## Prestazioni su Apple Silicon (M5)

Compute **non** è il collo di bottiglia: fit completo (~11k partite, ~217 squadre)
in **~6 secondi su CPU**, niente GPU, RAM trascurabile. Il vero limite è la qualità
dei dati (formazioni/infortuni last-minute), non la potenza di calcolo.

## Marcatori (`--scorers`)

Euristica gratis: i gol attesi di squadra (lambda dal modello) vengono distribuiti
tra i giocatori per quota storica di gol (pesata per recency), poi
`p_marcatore = 1 - exp(-lambda_giocatore)`. Esempio:

```
  marcatori casa (Spain): Mikel Oyarzabal 19%, Mikel Merino 15%, Ferran Torres 14%, ...
  marcatori ospite (France): Kylian Mbappé 31%, Randal Kolo Muani 8%, Adrien Rabiot 7%, ...
```

## Limiti noti / prossimi passi

- **Marcatori** = stima a livello di *rosa recente*, non di XI confermato: senza la
  formazione titolare (esce ~1h pre-partita) include potenziali ritirati. Per il salto
  di qualità serve un feed formazioni (es. API-Football). Il marcatore esatto resta ≈ fortuna.
- Nomi marcatori non normalizzati: la sorgente a volte duplica per accenti
  (`Julián Álvarez` vs `Julián Alvarez`).
- Backtest a split singolo (no walk-forward refit per partita): leggermente ottimista.
- Niente quote bookmaker → nessun calcolo di *value bet*.
- Normalizzazione nomi squadre minima (former_names non applicato).
- Layer LLM per aggiustamenti last-minute + spiegazione: non ancora presente.

## Dati & licenza

Dati: [martj42/international_results](https://github.com/martj42/international_results) (CC0).
Codice: vedi repo.
