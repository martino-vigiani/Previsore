# Previsore

**Exact-score + 1X2 + goalscorer** predictor for international football matches
(2026 World Cup and beyond). An **ensemble** of Dixon-Coles (bivariate Poisson) +
Elo, with ridge shrinkage and temperature calibration, plus an **optional market
anchor** (bookmaker odds, Shin de-vig). Monochrome `minimal-swiss` terminal output
+ a shareable SVG card. Runs on **CPU**, on **CC0** data, **offline by default**
(odds and real squads are opt-in). No API key required for the core to work.

> Honesty: football is low-scoring and high-variance. The single most likely exact
> score is right only **~1 in 7** even for the best models. Read the output as a
> *calibrated probability distribution + most likely score*, not as "the correct
> score". See the out-of-sample numbers below.

## Architecture

```
[data: martj42/international_results, CC0]   ← single source: history + future fixtures (score=NA)
        │
        ▼
[features: per-team attack/defence strength, home advantage, time decay (~3y), tournament weight]
        │
        ├─► [Dixon-Coles + ridge]  →  11×11 scoreline matrix ──► exact score + top-N + scorers
        │            │ 1X2
        │            ▼
        └─► [Elo]    [ ensemble  w·DC + (1−w)·Elo ]  ──► [temperature scaling] ──► calibrated 1X2
```

Weights `w` (≈0.7) and temperature `T` (≈0.85) are tuned **out-of-sample** on ~3,500
matches (2023→) by log-loss, never on the few World Cup games. Exact score and
scorers come from Dixon-Coles; the ensemble only improves the 1X2 probabilities.

Deliberate choice: the base model is statistical and calibrated, **not** an LLM
guessing numbers (LLMs are poorly calibrated for this — in a June 2026 test Copilot
went 0/4 on scorelines). An LLM layer only makes sense to adjust last-minute inputs
(lineups, injuries) and to write the human-readable explanation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
previsore update                                  # download/refresh data (CC0)
previsore fit                                     # train + tune the blend (~17s)
previsore squads                                  # (opt-in) real 26-man squads from Wikipedia
previsore odds                                    # (opt-in) odds, needs PREVISORE_ODDS_API_KEY
previsore predict --home Spain --away Germany --neutral --scorers
previsore predict --home Spain --away Germany --neutral --odds      # anchor to the market
previsore predict --home Spain --away Germany --neutral --scorers --card card.svg
previsore predict --upcoming --limit 8 --scorers  # only fixtures from today on
previsore evaluate --scorers                       # predictions vs ALREADY played matches
previsore walkforward                              # honest validation over ~7,800 matches
previsore backtest --cutoff 2024-01-01             # quick single-split backtest
```

Output (`minimal-swiss`, monochrome, NO_COLOR-safe):

```
  ────────────────────────────────────────────────────────────

  SPAIN  ·  GERMANY                      neutral · group stage
  FIFA World Cup 2026                            blend + market

  ────────────────────────────────────────────────────────────

  the line               1             X              2
                       48.5%         26.9%          24.6%
                  █████████████████████▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░

  expected goals  1.76  ·  1.22
  likely score    1–1                            ~11% · 1 in 9

  also            2–1    9.6   1–0    8.4

  goals o/u 2.5   over 51%  ·  under 49%
  both score      yes 58%  ·  no 42%
  double chance   1X 78%   12 73%   X2 50%
  clean sheet     spain 30%   germany 22%

  scorers · spain                   scorers · germany
    Mikel Oyarzabal  (p)  28%       Kai Havertz      (p)  19%
```

The score is labelled `likely score ~11% · 1 in 9`: it is the single most likely
outcome, **not** a certainty (football is too variable for that). The **derived
markets** (O/U, BTTS, double chance, clean sheet) come from the same matrix, for
free, and are well calibrated. `evaluate`/`walkforward` only train on earlier data
(no leakage).

## Validation (real, out-of-sample)

**Walk-forward, yearly refit, 7,875 international matches (2018→2026):**

| Metric              | **Blend** | Dixon-Coles | Elo    |
|---------------------|-----------|-------------|--------|
| 1X2 accuracy        | **60.4%** | 60.3%       | 58.5%  |
| RPS (↓ better)      | **0.167** | 0.167       | 0.175  |
| log-loss (↓)        | **0.859** | 0.862       | 0.898  |
| Brier (↓)           | **0.505** | 0.507       | 0.530  |
| ECE calibration (↓) | **1.86%** | 2.24%       | 6.21%  |

(With confederation effects: vs without, log-loss 0.867 → 0.859, RPS 0.169 → 0.167.)

CI95 of log-loss (blend − Elo) = **[−0.035, −0.026]**, entirely below zero → the
blend is **significantly** better. For reference, a public WC2026 model (Hicruben):
RPS 0.175, log-loss 0.89, ECE 2.3% on 763 walk-forward matches — here on 10× more.

**Diagnostic on the 2026 World Cup** (already-played matches, `previsore evaluate`):
1X2 accuracy ~60–64% (blend), exact score ~10–14%, scorers top-1 ~28–33% / top-3
~57–60%. The CI at n≈40 is wide: this is a sanity check, not proof — the proof is
the walk-forward.

> What was deliberately NOT added, and why: recent form, rest days, fixture
> congestion. A ~8,000-match replication shows they add ~0 RPS once team strength is
> modelled. Better not to add noise.

## Goalscorers (`--scorers`)

Team expected goals (lambda) are split across players by their time-decayed
historical share, **without renormalizing** over the gate survivors (so the
unattributed mass = squad depth, instead of inflating the leaders); a per-player
cap and the penalty fraction routed to the penalty taker `(p)`. With `previsore
squads` the pool is gated to the real 26-man roster. Result on WC2026: predicted
total scoring mass 71 vs 72 actual, scorer ECE ~1.2%.

## What's implemented

- Ensemble DC + Elo on the 1X2, weight tuned out-of-sample (CI below zero = significant).
- ~3-year half-life + ridge shrinkage → models minnows too (217 → 234 teams).
- Temperature scaling → ECE 6.2% (Elo) → ~1.9% (blend).
- **Market anchor** (`--odds`): Shin de-vig of the odds + linear pool; opt-in via
  `PREVISORE_ODDS_API_KEY` or `data/odds.csv`, model-only fallback. Only the 1X2 is anchored.
- **Real 26-man squad gate** (`previsore squads`, Wikipedia) + **penalty taker** routing.
- Scorer names de-duplicated by accents (`Álvarez`/`Alvarez`).
- **Confederation effects** (UEFA/CONMEBOL/CONCACAF/CAF/AFC/OFC): cross-confederation
  strength offset (every WC match is cross-confederation). Sensible offsets
  (CONMEBOL +0.73, UEFA +0.51, OFC −0.93), log-loss 0.867 → 0.859.
- **Derived markets** (O/U 2.5, BTTS, double chance, clean sheet) from the matrix.
- Score reframed as "1 in N" (no false certainty); adaptive grid + λ clip for blowouts.
- `minimal-swiss` monochrome output + `--card` SVG export; `walkforward`/`evaluate`
  with log-loss/Brier/ECE + bootstrap CIs; tests in `tests/`.

## Known limits / next steps

- **Market weight** cannot be tuned on history (no free historical international odds):
  fixed default `w=0.5`; revisit if a historical odds CSV is supplied.
- Exact scorer stays ≈ luck; the squad gate is at call-up level, not confirmed XI.
- Live lineups/injuries (the gap to bookmakers) and an LLM explanation layer: not yet.

## Automation (cron)

`scripts/daily.sh` pulls data → retrains → predicts. Schedule with crontab, e.g.:

```
0 8 * * * /path/to/Previsore/scripts/daily.sh >> /tmp/previsore.log 2>&1
```

## Performance (Apple Silicon, M-series)

Compute is **not** the bottleneck: fit + blend tuning (~11k matches, ~234 teams) in
**~17 s on CPU**; the walk-forward over 7,875 matches (8 refits) in **~50 s**. No GPU,
negligible RAM. The real limit is data quality (last-minute lineups/injuries), not
compute.

## Disclaimer

This is an **educational / entertainment** tool. It is **not** betting advice and not
a guarantee of any result: football is high-variance and the exact score is right
about 1 in 7 even for the best models. Predictions are probability distributions, not
certainties. If you gamble, do so responsibly and at your own risk. The authors are
not liable for any losses arising from use of this software.

## Data & license

- Code: **MIT** (see [LICENSE](LICENSE)).
- Results/scorers: [martj42/international_results](https://github.com/martj42/international_results) — CC0.
- Fixtures/venues: [openfootball](https://github.com/openfootball) — CC0.
- 2026 squads: Wikipedia — **CC-BY-SA** (attribution due; fetched with a descriptive
  User-Agent and cached, without hammering the site).
- Odds (optional): [the-odds-api](https://the-odds-api.com) with your own key.

Odds/squads are **opt-in**; by default the app uses only CC0 data and runs offline.
