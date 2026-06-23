# Previsore

**A football match predictor for international games** (the 2026 World Cup and
beyond). Give it two teams and it tells you, in plain probabilities, who is
likely to win, the most likely scoreline, how many goals to expect, and who is
likely to score.

It is a **statistics model**, not an LLM guessing numbers — so its probabilities
are *calibrated* (when it says 70%, that really happens about 70% of the time).
It runs on your laptop's **CPU**, on **free open data**, and works **offline** by
default.

> **One honest caveat up front:** football is low-scoring and unpredictable. Even
> the best models in the world guess the *exact* final score right only about
> **1 time in 7**. So read a prediction as "here are the odds and the most likely
> score", **not** as "this is what will happen".

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

previsore update                       # download the match data (free, one-off)
previsore fit                          # train the model (~17 seconds)
previsore predict --home Spain --away Germany --neutral
```

That's it. `--neutral` means "no home-team advantage" (true for most World Cup
games, played at neutral venues).

---

## How to read a prediction

Here is a real prediction, annotated:

```
  SPAIN  ·  GERMANY                      neutral · group stage
  FIFA World Cup 2026                            blend + market

  the line               1             X              2
                       48.5%         26.9%          24.6%      ← chance of home win / draw / away win
                  █████████████████████▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░

  expected goals  1.76  ·  1.22                                ← avg goals each team should score
  likely score    1–1                            ~11% · 1 in 9 ← most likely exact score (lands ~1 time in 9)

  also            2–1    9.6   1–0    8.4                       ← next most likely scores

  goals o/u 2.5   over 51%  ·  under 49%                        ← 3+ total goals vs 2 or fewer
  both score      yes 58%  ·  no 42%                            ← do BOTH teams score?
  double chance   1X 78%   12 73%   X2 50%                      ← "home-or-draw / home-or-away / draw-or-away"
  clean sheet     spain 30%   germany 22%                       ← chance each team concedes zero

  scorers · spain                   scorers · germany
    Mikel Oyarzabal  (p)  28%       Kai Havertz      (p)  19%   ← likely scorers; (p) = penalty taker
```

The big takeaway is **the line** (1 / X / 2) and **expected goals**. Everything
below is extra detail computed from the same prediction. The `likely score` is
just the *single* most probable result — useful, but it lands only ~1 in 9 here,
so don't bet the house on it.

---

## Glossary (the jargon, in plain words)

| Term | What it means |
|---|---|
| **1 / X / 2** | Home win / draw / away win. |
| **neutral** | Played at a neutral venue, so no home advantage is applied. |
| **expected goals** | The average number of goals each team "should" score in this match. |
| **likely score · 1 in N** | The single most probable exact score, and how often a score that likely actually happens (e.g. "1 in 9"). Not a prediction of the real score. |
| **o/u 2.5** | Over/Under 2.5 goals: will there be **3 or more** total goals (over) or **2 or fewer** (under)? |
| **both score** | Do **both** teams score at least one goal? (a.k.a. "BTTS".) |
| **double chance** | Two outcomes at once: **1X** = home win or draw, **12** = home or away (anything but a draw), **X2** = draw or away win. |
| **clean sheet** | A team finishes without conceding a goal. |
| **(p)** | That player is the team's penalty taker, so part of their scoring chance comes from penalties. |
| **blend / market** | "blend" = the model's own answer; "blend + market" = the answer nudged toward the bookmaker odds (only if you opt in with `--odds`). |

If you just want a forecast, the table above is all you need. The sections below
are for people who want to know *how* it works and *how well*.

---

## Commands

```bash
previsore update                                   # download/refresh the data (free, CC0)
previsore fit                                      # train the model (~17s)

# predict one match:
previsore predict --home Spain --away Germany --neutral
previsore predict --home Spain --away Germany --neutral --scorers   # add likely scorers
previsore predict --home Spain --away Germany --neutral --odds      # nudge toward bookmaker odds
previsore predict --home Spain --away Germany --neutral --card card.svg  # save a shareable image

# predict many upcoming matches at once:
previsore predict --upcoming --limit 8 --scorers   # next fixtures from today on
previsore predict --upcoming --until 2026-06-24    # only fixtures up to (and including) a date

# optional extras:
previsore squads                                   # real 26-man squads from Wikipedia
previsore odds                                     # bookmaker odds (needs PREVISORE_ODDS_API_KEY)

# check how good it is:
previsore evaluate                                 # compare predictions vs matches ALREADY played
previsore walkforward                              # honest validation over ~7,800 past matches
previsore backtest --cutoff 2024-01-01             # quick single-split test
```

---

## How it works (for the curious)

In one line: it estimates each team's **attack and defence strength** from years
of past results, turns that into a probability for every possible scoreline, and
blends it with a chess-style rating system for sharper win/draw/loss odds.

```
[data: every international result + future fixtures, free/CC0]
        │
        ▼
[for each team: attack strength, defence strength, home advantage,
 recent results weighted more (~3-year memory), tournament importance]
        │
        ├─► [Dixon-Coles model] → table of all scorelines → exact score, top scores, scorers
        │            │ win/draw/loss
        │            ▼
        └─► [Elo ratings]  → blend the two → calibrate → final 1 / X / 2 probabilities
```

- **Dixon-Coles** is a well-known football model: from attack/defence strengths
  it gives a probability to every scoreline (0–0, 1–0, 2–1, …). The win/draw/loss
  odds and the scorers all fall out of that one table.
- **Elo** is the chess-style rating: each team has a number that rises and falls
  after every match; good for win/draw/loss odds.
- **Blend** mixes the two; **calibration** ("temperature") makes sure the stated
  percentages are honest. The blend weights are tuned on past matches only — never
  on the few World Cup games — so the validation below isn't cheating.

Why not an LLM? Large language models are poorly calibrated for this — in a June
2026 test, one popular AI assistant went 0/4 on scorelines. An LLM only helps for
last-minute inputs (lineups, injuries) and for writing the human-readable summary.

---

## How good is it? (validation)

Tested **out-of-sample** — i.e. always predicting matches the model had *not*
seen during training. Lower is better for RPS / log-loss / Brier / ECE; these are
standard scores for "how good were the probabilities" (see note below the table).

**Walk-forward over 7,875 international matches (2018→2026), retrained each year:**

| Metric | **Blend** | Dixon-Coles | Elo |
|---|---|---|---|
| Win/draw/loss accuracy | **60.4%** | 60.3% | 58.5% |
| RPS (↓ better) | **0.167** | 0.167 | 0.175 |
| log-loss (↓) | **0.859** | 0.862 | 0.898 |
| Brier (↓) | **0.505** | 0.507 | 0.530 |
| ECE — calibration error (↓) | **1.86%** | 2.24% | 6.21% |

> **What those scores mean:** **RPS**, **log-loss** and **Brier** all measure how
> close the predicted probabilities were to what actually happened (lower = the
> probabilities were sharper and more honest). **ECE** is the calibration error: a
> ~1.9% ECE means that when the model says "70%", the real rate is within ~2% of
> 70%. **Accuracy** is just how often the most likely outcome was correct.

The blend is **significantly** better than Elo alone (the confidence interval for
the difference sits entirely below zero). For reference, a public WC2026 model
scored RPS 0.175 / log-loss 0.89 on 763 matches — here we get better numbers on
10× more matches.

**On the 2026 World Cup so far** (`previsore evaluate`): win/draw/loss accuracy
~60–64%, exact score right ~10–14% of the time, top scorer guessed ~28–33% of the
time (~57–60% within the top 3). With only ~40 matches this is a sanity check, not
proof — the real proof is the 7,875-match walk-forward above.

> Deliberately **left out**: recent form, rest days, fixture congestion. On ~8,000
> matches these add almost nothing once team strength is modelled — better not to
> add noise.

---

## Likely scorers (`--scorers`)

Each team's expected goals are split among its players by how much each one has
scored historically (recent goals count more). Penalties are routed to the actual
penalty taker, marked `(p)`. With `previsore squads` the list is limited to the
real 26-man World Cup roster (from Wikipedia). On WC2026 the predicted total
scoring matched almost exactly (71 predicted vs 72 actual).

---

## Known limits / next steps

- **Bookmaker odds weight** is a fixed default (there's no free archive of
  historical international odds to tune it on); revisit if such data appears.
- Guessing the *exact* scorer is mostly luck; the squad list is the called-up 26,
  not the confirmed starting XI.
- Live lineups/injuries and an automatic written summary: not done yet.
- Ideas being explored (tournament context, group standings, "dead rubber" and
  collusion effects) are tracked in [`docs/idee-feature.md`](docs/idee-feature.md).

---

## Run it automatically (optional)

`scripts/daily.sh` downloads data → retrains → predicts. Schedule it with cron,
e.g. every morning at 8:

```
0 8 * * * /path/to/Previsore/scripts/daily.sh >> /tmp/previsore.log 2>&1
```

It's fast: training takes ~17 s on a normal CPU, the full 7,875-match validation
~50 s. No GPU needed. The real bottleneck is data quality (last-minute lineups),
not compute.

---

## Disclaimer

This is an **educational / entertainment** tool. It is **not** betting advice and
**not** a guarantee of any result: football is high-variance and the exact score
is right about 1 in 7 even for the best models. Predictions are probabilities, not
certainties. If you gamble, do so responsibly and at your own risk. The authors
are not liable for any losses arising from use of this software.

---

## Data & license

- Code: **MIT** (see [LICENSE](LICENSE)).
- Results/scorers: [martj42/international_results](https://github.com/martj42/international_results) — CC0.
- Fixtures/venues: [openfootball](https://github.com/openfootball) — CC0.
- 2026 squads: Wikipedia — **CC-BY-SA** (attribution due; fetched politely with a
  descriptive User-Agent and cached).
- Odds (optional): [the-odds-api](https://the-odds-api.com) with your own key.

Odds and real squads are **opt-in**; by default the app uses only free CC0 data
and runs offline. No API key is needed for the core to work.
