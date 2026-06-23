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

## Technical terms explained

The names that show up in the code and the table below — what each one actually
means, in one or two sentences.

**The models**

- **Poisson distribution** — the standard way to model "how many times does a rare
  event happen". Goals fit it well: if a team is expected to score 1.5 goals, the
  Poisson tells you the chance of 0, 1, 2, 3… goals.
- **Dixon-Coles model** — a famous 1997 football model. It gives each team an
  **attack** number and a **defence** number, uses Poisson to turn those into goal
  counts for both sides, and adds a small correction for low scores (0–0, 1–0, 1–1
  happen a bit more often than plain Poisson predicts). Output: a probability for
  every scoreline. "Bivariate Poisson" just means "Poisson for both teams at once".
- **Elo rating** — the chess rating system. Every team has a single number; after a
  match the winner takes points from the loser, more if it was an upset. Great for
  win/draw/loss odds, but it doesn't predict scorelines.
- **Ensemble / blend** — using two models and mixing their answers, because the
  average of two decent models is usually better than either alone. Here: `w` parts
  Dixon-Coles + `(1−w)` parts Elo for the win/draw/loss odds.

**The tuning tricks**

- **Ridge / shrinkage** — pulls each team's strength gently toward the average. Stops
  the model from over-trusting tiny samples (a minnow that won one freak game doesn't
  get rated world-class).
- **Time decay / half-life (~3 years)** — recent matches count more than old ones. A
  "3-year half-life" means a result from 3 years ago counts half as much as a recent
  one.
- **Temperature scaling / calibration** — a final dial that stretches or squashes the
  probabilities so they're honest. Without it, models tend to sound overconfident
  (saying 90% when the truth is 75%).
- **Shin de-vig / overround** — bookmaker odds include the bookie's profit margin
  ("the vig"), so their implied percentages add up to more than 100%. "De-vigging"
  removes that margin to recover the fair probabilities; Shin's method is one
  standard way to do it. Only used with `--odds`.
- **Confederation effects** — a small adjustment for the fact that every World Cup
  game is between different continents (UEFA, CONMEBOL, CAF…), whose teams rarely
  play each other, so their strength numbers need a cross-continent correction.

**The score-keeping metrics** (all "lower = better")

- **Accuracy** — simplest one: how often the most likely outcome actually happened.
  Ignores *how* confident the model was.
- **RPS (Ranked Probability Score)** — the football-standard metric. Like Brier but
  it knows the outcomes are *ordered* (home → draw → away), so predicting a draw when
  the away team wins is punished less than predicting a home win. Measures how close
  the probabilities were to reality.
- **log-loss** — punishes confident wrong calls very harshly (saying 99% for
  something that didn't happen is almost infinitely bad). Rewards being both right
  and well-calibrated.
- **Brier score** — the average squared error between the predicted probability and
  what happened (1 or 0). Simple and robust.
- **ECE (Expected Calibration Error)** — measures honesty specifically: of all the
  times the model said "70%", did it happen ~70% of the time? An ECE of 1.9% means
  it's off by less than 2 percentage points on average.
- **Confidence interval (CI) / bootstrap** — a range showing how sure we are of a
  result given limited data. "CI entirely below zero" means the blend's advantage is
  real, not luck. "Bootstrap" is the technique used to compute it (re-sampling the
  matches many times).

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

> All four metrics (RPS, log-loss, Brier, ECE) are explained in
> [Technical terms](#technical-terms-explained) above. Short version: lower means
> the probabilities were sharper and more honest.

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
