# Decision record

Every choice that shaped the system, with the reason. Kept separate from `PLAN.md` because
the *why* outlives the *what* — a future change that contradicts one of these should have to
argue with the reason, not rediscover it.

> **Read this before quoting any number below — two separate corrections apply.**
>
> **1. The horizon lookahead (fixed 2026-08-12).** Until this date every multi-gameweek
> valuation in the backtest was assembled from forecasts built AFTER the decision it was
> informing: a GW10 transfer was judged on the GW15 forecast, which used rates, form and a
> team model as of GW15. Worth **-392 season points, negative in all three seasons**. Any
> number below that came from a full-season replay dated before 2026-08-12 is inflated by
> roughly 15%, and anything that compares a HORIZON policy against a myopic one is void
> rather than merely inflated — see "The horizon lookahead" near the end.
>
> **2. Three double-gameweek bugs** were found and fixed on 2026-08-10 (see the section near
> the end). Every season total and every simulated rank recorded before that date was
> measured under them.
>
> * **Points totals** are inflated by roughly 6% and their *rankings* mostly survive, but any
>   comparison involving Bench Boost or double-gameweek timing should be treated as unsettled.
> * **All rank figures from before that date are withdrawn.** The bug was propping up the
>   simulated field, and the headline "mean simulated rank 7,089, top-10k target reached" is
>   not supported. The field has since been rebuilt against an exact ownership anchor; ranks
>   quoted after 2026-08-10 come from that model.
>
> Superseded sections are kept rather than deleted — the reasoning is still worth having, and
> two of the conclusions were later reversed by better measurement, which is itself the most
> useful thing in this file.

## Decisions you made

| # | Decision | Consequence |
|---|---|---|
| 1 | **Expected points at the core, rank layer configurable** | Well-posed MILP; rank-awareness is additive rather than a fork |
| 2 | **Season target: top 10k** | Rank layer committed, not optional; ownership data becomes a real dependency |
| 3 | **Start with data ingestion (Item 2)** | Everything else was blocked on it |
| 4 | **Produce a best-effort GW1 forecast** | Cold-start machinery in scope; you get a squad on 21 Aug |
| 5 | **No-history players get a price/position prior + confidence flag** | Coventry and Hull squads are pickable but not over-trusted |
| 6 | **Fully automatic — no manual override channel** | Backtests fully reproducible; Item 5 must extract everything from FPL's own `news`/`chance_of_playing` |
| 7 | **Free data sources only** | FPL API, vaastav archive, football-data.co.uk. No paid odds |
| 8 | **Sprint to end-to-end, then deepen** | A working GW1 recommendation exists; rough parts marked SPRINT |
| 9 | **Finishing skill: shrunk multiplier** | Tested the hypothesis rather than assuming — and the answer was "pure xG" |
| 10 | **Set-piece duty: live API at inference, history for training** | Catches duty changes immediately; backtest understates live performance |
| 11 | **Register the local Windows snapshot task** | Hourly checkpoint ladder; GW1 state will be captured |
| 12 | **Build the simulator before deepening components** | Effort aimed by measurement — and it immediately overturned an assumption |

## Technical decisions I made

**Modelling**
- *Decompose, never regress on total points.* Points are a deterministic function of countable
  events; a single regression fits mostly noise.
- *Minutes as three buckets (0 / 1-59 / 60+), not a regression.* The scoring thresholds are
  discontinuous and the distribution is bimodal — predicting 45 minutes returns the least
  likely outcome.
- *Availability applied outside the model.* `chance_of_playing_next_round` does not exist in
  the archive, so learning it would be train/serve skew in one direction or the other.
- *Players keyed by name, not `element`.* FPL reassigns element ids each season; keying on
  the id makes every player a debutant each August.
- *Attacking returns allocated top-down from team expected goals.* Team totals stay consistent
  with clean sheets, and fixture difficulty is inherited for free.
- *Match model = Dixon-Coles blended with the market at weight 0.8.* Measured, not assumed —
  and the market alone is statistically indistinguishable. The blend is kept for graceful
  degradation when odds are missing (as they are for GW1), **not** as a claimed edge.
- *No clean-sheet recalibration.* The apparent 10pp top-bin bias flips sign across seasons;
  isotonic recalibration made one season worse. It is noise, and fitting it would be fitting
  noise.
- *Per-stat noise scale, measured.* xG carries 0.29 of Poisson noise, xA 0.17, goals 0.92.
- *Kish rescaling for decayed counts.* Poisson variance scales with `w^2`, not `w`.

**Point-in-time discipline**
- *`PointInTime` is the only sanctioned reader*, and it raises rather than falling back to
  post-deadline state.
- *Odds physically split* into `odds_features` (opening) and `odds_eval` (closing + results),
  so a feature join cannot reach closing data.
- *Closing line is a benchmark, never a feature.* It postdates the deadline. Used as a
  yardstick it detects a 7pp bias in under one gameweek versus ~16 from outcomes.
- *Guard against researcher-mediated leakage:* CLV is monitoring with pre-committed
  thresholds, and tuning is judged against the simulator on admissible data only.

**Engineering**
- Bench weight 0.10 — zero buys filler that never covers an absence, full buys a bench that
  never plays.
- MILP time limit of 60s. Symmetric pools can make branch-and-bound run indefinitely.
- Retraining on a fixed 6-gameweek cadence rather than by judgement.
- Rates computed per stat-availability window, so a recently added stat is not diluted by
  seasons that never recorded it.

## Rank-aware simulator (2026-08-10) — BUILT; verdict on `lambda_rank` is NEGATIVE

`backtest/field_sim.py` samples rival squads from the ownership actually recorded in the
archive (`selected`), so rank can be measured instead of only total points. This was the one
capability gap against the stated top-10k goal: `lambda_rank` had sat at 0 since it was built
because nothing could measure whether raising it helped.

**Where the system actually finishes**, against a 30,000-manager simulated field:

| season | our points | percentile | equivalent rank |
|---|---|---|---|
| 2023-24 | 2354 | 97.3% | ~293,000 |
| 2024-25 | 2524 | 96.3% | ~407,000 |
| 2025-26 | 2322 | 99.0% | ~109,000 |

A top 1-4% manager, not a top-0.1% one. That is the honest headline.

**`lambda_rank` makes things worse at every level tested** — worse points AND worse rank in
all three seasons:

| lambda_rank | mean rank in 11M |
|---|---|
| **0.00** | **269,622** |
| 0.15 | 397,833 |
| 0.30 | 549,266 |
| 0.50 | 508,200 |

This confirms the caveat written when the layer was built: being contrarian *without edge* is
pure variance. Discounting template players only pays where our forecast genuinely disagrees
with ownership, and ours does not disagree well enough. **Keep `lambda_rank` at 0.**

**One structural limit on that verdict.** Our season score is deterministic — one squad, one
realisation — so the measurement can see the cost of differentials but not their upside. To
test whether variance-seeking raises P(top 10k), our own points would have to be drawn from
their predicted distributions across many runs, which needs the full per-player pmf that Item
10 deferred. So: no evidence `lambda_rank` helps, and a known reason this test cannot detect
its main claimed benefit.

**Field realism was the hard part.** The first version re-drew each manager's squad every
gameweek and produced a 99.9th percentile of 2181, where real FPL's top 10k finishes near
2500 — the law of large numbers crushes 38 independent draws toward the mean. Drawing a squad
ONCE and keeping it restored the tail (p99.9 of 2447-2753). Persistence is what creates the
competition. A `churn` parameter was also removed rather than shipped: swapping picks for
other picks from the same distribution leaves the squad a single weighted sample and changes
nothing, so the knob did not do what its name claimed.

## Transfer MILP wired into the simulator (2026-08-10) — +213 points, rank 224k -> 9.6k

The largest single improvement in the project, from code that was already **built and tested
but never called**. `simulate_season` used a greedy one-swap policy; `optimise/transfers.py`
had been written for the `myteam` command and was unused by the backtest.

| policy | 2023-24 | 2024-25 | 2025-26 | mean points | mean rank |
|---|---|---|---|---|---|
| greedy | 2387 | 2524 | 2357 | 2423 | 223,789 |
| **milp (max 2)** | **2701** | **2722** | **2485** | **2636** | **9,656** |
| milp (max 3) | 2660 | 2680 | 2560 | 2633 | 18,822 |

Transfers 21 -> 48 of 38 free (the excess funded by hits); hits 0 -> 11. Allowing three
transfers a week is NOT better than two — more churn, no gain.

**This clears the 203-point gap to top 10k identified by the rank simulator.**

*Caveats that matter before believing the rank figure.* Selling price is taken as market
price because the simulator does not track purchase prices, so budget — and therefore
transfer freedom — is optimistic; 48 transfers a season would be harder in reality. The
historical replay also has no availability gate, which cuts the other way. Treat +213 as
directionally strong and the absolute rank as flattering.

## Forecast bias by ownership (2026-08-10) — the Haaland question, answered

With rank measurable, our opening squads turned out to average **20-25% ownership against the
field's 43%** — accidentally contrarian, not deliberately so. And the players missed were not
crowd errors: Palmer (76% owned, 214 pts), Semenyo (74%, 253 pts), Guéhi (50%, 248 pts).

Forecast bias by ownership band, consistent across three seasons:

| ownership | predicted | actual | bias |
|---|---|---|---|
| <5% | 2.00 | 2.27 | -0.28 |
| 15-30% | 3.32 | 3.84 | -0.53 |
| **>50%** | **4.80** | **6.05** | **-1.25** |

We under-forecast everyone, and under-forecast elite players nearly **3x** as much. Regression
shows this is not simple compression (slope 0.923, intercept +0.5); highly-owned players beat
even the adjusted line. Adding `log(ownership)` to the regression has a positive coefficient
in all three seasons (0.383 / 0.301 / 0.238) and lifts R^2 by ~7.5% relative.

**Ownership carries signal our model does not.** It is published pre-deadline, so it is a
legitimate feature.

### ...but calibrating on it made decisions WORSE. Built, measured, rejected.

`models/calibrate.py` fits `actual ~ a + b*predicted + c*log1p(ownership)` walk-forward and
applies it. Result:

| variant | 2023-24 | 2024-25 | 2025-26 | mean | rank |
|---|---|---|---|---|---|
| raw | 2701 | 2722 | 2485 | **2636** | **9,656** |
| calibrated | 2657 | 2633 | 2368 | 2553 | 51,822 |

**-83 points, rank five times worse.** The bias is real; correcting it is not exploitable:

1. A regression coefficient is not a *per-pound* coefficient. Highly-owned players are
   expensive, so raising their scores spends budget the fitted bias says nothing about.
2. Ownership lifted R^2 by 0.007 on a base of 0.09 — a slight gain in *prediction* applied to
   a *ranking* problem, where it reshuffles the entire selection.
3. Walk-forward coefficients (0.44-0.65) are larger than the within-season ones that
   motivated it (0.24-0.38), so the correction applied is stronger than the diagnostic
   suggested.

Second time a principled fix for a measured bias has backfired (the first was allocating six
bonus points per fixture). **Generalisable lesson: a bias that is real in a regression is not
necessarily exploitable inside a constrained optimiser.** Kept off by default;
`bias_by_ownership` remains a useful standing diagnostic.

## Transfer policy: club-limit block (2026-08-10) — BUG FOUND AND FIXED

With rank finally measurable, the diagnostic was: **14-17 transfers used out of 38 free ones,
and zero hits ever taken** across three seasons. Two hypotheses were tested and BOTH were
wrong before the real cause turned up:

1. *"The transfer threshold is too high."* Sweeping it from 1.5 down to 0.0 changed transfer
   counts barely at all and did not improve points. Raising it to 1.5 was marginally BEST.
2. *"It sells the cheapest player and cannot afford an upgrade."* It sells at £6.2m against a
   £6.7m squad average, and 71% of same-position players were affordable.

The actual cause: `_maybe_transfer` took the single highest-value candidate and **abandoned
the transfer entirely if that one player breached the 3-per-club limit**, instead of trying
the next best. That blocked **16 of 37 gameweeks (43%)** in 2025-26, while gains of 10+
horizon points sat available. Fixed by walking candidates best-first until one is legal.

| season | before | after | transfers | rank before -> after |
|---|---|---|---|---|
| 2023-24 | 2354 | 2387 | 14 -> 21 | 293k -> 204k |
| 2024-25 | 2524 | 2524 | 17 -> 17 | 407k -> 407k |
| 2025-26 | 2322 | 2357 | 16 -> 26 | 109k -> **60k** |

Worth remembering: neither hypothesis survived contact with measurement, and the cause was a
control-flow bug rather than a tuning problem. **The gap to top 10k is 203 points; this
recovered 23.** Transfers are still only 17-26 of 38, so the real MILP
(`optimise/transfers.py`, built and tested but not wired into the simulator) remains the
largest single opportunity.

## Chip usage layer (2026-08-10) — BUILT, +158 mean season points

Treated as **optimal stopping**, not valuation: eight chips, none carrying over, so a chip
unplayed at the GW19 deadline is thrown away. Values are per-chip and distinct — Triple
Captain adds one extra copy of the captain's score, not three.

| season | no chips | with chips | delta |
|---|---|---|---|
| 2022-23 | 1834 | 2329 | +495 |
| 2023-24 | 2226 | 2354 | +128 |
| 2024-25 | 2390 | 2390 | 0 |
| 2025-26 | 2108 | 2119 | +11 |

**First attempt cost -73 points.** It played Bench Boost in GW2 for +21 when GW33 was worth
+144. The stopping rule compared the current week against the median observed week, and early
in a season there is almost no history — so its view of the future was anchored on the
present. Fixed by encoding that the best remaining week beats a typical one by ~2x for Bench
Boost and Triple Captain (double gameweeks) and ~2.5x for Free Hit (blanks).

### RE-TESTED under the MILP transfer policy (2026-08-10) — wildcard STILL does not help

The earlier verdict was measured under the weak greedy policy, so it was fair to suspect the
wildcard's damage was an artefact of transfers being bad generally. **It was not.**

| chip set | 2023-24 | 2024-25 | 2025-26 | mean points | mean rank |
|---|---|---|---|---|---|
| **BB+TC only** | 2711 | 2734 | 2493 | **2646** | **7,089** |
| all chips | 2693 | 2695 | 2540 | 2643 | 14,055 |
| BB+TC+free hit | 2701 | 2722 | 2485 | 2636 | 9,656 |
| no chips | 2487 | 2481 | 2321 | 2430 | 258,378 |

Two conclusions. **Chips are worth a great deal** — +216 points and a rank move from 258,000
to 7,089. And **a permanent rebuild optimised over six gameweeks is simply worse than steady
MILP transfers**, which is why the wildcard loses regardless of policy strength. Free Hit is
also mildly negative, losing to BB+TC in all three seasons.

`DEFAULT_CHIPS` is now `("bench_boost", "triple_captain")`. The point gaps between the top
three sets are ~10 and within noise; the rank gaps are larger and consistently favour BB+TC.

### (superseded) The wildcard is excluded by default

Decomposing the delta separated the chip's own week from its after-effects:

| season | total | chip weeks | all other weeks |
|---|---|---|---|
| 2022-23 | +495 | +329 | +166 |
| 2023-24 | +128 | +128 | 0 |
| 2024-25 | 0 | +135 | **-135** |
| 2025-26 | +11 | +181 | **-170** |

The chips themselves are reliably positive (+329/+128/+135/+181). The **wildcard's permanent
rebuild** is what swings, costing 135-170 points across the remaining weeks in two seasons.
Testing chip sets directly:

| set | 22-23 | 23-24 | 24-25 | 25-26 | mean | worst |
|---|---|---|---|---|---|---|
| all chips | +495 | +128 | 0 | +11 | +158 | 0 |
| **no wildcard** | +9 | +128 | +134 | +214 | +121 | **+9** |
| BB+TC only | +54 | +112 | +101 | +202 | +117 | +54 |

Dropping the wildcard wins in three of four seasons. "All chips" leads on mean ONLY because
of 2022-23 (+495); excluding that season it averages +46 against no-wildcard's +159. Since
2022-23 was the World Cup season with unusually heavy fixture rescheduling, that is the one
season least likely to repeat.

`DEFAULT_CHIPS` therefore omits the wildcard. Re-enable once the rebuild funds itself with
selling rather than current prices, and stops over-fitting to a six-gameweek horizon it then
has to live with for the rest of the season. Free Hit contributes little (+4 mean) but is
kept because blanks are probably under-represented in this simulation.

Also fixed: the planner originally retired BOTH windows of a chip on first use, silently
discarding the second-half copy — 4 chips played instead of 8.

Caveats: the multipliers proxy for fixture structure a live planner could read off the
calendar directly; Free Hit still loses points when forced out at expiry; the wildcard
rebuild uses current rather than selling prices.

## Captaincy as a durable premium (2026-08-10) — BUILT, on by default

Gap found: the squad optimiser doubled the captain's points for the gameweek it was solving,
but `horizon_points` — the valuation driving every transfer decision — was a plain sum of
expected points with **no captaincy term at all**. A player who would wear the armband for
five straight gameweeks was valued identically to one who never would, despite the armband
granting a second copy of his score. That systematically underpriced reliable captains.

`optimise/captaincy.py` adds the premium, with two deliberate choices:

- **Credit is shared via softmax, not winner-take-all.** Assigning the armband to the
  forecast argmax gives a marginally-best player everything and his near-equal rival nothing.
  Forecasts are nowhere near that precise.
- **`variance_weight` exposed but left at 0.** The armband quadruples variance
  (`Var(2X) = 4·Var(X)`). A risk-neutral objective is right for maximising total points and
  wrong for chasing a rank; this is the same knob as `lambda_rank` and waits on the same
  measurement.

Effect on real data (GW10-15 window): Haaland +6.23 horizon points (25.78 -> 32.00, **+24%**),
Salah +2.57. No change to the top-10 ordering — captaincy value correlates strongly with raw
points — but it widens the gaps, which is what decides whether a premium is worth a -4 hit.

### Backtested across four seasons

The simulator could not test this at first: its transfer policy compared raw single-gameweek
expected points and never called `horizon_points`, so the premium had no path into the
simulation — the same blocker that made the flexibility test meaningless. Wiring the horizon
valuation into `simulate_season` fixed both.

| variant | 22-23 | 23-24 | 24-25 | 25-26 | mean |
|---|---|---|---|---|---|
| A myopic (old) | 1723 | 2089 | 2341 | 1901 | 2014 |
| B horizon, no captaincy | 1843 | 2226 | 2028 | 2159 | 2064 |
| **C horizon + captaincy** | 1834 | 2226 | 2390 | 2108 | **2140** |

Delta vs A: **+111, +137, +49, +207 — positive in every season.** No other change tested in
this project has held its sign across four seasons.

*Mechanism, visible in the captain-points column:* the horizon valuation ALONE degrades
captaincy (B harvests 263 captain points in 2024-25 against A's 334) because it drifts toward
players who are steady over six weeks rather than explosive in one. The captaincy term
restores it (C: 319). So B and C are not "good and better" — B introduces a regression that
C repairs.

**Honest limit on attribution.** C minus B is -9, 0, +362, -51: the captaincy contribution
*specifically* is dominated by one season. The COMBINATION is a reliable improvement; how much
of it is captaincy versus horizon is not settled. Both are now defaults
(`horizon=6`, `captaincy_weight=1.0`); `horizon=0` restores the old myopic policy.

This also changes **decision quality, not predictive power**. Forecast accuracy is untouched
by construction — Spearman, calibration and top-N precision are identical, because the point
forecasts themselves were not modified.

**Caveat:** captaincy reliability is only as good as the underlying forecasts, and we already
know those under-rate elite strikers (the model still does not select Haaland). Haaland's
0.23 captain share over six gameweeks is lower than it should be, and that is a symptom of
the forecast, not the captaincy layer. The softmax `temperature` is also an untuned free
parameter — it should be calibrated against how often the forecast argmax actually turns out
to be the best armband.

## Flexibility / pivot-value layer (2026-08-10) — INCONCLUSIVE, not rejected

Hypothesis: holding the most expensive player in a position is worth more than his expected
points, because downgrading is always affordable while upgrading often is not.

Implemented as positional reach — `bank + max(selling price held in that position)` — and
added to the squad objective as `flexibility_weight` (default 0). Verified to bind: at
weight 1.0 it buys a £14.0m forward the baseline does not hold, **costing 2.2 expected points
per gameweek (~84 per season)**.

Measured across three seasons:

| flex weight | 2023-24 | 2024-25 | 2025-26 | mean delta |
|---|---|---|---|---|
| 0.05 | +64 | -350 | 0 | -95 |
| 0.15 | +64 | -307 | 0 | -81 |
| 0.40 | +4 | -466 | -214 | -225 |
| 1.00 | +32 | +10 | +10 | **+17** |

**Non-monotonic, so this is noise, not signal.** But the test is underpowered *by
construction*: the simulator's transfer policy makes one greedy same-position swap per
gameweek with no hits, and the entire value of holding a premium is the ability to make a
decisive pivot when new information arrives. A policy that never pivots cannot exploit the
option it is being asked to price.

**Verdict: cannot be tested until the simulator uses `optimise/transfers.py`** (the real MILP
with hits and multi-transfer chains) instead of the naive policy. The layer is built, tested
and off by default until then. Do not read `+17` as support for the idea.

## Things tried and rejected

| Change | Result |
|---|---|
| Allocate exactly 6 bonus points per fixture | **-160 season points.** Normalising within a match destroys cross-match comparability, and the optimiser ranks players across fixtures |
| Isotonic recalibration of clean-sheet probability | No reliable gain; worse in one season |
| Understat / FBref for xG | Unreachable and Cloudflare-blocked. Unnecessary — FPL's own API carries xG from 2022-23 |

## Leak found in the simulator itself (2026-08-10)

`fpl simulate` originally loaded the **saved** minutes model, which `fpl minutes` fits on all
seven seasons — including the one being replayed. The point-in-time logic for rates and the
match model was correct; the minutes model, the single largest driver of points, was not.

Cost of the leak on 2025-26: **2086 -> 1901 season points, a 9% overstatement.**

`simulate` now retrains the minutes model per season on prior seasons only (`--strict`, the
default). `--reuse-model` restores the old behaviour and is only for speed on throwaway runs.
`walkforward.prior_seasons` exists as a named, tested function so this is hard to reintroduce.

Lesson worth keeping: the point-in-time discipline was enforced carefully on *data* but not
on *fitted models*. A model is data too.

## Full points distribution (2026-08-10) — BUILT, `models/distribution.py`

Every player-gameweek now has an exact pmf over integer FPL points, built by discrete
convolution of the component models conditional on the minutes bucket — not by simulation, so
there is no sampling noise in it. It reproduces the assembler's `expected_points` at r=0.997,
which is a genuine cross-check: two independently written paths to the same expectation.

Building it exposed a calibration failure that no mean could have revealed. The first version
— independent components, Poisson counts — under-predicted double-digit hauls by a factor of
**2.5**, and got worse the further into the tail it went (0.95x realised at 2+ points, 0.40x
at 20+). That is the exact quantity captaincy is a bet on.

Two measured corrections, in order of size:

1. **Bonus is conditioned on goal involvement.** In the archive, expected bonus is 0.095 with
   no returns and 2.248 with two — a factor of 24. Treating bonus as independent of returns
   is defensible for a mean and indefensible for a tail, because a haul IS a return plus the
   bonus that reliably accompanies it. The rescaling is mean-preserving per player.
2. **Goals and assists are negative binomial, not Poisson**, with dispersion measured by
   moments against realised counts (k = 2.85 and 1.88). A player's true rate moves with form,
   role and matchup; Poisson assumes that away and loses about a third of the multi-return
   tail.

After both, the predicted-to-realised ratio is flat at roughly 0.6 across every threshold from
5 to 20 points, instead of decaying from 0.95 to 0.40. **A flat ratio is a level error; a
decaying one is a structural failure.** The residual level is the forecast's known 13%
global under-prediction (mean 1.10 predicted against 1.26 realised), already documented above
under the ownership work — not a fault of the distribution.

Deliberately NOT applied here: the mean scale factors from the same fit (goals x1.073,
assists x1.268). They are a real finding about `attack.py`, and applying them in this module
would silently put its mean at odds with the number the optimiser reads.

## Repeated simulation (2026-08-10) — BUILT, `backtest/repeat_sim.py`

Season results were single deterministic numbers, and choices had been made on gaps of ~10
points with nothing to say whether 10 points was signal. Each season is now replayed many
times with every player-gameweek outcome redrawn from its own distribution.

Two design points carry the method:

* **Every variant meets the same drawn season.** Totals swing far more between draws than
  between strategies, and that swing cancels out of a paired difference. Comparing independent
  runs would need orders of magnitude more draws to resolve the same gap.
* **The field is scored on the same draw**, so our squad and the 20,000 rivals live in one
  world per replay and rank stays internally consistent.

What it does and does not establish: it grades strategies against the model's own beliefs, so
it says nothing about whether the forecasts are accurate. Absolute totals from it are worth
nothing. Differences between variants are worth a great deal, and that is all it is used for.
`is_resolved` and `draws_needed` make the conclusion a computed property rather than a
judgement made afresh each time a table is read.

### RESULT: free hit is back in, and the earlier exclusion was a reading of noise

600 paired draws (200 resampled seasons x 3 seasons), every variant on identical outcomes:

    variant vs BB+TC      mean diff    95% CI          wins
    BB+TC+free hit           +16.2    [+14.5, +17.9]   76%
    all chips                 +5.9    [ +2.6,  +9.2]   56%
    BB+TC+wildcard            +4.6    [ +1.4,  +7.9]   55%
    no chips                 -46.9    [-48.0, -45.9]    0%

Re-run afterwards against the rebuilt field over 450 draws, unchanged: free hit +16.9
[+14.9, +19.0], 76% of draws. The field affects ranks and not points, so this was expected —
but a conclusion that survives having its measuring instrument replaced is worth more than one
that has not been asked to. Free hit also improved the median rank in both seasons where rank
resolved (29,700 against 53,350 in 2023-24; 314,325 against 420,750 in 2024-25).

`DEFAULT_CHIPS` is now `("bench_boost", "triple_captain", "free_hit")`.

I had excluded free hit on the grounds that it "loses to BB+TC in all three seasons" — three
deterministic replays, a gap of 10 points, no error bar. That is exactly the reasoning this
module was built to stop. On realised outcomes free hit is still only +3 over three seasons,
but +3 on n=3 is not evidence against +16 on n=600.

Adding the wildcard on top makes things worse (+5.9 for all chips against +16.2 for free hit
alone): when both are available the planner spends the slot on the wildcard, and a permanent
rebuild is worth less than a one-week one. The wildcard's exclusion survives.

**Checked before acting on it.** Resampling grades strategies against the model's own beliefs,
which could in principle flatter strategies that optimise harder against those beliefs —
precisely what a rebuild chip does. It does not: the gap between realised and resampled
totals is 194.4 points for rebuild variants and 194.6 for the rest, uniform and fully
explained by the forecast's known 13% under-prediction.

**Chips are worth far less than previously claimed.** -46.9 for playing none, against the
+216 recorded earlier. Most of that +216 was the double-gameweek bug below: bench boost is
timed onto double gameweeks, and the bug inflated realised points for exactly those players.

## Field model rebuilt (2026-08-10) — rank is measurable again, and it has an exact anchor

The field is no longer a guess that has to be argued about, because ownership pins the answer.
Every manager owns exactly fifteen players, so:

    managers       = sum(selected) / 15          -> recovers 10.77M for 2024-25, the real figure
    mean squad pts = sum(selected * points) / managers

Both exact. That gives a target the simulator must hit — 2014 / 1991 / 1958 season points for
the three seasons — testing squad composition alone, with no modelling in between.

Three separate errors had to be fixed, each worth more than the last:

1. **The sampler did not reproduce the marginals it was given.** Gumbel top-k is
   Plackett-Luce, and Plackett-Luce compresses heavily-weighted items below their share. In
   FPL those are the most-owned players, who are also the highest scorers. Replaced with
   Pareto order sampling, which targets inclusion probabilities directly.
2. **Squads were frozen for 38 gameweeks.** Fixed by giving each manager a fixed uniform per
   player while the ownership targets move weekly. Persistence and transfers then come from
   one construction rather than two mechanisms fighting each other — and the transfer rate is
   whatever the ownership data implies rather than something invented.
3. **Every manager was equally skilled.** A field of average managers has no top end and
   cannot be ranked against. Skill tilts a manager's ownership toward players our forecast
   rates, balanced by Sinkhorn so squads stay legal AND the field still owns what the real
   field owned.

Result — composition within 0.3% everywhere, exact in two of three seasons:

    season     squad-15 (anchor)     median    top 10k    our score -> rank
    2023-24     2014  (2014)          2039       2494      2617 -> saturated
    2024-25     1985  (1991)          2053       2500      2467 -> ~35,200
    2025-26     1958  (1958)          1919       2281      2359 -> ~550

**One number here is fitted and it should be treated with suspicion**: the skill spread, set
so the simulated top-10k threshold matches FPL's published ~2500. Nothing in the archive
records how individual managers finished, so the tail cannot be derived the way composition
can. Any rank quoted inherits that assumption.

**Two of three seasons still saturate, and that is a finding about the backtest.** The field
now reproduces real ownership exactly and reaches a realistic top-10k threshold, so a score
above nearly all of it more likely means the replay flatters us — selling price taken as
market price, no availability gate, XI chosen on the same forecast it is scored against —
than that these squads would have finished in the top few hundred.

### Rank is far more fragile than points, and here is the measurement that shows it

Running the field against RESAMPLED outcomes instead of realised ones moves our median rank
from ~35,200 to ~420,750 for 2024-25 — a factor of twelve. The obvious explanation is that
our edge over the field shrinks under resampling, since the model's known under-prediction of
elite players would flatter a squad built out of them. **Measured, that is not what happens:**

                       us      field median    gap
    realised          2467         2053        +414
    resampled (n=12)  2367         1978        +389

Our advantage is essentially intact — 414 against 389. The entire rank difference comes from
the SHAPE of the field's upper tail, which the two outcome models disagree about, and which
percentile is exquisitely sensitive to. A 25-point change in our edge moved the median rank by
an order of magnitude.

The practical rule: **quote points differences, treat ranks as indicative.** A points gap is
stable across every modelling choice tested here; a rank is not. Which tail is the right one
remains unresolved and is not settleable from the archive, which records no manager results.

## Rank was not measurable before that rebuild (2026-08-10) — EARLIER RANK FIGURES WITHDRAWN

Fixing the double-gameweek bugs removed an inflation that had been propping up the simulated
field, and revealed that the field is far too weak to rank against. Measured on 2024-25,
where real FPL's average manager scored ~2200 and the top 10k ~2500:

    simulated field    median 1640    p90 1979    p99 2165    p99.9 2277    best 2374

Our squads score ~2480 and therefore beat 100% of a 20,000-manager field. Every rank quoted
before this date — including **"mean simulated rank 7,089, reaching the top-10k target"** — was
computed under the bug and is withdrawn. It was never a measurement of rank; it was a
measurement of how far off the top of an inadequate distribution we sat.

`rank_metrics` returns `saturated: True` and `rank_in_11m: None` rather than extrapolating,
because an extrapolated rank improves precisely as the field gets worse. That guard is kept
now the field is rebuilt, because saturation still happens and still means the same thing.

**Points comparisons were never affected; only rank was.**

## Three double-gameweek bugs (2026-08-10) — FOUND AND FIXED

Found while checking the frame the resampler was about to be built on. All three were silent,
all three were confined to double gameweeks, and double gameweeks are precisely when chips are
played — so they landed on the decisions that matter most.

1. **The team join was a cross product.** Archive rows are already one per player-FIXTURE, and
   `assemble` joined team forecasts on `team` alone. Two fixtures met two team rows and came
   out as **four**.
2. **Goal allocation was grouped by team, not by team-and-fixture.** The group then held two
   fixtures' worth of rows but only one fixture's expected goals, so a double gameweek
   player's attacking returns were credited **once instead of twice** — running opposite to
   bug 1. Netting two errors is not the same as having none.
3. **Realised points were summed across fixture rows.** They are recorded per player-GAMEWEEK,
   so a double gameweek player was credited with points he never scored. The backtest was
   missing the `aggregate_gameweek` step the live path already had.

Fixed by keying the join and the allocation on `fixture` where the caller has one (the live
path, which passes one row per player, is untouched and still fans out on team correctly), and
by collapsing to gameweek level before actuals are joined. `tests/test_double_gameweeks.py`
covers all three, including the asymmetry that expected points SHOULD sum across fixtures
while realised points must not.

All previously reported season totals were measured under these bugs and are superseded.

## Attack under-prediction traced and fixed (2026-08-11) — two defects, both exact

The 13% global under-prediction was not vague model weakness. It decomposed into two specific
defects, each of which explains its share almost exactly.

**1. A penalty-mass leak worth 6.1% of every team's expected goals.** `allocate_team_goals`
subtracted the team's theoretical penalty total from the open-play pool, then allocated
penalty goals only to identified takers. The archive has no set-piece order, so
`penalty_share` is 0 on every backtest row and that mass was allocated to nobody:

    0.11 / 1.43 * 0.79 = 0.0608 of team xG deleted   ->  conservation 0.9392
    measured correction needed: x1.073

Fixed by subtracting only the penalty goals actually handed to somebody, which makes the
allocation conserve the team total by construction — full takers subtract the lot, no takers
subtract nothing, and the mass is spread by open-play share instead. Conservation is now
1.0000 exactly.

**2. `ASSIST_RATE` was 0.72 where the archive says 0.905.** An unmeasured estimate. FPL
assists per FPL goal, over all seven archived seasons, is remarkably stable:

    2019-20 0.900   2020-21 0.918   2021-22 0.896   2022-23 0.893
    2023-24 0.896   2024-25 0.905   2025-26 0.934        overall 0.905

and 0.72 x 1.268 = 0.913, which is the measured assist correction almost to the decimal.

Effect on calibration:

    quantity                        before     after
    sum(player xG) / team xG        0.9392    1.0000
    sum(player xA) / team xG        0.7200    0.9050
    mean predicted / realised       0.87      0.9346
    distribution P(>=10) ratio      0.58      0.78
    distribution P(>=15) ratio      0.61      0.87

**The methodological point.** These scale factors were measured weeks earlier and deliberately
NOT applied as a correction inside `models/distribution.py`, on the grounds that a fudge there
would put that module's mean at odds with what the optimiser reads. That restraint paid off:
both turned out to be specific, fixable defects in `attack.py`. Applying the fudge would have
left both bugs in the model that makes decisions, hidden behind a correction in the model that
only describes them.

Residual under-prediction is 6.5%, of which the match model accounts for about 2% (team
expected goals 1.470 against actual 1.441).

## Selling price now follows the real rule (2026-08-11) — and it costs us

`backtest/season_sim.py` took selling price to be market price because it never recorded what
a player was bought for. That handed the optimiser the full rise on every player who had gone
up. It now keeps a purchase-price ledger and reuses the live path's
`data/my_team.selling_price_tenths` — half of any profit, rounded down, losses in full.

This makes results WORSE and that is the point: 2023-24 fell from 2617 to 2479 and its rank
stopped saturating. One of the reasons the backtest was flattering us has been removed.

## The field plays chips (2026-08-11)

Rival managers played none while we played three. Each now plays one Bench Boost and one
Triple Captain per half-season, timed with probability proportional to a gameweek's fixture
count so they aim at doubles the way real managers do — but NOT timed with our forecasts,
which would make every rival as good at chip timing as the system being measured.

    field           median   p99    top 10k   best
    without chips    2053    2428    2496     2563
    with chips       2081    2466    2539     2613

`SKILL_DISPERSION` was re-fitted from 0.35 to 0.25 as a result: chips supply part of the
spread that skill previously had to supply alone, and leaving both at their old values would
have double-counted it and pushed the top-10k threshold to 2545 against a target of 2500.

## Phantom goalkeeper goals (2026-08-11) — the largest single modelling error found

Chasing the residual under-prediction turned up something worse than a miscalibration. The
allocator was giving goalkeepers **79 expected goals across three seasons**, against zero
actually scored — and a goalkeeper goal is worth 10 points.

Two causes, one behind the other:

1. `attacking_rates` groups its shrinkage by position, but the frame `pipeline.player_rates`
   passed it had no `position` column. The `[c for c in group_cols if c in df.columns]` guard
   then quietly produced an empty grouping, and **every player in the league shrank toward one
   common prior** — dragging goalkeepers toward an outfielder's attacking rate. GKs came out
   with a median 0.111 xG per 90, higher than defenders.
2. Missing rates were filled with the league-wide median rather than the position's.

Both fixed, and `attacking_rates` now warns loudly instead of silently falling back. The
effect on goal allocation:

    position    predicted share (before -> after)    realised share
    GK               0.023  ->  0.000                    0.000
    DEF              0.196  ->  0.117                    0.129
    MID              0.533  ->  0.572                    0.555
    FWD              0.247  ->  0.312                    0.316

Every position now tracks reality, forwards almost exactly. Season points rose in all three
seasons and the 2023-24 rank went from 19,800 to 8,800.

**The lesson is about the guard, not the goalkeepers.** `[c for c in group_cols if c in
df.columns]` is a defensive idiom that reads as robustness and behaves as silent degradation.
It turned a modelling choice into a no-op and nothing failed.

## A stale partition made a re-measurement look 12% worse than it was

While diagnosing the above, a component calibration showed assists 12% under-predicted, which
contradicted a conservation fix known to be exact. The cause was not the model: `sim_forecasts`
still held a **2022-23 partition written before the attack fixes**, and reading the table
concatenated two generations without complaint.

`write_table` now stamps each partition and `partition_report` surfaces the spread, warning
when a table holds more than one generation. Partitions are refreshed one season at a time, so
this will recur otherwise — and it sends you hunting a bug in the model that is really in the
data.

## Price-change model (2026-08-11) — BUILT, `models/prices.py`, `fpl prices`

Nothing modelled price movement, and team value compounds all season. FPL moves prices on net
transfers, which the archive records directly, so the driver is observable rather than
inferred. An ordered logit over {fall, same, rise} — one linear predictor with two cutpoints,
so a player can never come out simultaneously likely to rise and likely to fall, which two
independent classifiers would allow.

Walk-forward log loss against a base-rate baseline (92.5% of player-gameweeks see no change,
so accuracy is useless here):

    2023-24  0.2695 vs 0.2956    2024-25  0.2853 vs 0.3120    2025-26  0.2342 vs 0.3085

**The first version was wrong in an instructive way.** Fitted on net transfers relative to a
player's own ownership, it nominated as the week's likeliest risers a set of players owned by
32, 53 and 122 managers — a fraction of 0.66 on 32 owners is 21 transfers and moves nothing.
FPL's thresholds are absolute, and the archive says so plainly:

    owners      n        P(price changes)
    < 1k      15,353         0.0096
    > 200k    33,179         0.2215

Adding absolute volume took walk-forward log loss from 0.2770 to 0.2342 — from 10% better than
knowing nothing to 24% — and the riser list became Gyokeres, Saka, Doku.

Reported, not optimised against. Expected movement is about £0.1m per player per gameweek at
the extremes, against forecast differences of whole points, and this project has a track record
of real signals that are not exploitable inside a constrained optimiser.

## Distribution refinements (2026-08-11) — all three measured, none applied

    dependence                        measured        applied?
    goals <-> assists                 1.74x lift        no
    clean sheet <-> returns           1.37x lift        no
    defcon overdispersion        var/mean 1.5-1.8       no

The first two are real but an order of magnitude smaller than the bonus-to-returns coupling
(24x) that WAS worth modelling, and the tail they would fatten now sits at 0.78-0.95 of
realised after the attack fixes.

The third was tested properly and rejected on evidence: a negative binomial with measured
dispersion improves defenders (0.2633 -> 0.2749 against a realised 0.2794) and makes
midfielders and forwards worse by overshooting. Two of three positions degrade, so Poisson
stays. Recorded in `models/secondary.py` so nobody re-measures it.

## Previously rejected results, re-tested (2026-08-11)

    variant vs baseline    mean diff    95% CI            wins
    lambda_rank 0.1          +8.19    [+2.96, +13.43]     53%
    flexibility 0.3          +0.78    [-3.25,  +4.81]     34%
    flexibility 0.1           0.00    [ 0.00,   0.00]      —
    lambda_rank 0.3          -3.04    [-9.69,  +3.61]     49%

**`lambda_rank` at 0.1 appeared to refute the earlier "worse at all levels" verdict** — +8.19
with an interval excluding zero. It was not adopted, on the grounds that a 0.53 win rate meant
the mean rode on a few large wins and that a rank term gaining points was anomalous.

**That caution was correct, and re-measuring on the corrected forecasts closed it.** Per
season, on a model with the goalkeeper allocation bug fixed:

    weight     2023-24    2024-25    2025-26    pooled
    0.1         -5.05      +1.44      +1.63     -0.66  [-5.61, +4.30]
    0.2        +11.89     -21.04      +9.10     -0.02  [-5.77, +5.73]

Worth nothing at either weight, with signs flipping between seasons. The original +8.19 was an
artefact of the model underneath it — which is exactly what ground rule 2 warns about, and why
a pooled interval should never be adopted without looking at the per-season signs.
`lambda_rank` stays at 0, now on much firmer evidence than before.

**`flexibility_weight` at 0.1 changes literally nothing** — a paired difference of exactly
zero, meaning the term never once altered a decision. That is a more useful answer than the
original "inconclusive": the layer does not bite until the weight is large enough that it is
also unhelpful.

### A limitation of the instrument, worth stating plainly

Pooling draws across seasons treats them as independent, and for anything that changes
DECISIONS they are not: within a season every draw shares one decision path, and only the
outcomes vary. So the confidence intervals above measure "how uncertain is this effect on
these three seasons", not "how uncertain is it in a new season" — for which the effective
sample size is three. Effects that hold with a consistent sign in every season (free hit:
+21 / +20 / +7) are on much firmer ground than an equally narrow interval built from a pooled
mean. Check per-season signs before adopting anything on the strength of a pooled CI.

## Component ablation re-measured (2026-08-11) — the previous ordering was wrong

Re-run on the corrected forecasts, then re-run again through the paired instrument because the
first pass was visibly noise: on single deterministic replays every component except attack
changed SIGN between seasons (bonus -71 / +92 / +67, saves -41 / +107 / +63).

Pooled over 360 paired draws, season points lost when each component is removed. Re-run again
after the attack fixes, which is the right-hand column — the model changed underneath it:

    component            before      after attack fixes     resolved?
    attack             -106.4       -122.5 [-132.7, -112.4]  yes, dominant
    clean sheets        -45.9        -40.5 [ -47.5,  -33.4]  yes
    defensive contrib   -15.5        -19.4 [ -24.0,  -14.9]  yes
    saves                -5.7        -12.8 [ -18.0,   -7.7]  yes (was marginal)
    appearance           +1.2        -12.6 [ -18.9,   -6.3]  yes (was NO)
    bonus                -3.1         -2.8 [ -10.1,   +4.6]  NO
    cards                -0.8         -2.0 [  -7.8,   +3.7]  NO

Fixing the attack model made attack MORE dominant, and pulled saves and appearance points over
the line into resolved. Bonus and cards remain indistinguishable from zero in decision terms
through both rounds, which is what keeps "rebuild bonus from BPS components" off the list.

**Attack is the dominant component by a factor of two over the next one**, reversing the old
claim that it ranked last. Three components cannot be distinguished from zero at all, which
also means the bonus model — long flagged as the crudest thing here and a standing target for
rebuilding from BPS components — is not currently worth that work: it does not move decisions.

Note what "resolved" means: the paired difference excludes zero. It does NOT mean the
component is unimportant in the scoring rules, only that changing it does not change which
squad the optimiser picks. A component identical for every player carries no decision weight
however many points it is worth.

## Rank-aware captaincy (2026-08-11) — BUILT, measured, NEGATIVE. Off by default.

`optimise/rank_objective.py` picks the armband to maximise the share of the field beaten,
using the points distribution and the simulated field on shared draws. Measured across three
seasons against the expected-points rule:

    season     points            percentile          armband changed
    2023-24    2617 -> 2614      1.00000 -> 1.00000      9 / 38 gws
    2024-25    2467 -> 2467      0.99680 -> 0.99680      8 / 38 gws
    2025-26    2359 -> 2343      0.99995 -> 0.99995     12 / 38 gws

It costs points and improves rank in no season. The reason is structural rather than a fault
in the objective: at the 99.7th percentile there is almost no rank left to win and only points
to lose. A rank objective is worth something when you are close to the field, and these
backtests are not close to it.

Kept, off by default, because the mechanism is sound and tested — it correctly prefers a
steady captain when ahead and a volatile one when behind, and it correctly cancels a player
the field already owns. It is a tool for a real season where the gap to a rival is small, not
for a backtest that is already beating everything.

**What this does NOT show.** The percentiles above come from realised outcomes, one per
season. A proper test runs it through `repeat_sim` for a paired comparison on rank. That was
not done and is the first thing to do if this is revisited.

## Independent review (2026-08-12) — six material findings, one fixed same day

An outside reviewer audited the system end to end. It found things the self-audit below did
not, including one defect the author had *noticed and then failed to act on*. Verified and
actioned as follows.

### FIXED — defensive contributions were evaluated at mean minutes, and were half their size

`secondary.defensive_contribution_points` computed `2 x P(Poisson(rate x E[minutes]) >= T)`.
`P(N >= T)` is sharply convex in lambda while T sits above it (10 against a league mean near
7.5), so evaluating at average minutes is not an approximation of the bucket mixture — it is
roughly half of it. `models/distribution.py` had always done it correctly, so the two paths
silently disagreed.

    2025-26 defcon points   at mean minutes   bucket-conditional   realised
                                 1394               2599            2834

That 1,440-point gap was essentially the entire 2025-26 residual (expected 32,978 against
actual 34,409) — a residual this file previously attributed to the match model. Wrong module.

Fixed by conditioning on the minutes bucket. Effect:

    season      calibration before   after
    2023-24          0.9849          0.9849
    2024-25          0.9968          0.9968
    2025-26          0.9584          1.0037
    pmf vs assembler agreement on 2025-26:  1.042 -> 1.006

**The author had seen this.** Early in building `distribution.py` the divergence was noticed
and written down as "the assembler's version is the biased one" — and then never fixed. That
is precisely what an outside reader is for.

### CONFIRMED, NOW FIXED — the horizon valuation read forecasts built after the decision

`horizon_valuations` valued a transfer at GW10 using the GW15 forecast, which was built with
rates and minutes features as of GW15. No realised outcome was read, so the solve-once
machinery was still valid — but a forecast built from post-decision data is future information.
The docstring claiming it used "only forecasts already available at" the decision point was
false. The reviewer's magnitude estimates (70-500 points) were acknowledged as over-corrections
and unreliable; the structural fact was not in doubt.

**Fixed 2026-08-12. It was worth -392 points a season, negative in all three.** See "The
horizon lookahead, fixed and measured" below — it is the largest single correction in this
file, and it withdraws more previous conclusions than any other.

The earlier sub-result "removing the chip planner's lookahead changes nothing (+3 / 0 / +9)"
was measured by degrading the planner alone while the transfer horizon stayed leaky, so it
does not survive; the chip set is re-derived properly below.

### CONFIRMED — `repeat_sim` estimates a deterministic quantity

`rescore` is linear in the drawn points and each draw has mean equal to the pmf mean, so
`E[rescore]` is exactly `sum of dist_mean over the scoring ids` — computable in closed form.
The 600-draw confidence interval is Monte-Carlo error on that number, not uncertainty about
whether a strategy is better. Worse, the author's own guard ("the realised-minus-resampled gap
is uniform across variants, 194.4 vs 194.6") does **not** hold on current data: the spread is
~49 points and ordered against the variants that optimise hardest against the forecast.

This weakens every conclusion drawn through `repeat_sim`, including the Free Hit reversal.
Ground rule 2 (per-season signs) is a partial fix but does not address the estimand being the
model's own belief. The remedy is to grade on something the strategy did not optimise against.

### CONFIRMED — the ownership identity is not exact in blank gameweeks

Players without a fixture have no row, so `sum(selected)/15` understates the manager count and
inflates the anchor by 1.5-2.1% per season. The claim "both exact" is wrong.

A correction in the project's favour: measured against the *achievable* anchor given the
forecast frame, the sampler hits 1985.0 against 1984.3 — 0.04%, not the 0.3% claimed. The
sampler is better than advertised; the gap is missing players.

### CONFIRMED — the market blend never fires

`build_match_forecasts` uses odds only when the fixtures frame carries `p_home_open`. Neither
the backtest path nor the live path supplies it; `features/fixtures.attach_odds` is called from
nowhere in `src/`. Every backtested number describes a **Dixon-Coles-only** match model, and
README's "blended with odds-implied goal expectations" describes a code path that does not run.

### CONFIRMED — the reported rank is nearly a pure function of `SKILL_DISPERSION`

Sweeping it across a +/-60-point band around the assumed 2500 top-10k threshold moves the
2023-24 rank from 1,100 to 73,150. Ranks in README and NEXT_SESSION should be read as
restatements of that assumption. The reviewer also reproduced 2023-24 at 6,600 where README
says 8,800 — a four-manager difference in a 20,000-manager draw.

### Other confirmed items

Vice-captain is not modelled (the captain's zero is doubled; and the `xi.head(1)` fallback
hands the armband to the *goalkeeper*, which a test asserts loosely enough to pass).
Minutes features leak within a double gameweek (1,766 of 86,765 rows). `rate_features` never
passes `noise_scale`. The free-hit rebuild is unguarded against an infeasible solve.

### What the reviewer checked and found sound

No decision path reads `actual_points` — traced through every path including chips and the
rebuilds. The squad and transfer MILPs encode the real rules. The Pareto/Sinkhorn sampler,
the `_pick_starting_xi` greedy argument, the Dixon-Coles point-in-time cut, `distribution.py`'s
numerics, the double-gameweek asymmetry, strict walk-forward for minutes, selling price, and
the saturation guards all verified correct.

## The horizon lookahead (2026-08-12) — FIXED, and it is the largest correction in this file

The independent review's first finding, now built and measured. It cost **-392 season points
on average, negative in all three seasons**, and it withdraws more previous conclusions than
any other entry here.

### What was wrong

A forecast has two dates attached to it: the gameweek being FORECAST, and the gameweek the
forecast is made IN. `horizon_valuations` conflated them. Valuing a GW10 transfer over GW10-15
pulled GW15's row out of a flat table, and that row had been built with rates, EWM form and a
Dixon-Coles fit as of GW15 — none of which existed in October.

No realised outcome was ever read, which is why every no-hindsight guard in `test_season_sim`
kept passing. The leak was one level up: not the outcomes, but the *forecasts of* the outcomes.

### The fix

`historical_forecast.forecast_horizon(history, results, model, features, season, as_of_gw,
targets)` separates the two dates. Everything derived from results is cut at `as_of_gw`; only
the fixture list and kickoff times come from the target week, because FPL publishes the
schedule months ahead. Batched, because the expensive half — the team model and league-wide
rates — depends only on the decision week: seven targets cost 2.6s against 1.2s for one.

`decision_state` pins what one deadline knows, and closes a second leak on the way. In a double
gameweek the second fixture's feature row is built by `shift(1)` over the first, so it already
contains that match's minutes; taking the earlier kickoff's row fixes all 1,766 affected rows.
It uses `drop_duplicates`, NOT `groupby().first()` — the latter takes the first non-null value
per column independently and would reach into the discarded row for exactly the lagged columns
being protected.

### The measurement

Three variants, same code, strict walk-forward, chips on:

    season     A old / flat   B + state fix   C + point-in-time
    2023-24        2598           2540              2340
    2024-25        2597           2631              2113
    2025-26        2537           2529              2070

**C is the honest number.** B - A is the double-gameweek state fix alone: -58 / +34 / -8, signs
flipping, a wash. C - B is the lookahead: **-200 / -518 / -459**.

### Two mechanisms, and the second is the one nobody would have guessed

The leaky horizon ranked players better, because it contained the future — rank correlation
with the realised discounted six-week return of **0.884 against 0.802**.

But the larger effect is stability. Consecutive leaky windows shared five identical frames, so
the forward valuation barely moved from week to week: **rank stability 0.9932, against 0.9624
honest.** A forward view that is 99.3% frozen gives a re-optimiser almost no reason to trade.
Once it updates weekly, as it does live, transfers go 49 to 64 and hits 12 to 27 (2024-25).

The backtest was not simulating the decision problem a live manager faces. It was simulating
one in which next month's judgement had already been made and would not change.

### What this withdraws — the horizon itself is no longer justified

Against a myopic single-gameweek policy, on honest forecasts:

    season      myopic     horizon 6      difference
    2023-24      2229        2340            +111
    2024-25      2287        2113            -174
    2025-26      2055        2070             +15
                                        mean   -16

**The horizon is worth nothing.** Indistinguishable from not having one, with the sign flipping
twice. Two recorded claims die here:

* "Wiring in the MILP horizon was worth **+213 season points**" (module docstring).
* "+111/+137/+49/+207 across 2022-23 to 2025-26 — **positive in every season, which no other
  change tested here has managed**" (module docstring).

Both were measuring the leak. Note the second claim's rhetorical force came precisely from its
consistency across seasons — and that consistency was an artefact of every season's horizon
being assembled the same wrong way.

The horizon machinery stays in place at `DEFAULT_HORIZON = 6`, because nothing yet shows it is
worse than myopic either; but it is now an unproven default rather than the best-supported
result in the project.

### Where the points went — an exact partition, fitting nothing

A season total is exactly `base XI + captain - hits`, so the three columns reconcile to the
total with no residual (verified: all nine rows exact). `bought-sold` is a diagnostic on top —
for every transfer actually made, the realised SIX-WEEK return of the player bought minus the
player sold, which answers what the hit count cannot.

    season   variant   total   base XI  captain   hits  moves  bought-sold
    2023-24  myopic     2229     2031      226     -28     44      +313
    2023-24  leaky      2540     2351      257     -68     54      +650
    2023-24  honest     2340     2167      273    -100     62      +593
    2024-25  myopic     2287     2013      286     -12     40      +431
    2024-25  leaky      2631     2309      370     -48     49      +568
    2024-25  honest     2113     1900      321    -108     64      +427
    2025-26  myopic     2055     1812      271     -28     44      +246
    2025-26  leaky      2529     2271      298     -40     47      +516
    2025-26  honest     2070     1918      256    -104     63      +323

**The hit bill is systematic; the payoff is not.** Against myopic, the honest policy pays
-72 / -96 / -76 more in hits every season — reliable, mean -81 — while the extra transfer value
it buys is +280 / -4 / +77, mean +118 but wildly inconsistent. In 2024-25 it made 24 more
transfers than myopic and its trades returned **+427 against myopic's +431**: identical within
noise, for 96 points of hits. That season's entire -174 is over-trading.

The leaky row shows why the bug was so expensive and so hard to see. It paid the LEAST in hits
(-40 to -68) while extracting the MOST transfer value (+516 to +650) — it did not need to trade
often, because it already knew which trades would pay.

This is a churn signature, not a ranking problem: certain costs against variable benefits. It
says the transfer policy's hit discipline was implicitly calibrated against a forward view that
did not move, and needs re-deriving — out of sample, since two of three seasons would happily
support a more conservative rule that the third does not.

### Is the loss churn or information? Mostly information — smoothing REJECTED

The churn signature above suggests a fix that adds no information: damp the valuation along the
DECISION axis with a causal exponential mean, so a week's revision is believed only in part.
The mean at GW10 uses GW1-10 only, so it is as available live as the raw value. Built as
`_smooth_across_decisions`, exposed as `simulate_season(smoothing=halflife)`, default OFF.

    honest       2023-24  2024-25  2025-26    mean    stability   hits
    hl = 0          2340     2113     2070    2174      0.9662    25/27/26
    hl = 1          2252     2295     2148    2232      0.9866    15/13/14
    hl = 2          2303     2188     2066    2186      0.9917     8/11/8
    hl = 4          2360     1999     2073    2144      0.9942     6/7/7

    leaky (control)
    hl = 0          2540     2631     2529    2567      0.9930    17/12/10
    hl = 1          2570     2425     2548    2514      0.9965    13/7/6
    hl = 2          2468     2529     2435    2477      0.9977     6/5/4
    hl = 4          2395     2379     2310    2361      0.9982     2/2/1

**The control works and the treatment does not.** Smoothing the LEAKY horizon degrades it
monotonically (2567 -> 2514 -> 2477 -> 2361), exactly as predicted: that horizon is already
frozen at 0.993, so damping it can only destroy information. So the instrument is measuring
what it claims to.

On honest forecasts the best setting recovers **+58 of the ~392 gap, about 15%**, and does it
inconsistently: hl=1 is -88 / +182 / +78. It cuts hits almost in half (27 to 13), so it IS
acting on churn — it simply turns out that churn was not where most of the money was.

**Rejected.** It fails the per-season sign test and buys back a sixth of the loss at best.
The conclusion is the useful part: **most of the -392 was information, not over-trading.** The
leaky horizon was not merely calmer, it was better informed, and no amount of policy discipline
recovers that. Kept in the code at default 0.0 because the measurement is worth being able to
repeat, not because it is close to worth switching on.

A second thing this table shows: the honest simulator has visibly more configuration-to-
configuration variance than the leaky one did (2023-24 runs 2252 / 2303 / 2340 / 2360
non-monotonically across a single parameter). The frozen horizon was making every comparison
look more precise than it was.

### Free Hit loses its place — the third instrument in a row to overturn the last

See `DEFAULT_CHIPS`. Re-derived on `bootstrap_realised` with point-in-time forecasts, Free Hit
goes from +7.2 [+5.6, +8.9] to **-5.5 [-6.9, -4.0]**, win rate 0.52 to 0.41. Chip TIMING is
precisely what a frozen forward view distorts, so this is the conclusion most exposed to the
lookahead and it did not survive.

`no chips` still loses in all three seasons at a 0.00 win rate, so playing chips is settled.
WHICH chips is not: Free Hit is -11.4 / -16.3 / +11.3 per season and wildcard swings 176 points
between seasons. The default reverts to BB+TC on the grounds that the positive evidence which
promoted Free Hit is gone and the pooled difference now runs against it.

### The consequence that matters most — the model no longer beats picking on price

Full regeneration on point-in-time forecasts, `DEFAULT_CHIPS = BB+TC`:

    season      model    price-only    myopic    random
    2023-24      2349        1921       2229       586
    2024-25      2128        2205       2287       605
    2025-26      2060        2143       2055       591

**One win in three.** This file has always treated the price-only baseline as the real bar —
FPL prices encode a great deal of expert judgement, and a model that cannot beat them has
earned nothing. Under the leaky horizon it cleared that bar comfortably. It no longer does.

Two things this is NOT. It is not a forecasting failure: calibration is unchanged at 0.995,
and the ablation below still finds attacking returns carrying real decision weight. And it is
not the chip change: BB+TC scores 2349 / 2128 / 2060 against BB+TC+free hit's 2340 / 2113 /
2070, a difference far too small to matter here.

What it is: the DECISION layer — horizon, transfer policy, hit discipline — was tuned against
a forward view that did not move, and on an honest one it is not currently adding value over a
constant heuristic. The forecasts are sound; what is built on top of them is not yet earning
its complexity.

### Ablation re-run lookahead-aware — attack survives, everything else is noise

    variant           2023-24  2024-25  2025-26    mean
    no_attack           -289     -239      -43    -190
    no_defcon              0        0     -143     -48
    no_clean_sheet      -125     +105      -49     -23
    no_saves            -133      +45      +55     -11
    no_cards             -71       -7     +129     +17
    no_appearance       -105      +81      +81     +19
    no_bonus             -76     +145      +25     +31

**Attacking returns are the only component with a consistent sign**, and they dominate. That
conclusion is now the most robust thing in this file: it has survived the double-gameweek
fixes, the goalkeeper fix, the defcon fix and the lookahead fix, having originally been
measured the wrong way round.

Everything else flips between seasons. Removing bonus, cards or appearance points HELPS on
average — which is a statement about noise, not about those components, and confirms the
existing entry that they are not worth further work. `no_defcon` is zero in two seasons only
because the statistic exists solely in 2025-26.

Magnitudes swing up to 145 points between seasons, much more than under the leaky horizon.

### The Haaland conclusion survives; its stated reason does not

    2025-26            season   owned   started   captained
    myopic h=0           2055    0/38       2       2
    leaky  h=6           2529   33/38      30      19
    honest h=6           2070   27/38      26      16

The model still buys and captains him, so the headline entry stands. But the reason recorded
alongside it — "with a MYOPIC valuation he starts 2 of 38 and the season falls from 2431 to
2135" — was a lookahead artefact. Honest myopic owns him ZERO times and still finishes within
15 points. Owning Haaland is right; it is not worth 296 points, and the horizon is not
justified by his case.

(Note `started > owned` for myopic: on a free-hit week `squad_ids` records the retained squad
while `scoring_ids` records the loaned one, which is the correct asymmetry.)

### One unexpected positive

Column A above (2598 / 2597 / 2537) is the *previous* forecasts run through *current* code, and
it sits well above the published 2551 / 2535 / 2454. The only intervening change is the
captaincy double-count fix of the same day, so that fix is worth **+47 / +62 / +83** — which was
not visible when it was made, since it was made as a correctness repair.

### Cost, and why the live path was never affected

`fpl simulate` now forecasts GW..GW+6 from each decision week: 38 decision weeks a season
rather than 38 gameweeks, at 2.6s each instead of 1.2s. Roughly 100s per season.

The LIVE path never had this bug. `cli.squad` builds its horizon with
`forecast_gameweek(future, planning=True)`, which forecasts every future week from today's
state — structurally what `forecast_horizon` now reproduces. That asymmetry is exactly why the
backtest flattered itself: it had access to a kind of information the live system cannot get,
and the headline totals were therefore never reproducible live. Now they are.

## Penalty xG is counted twice, live only (2026-08-14) — UNFIXED, and unmeasurable here

Found while checking whether free-kick and corner order could be modelled the same way as
penalties. The answer is no, and the reason is that the penalty term itself has a defect.

`xg_per90` comes from the archive's `expected_goals`, which is Opta's and **includes penalty
xG at 0.79 a spot-kick**. Nothing nets it out — `RATE_STATS` takes `expected_goals` whole. So a
designated taker's rate already embeds his penalties. `allocate_team_goals` then computes his
open-play share *from that inflated rate* and adds `expected_penalty_goals` on top:

    raw    = xg_per90 x 90s x finishing        <- already contains his penalty xG
    share  = raw / raw.sum()
    goals  = share x open_play_total + expected_penalty_goals

The team total is conserved by the rescale, so calibration cannot see it. What is wrong is the
SPLIT: the taker is over-weighted against his own team-mates. Measured on the GW1 forecast, the
explicit term is **9.5% of a taker's expected goals** (median across 55 takers).

**The backtest is structurally blind to this.** `penalty_share` is 0 on every historical row —
the archive has no set-piece column — so the double count never occurs there. The 1.0000
calibration, the ablation and the +399 result against form are all silent on it. It exists only
in the live path, which is the path nobody has measured.

Not fixed, because it cannot be fixed exactly: netting penalty xG out of the rate needs
penalties taken or scored per player, and the archive stores `penalties_missed` but neither of
those. An approximation is possible. Validating it is not — see the noise floor entry, and the
five principled fixes that have measured to nothing.

Recorded rather than patched, in line with "fix causes, not symptoms": a guessed correction on
top of an unmeasurable defect would make the code harder to reason about without making the
forecast better.

### The jobshare limitation, found the same day

`PENALTY_ORDER_SHARE = {1: 0.85, 2: 0.11, 3: 0.03}` maps ordinal position to a fixed share, so
two co-takers listed 1 and 2 can never come out even. Verified against a published takers list:
14 of 20 clubs agree exactly on the primary taker, but Arsenal (Saka and Gyokeres) and
Sunderland (Le Fee and Diarra) are true 50/50s and the model gives them 0.85/0.11 — a ~70%
over-allocation to Saka, a premium asset where it matters most.

`penalties_text` is ingested and unused; it carries FPL's own note, which is where a jobshare is
described. That is the route to detecting them from data rather than from a hardcoded list.

## The price-only baseline is degenerate and is RETIRED (2026-08-12)

This file and the README have treated "pick purely on price" as the bar worth clearing for
months. It cannot serve as one, and the reason is structural rather than a matter of degree.

Maximising a price-derived objective under a budget that binds at exactly £100m leaves a large
set of near-optimal solutions. Measured on 2023-24 GW1: **two solutions 0.8% apart in objective
(394.8 against 391.6) shared 5 of 15 players and scored 90 against 36 in the same gameweek.**
Which one the solver returns is decided by arbitrary detail in the valuation.

So price-only has no single value to quote. The 1921 / 2205 / 2143 recorded on 2026-08-12 was
one draw; running the same baseline through the ensemble gives 1275 / 1336 / 1056. Neither is
"the" price-only score.

**A claim was withdrawn because of this.** The regeneration that produced the first set had the
model on point-in-time forecasts and the benchmarks left on the leaky flat horizon, on the
reasoning that any lookahead they kept "flatters THEM, not us" and was therefore conservative.
It is conservative only if the two are otherwise comparable, and they were not — that run
compared an honest model against a leaky baseline drawn arbitrarily from a degenerate set, and
produced the headline "the model does not beat the price-only baseline". That was wrong.

The lesson is narrower and more useful than "check your baselines": **a baseline whose objective
is collinear with its binding constraint has no well-defined value.** Weakening a baseline is
not a safe direction to err in, because it makes the comparison meaningless rather than
conservative.

## The recent-form baseline (2026-08-12) — and the model clears it

`season_sim.form_scores` / `attach_form` / `benchmark_form`. Points per gameweek over a
player's last six, `shift(1)` so his own result never informs the squad picked to include him,
shrunk toward a price prior while the history is thin. The prior scale is a fixed constant, not
fitted, so the baseline carries nothing from the season it is scored on. At GW1 it reduces to
price, which is the information a real manager has in August.

It is a real bar, unlike the one it replaces. Mean per-gameweek rank correlation with realised
points:

    price               0.381
    recent form         0.641
    our model           0.692

Ensemble, 8 perturbed decision paths per season, paired within each path:

    paired vs form     mean_diff     se     95% CI          wins   adoptable
    horizon               +399.0   22.5   [+355, +443]      1.00      True
    myopic                +388.0   14.8   [+359, +417]      1.00      True
    price_only            -567.8   13.0   [-593, -542]      0.00      True

    per season            2023-24    2024-25    2025-26
    horizon vs form         +503       +270       +425
    myopic  vs form         +352       +417       +395

**The model beats it by +399, in every season, on all 24 of 24 paths.** This is the first
result of the day to satisfy both halves of the adoption rule, and the smallest per-season
margin is twelve standard errors clear of zero. The forecasting layer earns its keep.

Two further things fall out. Price-only sits **568 BELOW form**, so quoting it as a hard bar
understated this system by roughly 550 points a season for months. And the horizon-versus-myopic
gap is unchanged at +11 +/- 28 — **both** beat form comfortably, so the value is in the
forecasts, not in the multi-week optimisation built over them.

## Decision-layer parameter sweep (2026-08-13) — nine variants, NOTHING adoptable

The first sweep run on an instrument capable of resolving it: 9 variants x 3 seasons x 10
perturbed decision paths, paired within each path. Baseline is the incumbent
(`hit_bar` 4, `decay` 0.84, `captaincy_weight` 1.0).

    variant             mean_diff     se      95% CI        wins   signs      adoptable
    decay_0.70              +53.0   19.9   [ +14, +92]      0.67   - + +          no
    hit_bar_12              +30.8   15.5   [  +0, +61]      0.57   - + -          no
    hit_bar_8               +16.2   11.9   [  -7, +40]      0.43   - - +          no
    horizon (incumbent)       0.0      —            —          —   0 0 0           —
    hit_bar_9               -11.7   10.5   [ -32,  +9]      0.43   - - +          no
    no_captaincy_term       -19.3   11.8   [ -43,  +4]      0.33   - - +          no
    hit_bar_10              -23.3    6.4   [ -36, -11]      0.40   - - +          no
    hit_bar_6               -27.1   13.2   [ -53,  -1]      0.57   - + +          no
    decay_0.92              -46.3   13.5   [ -73, -20]      0.23   - + -          no

**Every single variant flips sign between seasons.** Four have pooled intervals excluding zero
and not one of them survives ground rule 2. The incumbent settings stay — not because they are
proven, but because nothing beats them reliably.

### The transfer-margin correction is now tested and rejected

`hit_bar` was built on the most stable finding of the previous session: the forecast margin
behind a transfer is overstated ~2.3x, with a slope of 0.436 holding in every season. The
implied bar of ~9 measures **-38 / -42 / +45** across the three seasons. The bias is real and
correcting it does not help.

That makes five principled fixes for measured biases that have produced nothing in a
constrained optimiser (bonus match-allocation, ownership calibration, rank-aware captaincy,
defcon overdispersion, and now the hit bar). Ground rule 3 has earned its place several times
over: **a bias that is real in a regression is not necessarily exploitable in an optimiser.**

### A correction to the previous session's inference

The 2024-25 diagnosis found the captaincy contribution consistently positive (+41 / +48) while
transfer aggression was consistently negative, and it was suggested that the armband term was
carrying the horizon and should be isolated. The direct test says otherwise:
`no_captaincy_term` is **-59 / -58 / +59**, pooled -19.3 with an interval spanning zero. The
captaincy term is as unproven as everything else around it. The earlier reading came from
decomposing a horizon-versus-myopic difference, which attributes to the armband any change
that flows through it; only toggling the term itself answers the question.

### An observation worth carrying forward

**2023-24 prefers the incumbent on every one of the nine variants**, while 2024-25 and 2025-26
each prefer several alternatives. It is also by far the least noisy season (baseline sd 5.8,
several variants under 2.0, against sd up to 97 in 2024-25). Two readings, not separable here:
the defaults were originally tuned on a leaky backtest that included 2023-24, so some residual
fit to it is plausible; or a near-deterministic season simply lands on a local optimum. Worth
remembering before treating 2023-24 as an equal vote.

## The simulator has a ~40-point noise floor (2026-08-12) — read this before trusting any margin

Attempting to act on the transfer-calibration finding below produced a result that matters far
more than the attempt did.

Perturb the FORECAST by 0.1% — a rounding error, smaller than any modelling change ever tested
here — leave the outcomes untouched, and re-run 2024-25:

    seed        0      1      2      3      4      5
    season   2128   2193   2128   2128   2193   2103      sd 38, range 90

**A meaningless perturbation moves a season total by up to 90 points.** The mechanism is path
dependence: a hair's difference flips one transfer decision, that changes the squad, and the
squad changes every subsequent decision for the rest of the season. Transfer counts stay almost
constant (64, 64, 64, 64, 64, 65) while points swing — so this is not "more trading", it is the
same policy landing on a different path.

The same thing shows up directly in a parameter sweep. Raising the optimiser's hit bar on
2024-25 gives 2128, 2144, 2111, 2088, **2023**, **2139**, 2113, 2294 at bars of 4, 6, 8, 8.5,
9, 9.5, 10, 12 — non-monotonic, with 116 points between adjacent settings while transfer counts
fall smoothly from 64 to 42.

### What this rules in and rules out

Every effect measured on season totals must now be read against a per-season noise floor of
roughly 40 (sd), so a three-season mean has a standard error near 22.

    effect                                   per season          verdict
    horizon lookahead                 -200 / -518 / -459    FAR outside — real
    attack ablation                   -289 / -239 /  -43    outside — real
    horizon vs myopic                 +111 / -174 /  +15    inside — unresolved
    valuation smoothing (hl=1)         -88 / +182 /  +78    inside — unresolved
    corrected hit bar (LOSO)           -54 /  +11 / +128    inside — unresolved
    Free Hit vs BB+TC                  -11 /  -16 /  +11    WELL inside — unresolved

**The two conclusions this session leaned on hardest survive; almost everything else is inside
the noise.** That is not a reason to distrust the lookahead fix — it is 5 to 13 times the floor
— but it does explain why three separate results collapsed on the per-season sign test today.
They were never resolvable in the first place.

### The consequence: n=3 was the wrong diagnosis

More seasons would help, but the binding constraint is that each season is ONE draw from a
chaotic decision path. The right instrument is an ensemble: run each variant over k perturbed
paths per season and compare distributions rather than point estimates. At ~80s a path, k=10
across three seasons is 40 minutes a variant — affordable, and it would shrink the standard
error by roughly a factor of three.

Note this is ground rule 1 ("never conclude from a single deterministic replay") applied to the
DECISION path rather than to outcomes. `repeat_sim` resamples outcomes while holding decisions
fixed, which is the opposite axis and the reason it never exposed this.

## Transfer-margin calibration (2026-08-12) — a real bias, not safely exploitable

Graded at the DECISION level rather than the season level: 111 transfer decisions across three
seasons, each with the forecast margin that justified it and its realised discounted six-week
return.

    fit of realised on forecast gain     slope   mean forecast   mean realised
    pooled  (n=111)                      0.436       13.99            9.43
    2023-24 (n= 37)                      0.433       15.34           12.46
    2024-25 (n= 37)                      0.499       12.97            8.81
    2025-26 (n= 37)                      0.368       13.66            7.03

**The forecast margin is overstated by roughly 2.3x, and that slope is stable in every season**
— which is more than can be said for any season-total effect. A move the optimiser expects to
gain 14 points gains about 9.4. Correlation is only 0.27, so the margin is a weak signal as
well as a biased one.

This matters because the MILP compares `gain` directly against `hit_cost = 4`. If gain is
inflated 2.3x then a transfer clearing the nominal bar is really worth about 1.7 points, and
the policy takes hits it should not.

**Acting on it is not supported.** `hit_bar` was added to `simulate_season` (default `None` =
nominal 4) and tested leave-one-season-out, fitting the slope on two seasons and scoring the
third: -54 / +11 / +128, mean +28. But the pooled value 9.0 gives -36 / -105 / +56, mean -28 —
the opposite sign — and the sweep above shows why: the response surface is noise. The apparent
LOSO gain was luck in which value each fold happened to pick.

So the bias is established and the remedy is not. `hit_bar` stays at the nominal 4, which is
worth noting was never itself measured — it was assumed by taking the game's real penalty as
the optimiser's bar, which is correct only if the forecast is unbiased. Both the status quo and
the correction are unevidenced; the status quo is simpler.

### A limitation of the decision-level instrument, stated so nobody over-reads it

`bought_minus_sold` values each move against "hold the sold player six more weeks". With ~63
transfers over 38 gameweeks those windows overlap heavily and the counterfactual is wrong —
you cannot hold a player you already sold. So the per-decision returns DO NOT sum to a season
effect, and indeed they look uniformly positive (+7.5 net per paid transfer) in a policy whose
season totals are mediocre. Use it for calibration slope, which is a within-decision quantity;
do not use it to price a policy.

## Self-audit (2026-08-12) — NOT an independent review

An independent reviewer was commissioned and failed on a session limit before producing any
findings. What follows is the author auditing their own work, which is worth less. It is
weighted toward checks where authorship bias matters least — factual properties of the code —
and the judgement calls it cannot answer are named at the end.

### 1. Auto-substitutions are not modelled. Real, small, and in our DISFAVOUR.

`optimise/squad.py` cites autosubs to justify weighting the bench, but `score_gameweek` never
applies them: a starter who plays no minutes simply scores zero, where FPL would substitute a
bench player who did play.

Measured on 2025-26: **9 starters blanked across the season, 0.2 per gameweek**. At roughly 3
points for a player who appears, autosubs are worth on the order of 27 points a season, about
1%. Direction matters — this makes the backtest UNDERSTATE our score, so it partly offsets the
saturation problem rather than adding to it. Low priority, but it should not be claimed that
the scoring is faithful to the rules, because on this point it is not.

### 2. Researcher-mediated leakage in the fitted constants. Real, quantified, immaterial.

Several constants were measured from data that INCLUDES the seasons they are scored on:
`ASSIST_RATE` (0.905), `GOAL_DISPERSION`/`ASSIST_DISPERSION` (2.85/1.88),
`BONUS_GIVEN_INVOLVEMENT`, `BUCKET_MINUTES`, `RED_CARD_FRACTION`. This is the leakage class
that walk-forward discipline on models does not catch, because it enters through the author.

Quantified for the one that most affects squad selection:

    test season   strictly walk-forward ASSIST_RATE   value used   difference
    2023-24                 0.9015                      0.905        +0.39%
    2024-25                 0.9002                      0.905        +0.54%
    2025-26                 0.9009                      0.905        +0.45%

Under half a percent, because the quantity is stable across seasons (0.893-0.934). For scale,
the bug this constant replaced was 0.72 — a 26% error. The leak is real and should be stated;
it is not material.

**One that is more than cosmetic:** `SKILL_DISPERSION` is fitted so the simulated top-10k
threshold matches ~2500 on 2024-25, and 2024-25 is then one of the seasons whose rank is
reported. That specific rank is circular. Points are unaffected.

### 3. Documented numbers had gone stale. Corrected.

Data was regenerated four times on 2026-08-11 and several docstring measurements predate the
last round. Re-measured 2026-08-12:

    quantity                    documented    actual
    distribution P(>=5) ratio      0.95        1.04
    distribution P(>=8) ratio      0.80        0.84
    distribution P(>=10) ratio     0.78        0.81
    overall points calibration     1.0000      1.0091

`models/distribution.py` is corrected. **The component ablation table below was measured
before the goalkeeper fix and has NOT been re-run** — attack was the dominant component then
and the fix made attack more accurate, so the ordering is unlikely to have reversed, but the
magnitudes should not be quoted without re-running `ablation_variants` through `fpl repeat`.

### What this audit cannot answer

The reviewer was asked whether there is a pattern of rejecting results that threatened a
favoured conclusion. The author is the wrong person to answer that, and it is left open. Also
unexamined here: whether the MILP constraints are genuinely the FPL rules, the Dixon-Coles fit
and odds de-vigging, and whether the tests encode the same misconceptions as the code.

## Findings that should change how you read the output

1. **Finishing skill is not detectable.** Spread in goals-per-xG among forwards is *smaller*
   than Poisson noise alone would produce.
2. **The market matches or beats our match model.** Do not read the blend as an edge.
3. **Attacking returns are the DOMINANT component, and it is now the most robust finding
   here** — the only ablation component whose sign holds in every season, having survived the
   double-gameweek fixes, the goalkeeper fix, the defcon fix and the lookahead fix.
   See the re-measured ablation.
   This reverses the previous entry here, which claimed attack ranked last behind minutes,
   clean sheets, cards, bonus and DefCon, and concluded that "most FPL analysis focuses on
   exactly the wrong thing". That ordering was measured under the double-gameweek bugs and on
   one deterministic replay per season. Conventional FPL emphasis on attacking returns is
   right and this project was briefly wrong to doubt it.
4. **Saves, cards, bonus and appearance points are not distinguishable from zero** in the
   decisions they drive. Not worth further work on current evidence.
5. **The Haaland problem is resolved — it selects him now** (34 of 38 gameweeks in 2025-26,
   captained 18 times). This entry previously read "the model does not select Haaland at
   £15.5m", flagged as the most checkable disagreement with FPL consensus. Re-tested
   2026-08-11 after the attack fixes:

       forecast     6.16 pts/gw predicted against 6.64 realised (7.8% under)
       season       222 predicted, 239 realised — the highest predicted total in the game
       rank by expected points     7th
       rank by points per £m     186th

   Two things had to be true. The forecast had to rate him correctly, which took fixing the
   penalty leak and the position-grouped shrinkage; and the valuation had to be multi-week.
   Tested by ablation: with a MYOPIC single-gameweek valuation he starts **2 of 38** and the
   season falls from 2431 to 2135. The captaincy premium turns out not to be what pays for
   him (34 starts with it, 35 without) — the horizon is.

   **Points-per-million is the trap.** He is 186th on it and 7th on expected points, and for a
   player held all season the per-week ratio is the wrong lens: the fee is paid once and the
   return accrues 38 times. A myopic optimiser cannot see that, which is why it refused him.
