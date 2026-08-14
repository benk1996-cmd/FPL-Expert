# FPL Expert — Project Plan

**Goal:** maximise total FPL points over a season by (a) forecasting a *distribution* of points
for every player in every upcoming gameweek, and (b) solving a multi-gameweek optimisation for
squad, starting XI, captain, transfers and chips.

**Season target:** 2026/27 (starts ~mid-Aug 2026).

---

## 0. Guiding design decisions

These shape everything downstream, so they are stated up front.

1. **Decompose, don't regress on total points.** A single model of `total_points ~ features`
   fits mostly noise. Points are a deterministic function of countable events
   (minutes, goals, assists, clean sheets, saves, bonus, cards). Model each event process,
   then compose. This gives interpretability, better sample efficiency, and a natural
   distribution rather than a point estimate.
2. **Minutes dominate.** The largest source of variance in a player's score is whether they
   play 0, ~25 or 90 minutes. A mediocre minutes model wrecks an excellent attacking model.
   This gets first-class treatment.
3. **Predict distributions, not means.** Captaincy, differentials and chip timing are decisions
   about the *upside tail*. We carry `P(points = k)` per player-gameweek, not just `E[points]`.
4. **Team-level first, player share second.** Clean sheets and goals conceded are properties of
   the match, not the player. Simulate the match scoreline, then attribute to players.
5. **Point-in-time correctness or the backtest is a lie.** Every feature used to predict GW *t*
   must be computable using only data available *before* the GW *t* deadline. This is the single
   easiest way to build a model that looks brilliant and loses money.
6. **Optimisation is where the points are won.** A good forecast with a greedy "highest EP"
   transfer rule underperforms a mediocre forecast with a proper multi-period planner,
   because transfers are a scarce, path-dependent resource.

---

## 1. Repo layout

```
FPL Expert/
├── config/
│   ├── config.yaml            # paths, horizon, decay, solver, seasons
│   └── scoring_rules.yaml     # points table + squad rules (VERIFY each season)
├── data/
│   ├── raw/                   # immutable API/scrape dumps, partitioned by pull timestamp
│   ├── interim/               # parsed & typed, still per-source
│   ├── processed/             # model-ready feature tables (parquet)
│   └── external/              # odds, understat, reference tables
├── notebooks/                 # exploration only; nothing importable lives here
├── src/fpl_expert/
│   ├── config.py              # typed config loading
│   ├── data/                  # ingestion & storage
│   │   ├── fpl_api.py         # bootstrap-static, fixtures, element-summary, live, entry
│   │   ├── historical.py      # vaastav/Fantasy-Premier-League archive loader
│   │   ├── understat.py       # shot-level xG/xA
│   │   ├── odds.py            # bookmaker match odds -> team strength prior
│   │   └── snapshot.py        # point-in-time snapshot writer/reader  <-- critical
│   ├── features/
│   │   ├── fixtures.py        # opponent, home/away, congestion, blanks/doubles
│   │   ├── team_strength.py   # rolling attack/defence ratings, odds-implied
│   │   ├── player_form.py     # per-90 rates, EWMA, regression to position/team priors
│   │   └── build.py           # assembles the training matrix
│   ├── models/
│   │   ├── minutes.py         # P(start), P(bench appearance), E[mins | appear]
│   │   ├── match_sim.py       # Dixon-Coles / bivariate Poisson -> scoreline grid
│   │   ├── attack.py          # per-90 goal & assist rates, team-share allocation
│   │   ├── saves.py           # GK saves given shots faced
│   │   ├── defcon.py          # defensive-contribution threshold probability
│   │   ├── bonus.py           # BPS components -> rank -> 3/2/1
│   │   ├── cards.py           # yellow/red risk
│   │   └── points.py          # ASSEMBLER: composes all of the above into a pmf
│   ├── optimise/
│   │   ├── constraints.py     # squad legality: budget, positions, 3-per-club
│   │   ├── milp.py            # multi-period squad/lineup/captain/transfer solver
│   │   ├── transfers.py       # price change & selling-price accounting, FT bank
│   │   ├── chips.py           # wildcard / free hit / bench boost / triple captain
│   │   └── risk.py            # mean-variance & rank-chasing objectives
│   ├── backtest/
│   │   ├── walkforward.py     # rolling-origin model evaluation
│   │   ├── season_sim.py      # end-to-end agent vs benchmarks over a full season
│   │   └── metrics.py         # calibration, rank corr, top-N precision, points delta
│   ├── reporting/
│   │   └── gw_report.py       # weekly decision brief
│   └── cli.py                 # `fpl update | predict | plan | backtest | report`
├── tests/
└── scripts/
```

---

## 2. Workstreams — we will go through these one at a time

### Phase A — Foundations

**Item 1. Rules, scope & success metrics**  — *rules half DONE (verified 2026-08-09)*
2026/27 scoring and squad rules are pinned in `config/scoring_rules.yaml`, verified against
`game_config` in the FPL API (the public rules page is a JS app and can't be scraped).
Re-verify at the start of each season; the API is the canonical source.
Still to define — what "good" means before modelling:
- Model-level: calibration of clean-sheet and start probabilities; Spearman rank correlation
  of predicted vs actual points within position; MAE on players with >0 predicted minutes.
- Decision-level: simulated season points vs benchmarks (template team, FPL's own `ep_next`,
  last-season PPG heuristic, random legal squad).

**Item 2. Data ingestion**  — *DONE (free sources only)*
- Official FPL API (free, no key) — players, fixtures, prices, ownership, injury news.
- `vaastav/Fantasy-Premier-League` — 185,964 player-gameweek rows, 2019-20..2025-26.
- football-data.co.uk — free CSVs, no key: results plus opening AND closing 1X2,
  over/under 2.5 and Asian handicap odds. 380 matches per season, all 7 seasons.

*Decision — where xG comes from.* Understat is unreachable (connection refused, not a
timeout) and FBref returns Cloudflare 403, so neither can be scraped or verified. This
turns out not to matter: **the FPL API already carries Opta-derived `expected_goals`,
`expected_assists` and `expected_goals_conceded` per player per gameweek from 2022-23**,
and team xG per match is exactly reconstructable by summing player xG (verified: 100%
of fixtures). What we lose is shot-level detail — shot locations and big-chance flags —
which would refine finishing models but is not required for Items 6-7. Revisit only if
the attacking model proves shot-quality-limited.

*Odds: opening vs closing.* Closing odds are sharper because they embed late team news —
but the FPL deadline falls BEFORE the closing line exists, so closing odds are NOT
legitimate features for a point-in-time backtest. Both are ingested and labelled
(`p_home_open` / `p_home_close`); Item 3 governs which is admissible where. Using
closing odds to predict a gameweek we "decided" pre-deadline is a subtle, and very
flattering, leak.

Deliverable: idempotent, cached, rate-limited pullers writing partitioned parquet.
Effective ownership — see Item 2b.

**Item 2b. Effective ownership (`data/ownership.py`)**  — needed by the rank layer (Item 13)
Two tiers:
- *Overall EO*: `selected_by_percent` in bootstrap-static. Free, live, no scraping.
- *Top-10k EO*: page the Overall league (id 314, 50 entries/page, public, verified 200) for
  entry IDs, then pull `entry/{id}/event/{gw}/picks/` for a uniform sample of ~1,000 of them
  (~1.5% SE at p=0.5 — ample). ~10 min/week at 2 req/s.

Two constraints that shape the design:
1. **Picks are only exposed after the deadline** (`.../event/{gw}/picks/` 404s before it).
   EO for the gameweek being decided is therefore always a *forecast*: last gameweek's picks
   plus `transfers_in_event`/`transfers_out_event` drift. Squad ownership is sticky (>90%
   carryover); **captaincy is not, and captaincy is the half that drives rank** — it needs its
   own small model, fit on observed captaincy share vs our EP output.
2. **Current-season rank is meaningless early** — after GW1 the "top 10k" is whoever got
   lucky once. RESOLVED, not waited out: entry IDs **do** persist across seasons (verified —
   `entry/{id}/history/` returns a `past` array with each season's finishing rank), so
   `skilled_cohort()` filters candidates on *prior-season* rank. That gives a proven-skill
   cohort from GW1 with no stabilisation period, and it is a better target than the raw
   top-10k anyway: "managers who are good" rather than "managers who started well".

*Status: DONE and tested; execution waits only on the season starting.* `fpl ownership --gw N`
runs it for a completed gameweek. EO is defined as the mean points multiplier (0 bench,
1 starting, 2 captain, 3 triple captain), which is the quantity rank actually depends on.

**Item 3. Point-in-time snapshots**  — *DONE*
`data/snapshot.py`. Three parts:

1. **Capture** (`take_snapshot`) — archives bootstrap, fixtures and current odds under one
   UTC stamp, plus a manifest. Bypasses the HTTP cache deliberately: a cached bootstrap
   would record stale availability under a fresh timestamp, which is worse than no
   snapshot because it looks valid. The manifest's key field is `taken_before_deadline`.
   What is genuinely unrecoverable, and therefore why this exists: `news`,
   `chance_of_playing_next_round`, `status`, `now_cost`, `selected_by_percent`,
   `kickoff_time` and deadline odds are all overwritten in place and archived by nobody.
2. **Enforcement** (`PointInTime`) — the only sanctioned reader for a gameweek being
   predicted. Resolves to the latest snapshot taken *before* that deadline and raises
   `MissingSnapshotError` rather than falling back to a later one. It exposes no accessor
   for closing odds or results, so lookahead is structurally impossible rather than
   merely discouraged. `assert_no_post_deadline_columns` is the belt-and-braces check on
   assembled frames, catching the realistic accident: a join that drags `p_home_close`
   along for the ride.
3. **Scheduling** — `fpl snapshot --if-due` walks a ladder of checkpoints towards each
   deadline (48h, 24h, 6h, 2h, 0.5h), capturing at most once per band, so it is safe to run
   hourly. Escalating rather than one-and-done for two reasons: a capture 24h out misses the
   Friday press conferences and `PointInTime` always uses the latest pre-deadline snapshot,
   so a later one is strictly better; and a job on a personal machine only fires while that
   machine is awake, so several chances beat one.
   `scripts/register_snapshot_task.ps1` registers the Windows task.

Odds ingestion is now physically split: `external/odds_features` (opening only, reachable
from `PointInTime`) and `external/odds_eval` (closing + results, reachable only from
Item 11's evaluation code).

**FIRST HARD DEADLINE: GW1 at 2026-08-21 17:30 UTC.** Miss it and GW1 can never be used
for point-in-time evaluation.

**Item 4. Feature engineering**  — *core DONE; assembler deferred to Item 5*
- `features/rates.py` — Gamma-Poisson empirical-Bayes per-90 rates with exponential recency
  weighting. One estimator covers both hard cases: a two-goal cameo is shrunk toward the
  position/price prior, and a player with no history at all falls back to that prior exactly,
  with `confidence = 0`. `confidence = e/(e+beta)` is the low-confidence flag from decision 2.
- `features/fixtures.py` — team-fixture grain, rest days (congestion), and an explicit
  team x gameweek grid so blank gameweeks are zeros rather than missing rows.

*Deliberately deferred:* the final feature assembler (`build.py`). Building it now would mean
guessing what the minutes model needs; it will be written in Item 5, driven by real
requirements rather than speculation.

**Bug worth remembering** (regression-tested in `test_rates.py`): with exponential decay the
Poisson variance of a weighted count scales with `w^2`, not `w`. Treating decayed exposure as
raw exposure over-states the noise, drives the fitted between-player variance negative, and
collapses every prior onto its floor — on real data this handed the entire league an identical
0.417 goals/90 with zero confidence, which looks plausible enough to ship. Kish rescaling
(`e^2/e2`) fixes it. Any future weighted estimator here needs the same correction.

*Decisions taken:*
1. **Produce a best-effort GW1 forecast.** Cold-start machinery is in scope: carry last
   season's rates forward, fall back to price/position priors for newcomers.
2. **No-history players get a price+position prior with a confidence weight.** FPL's own
   price is an informative expert estimate. Two clubs are genuinely cold — Coventry City
   and Hull City have no Premier League data in the 2019-20+ archive. (Ipswich Town does
   have 2024-25 data; it was hidden by an archive/FPL name mismatch — "Ipswich" vs
   "Ipswich Town". The archive's team names drift between seasons and must be mapped
   through `data/teams.py` like football-data's are.)
3. **Fully automatic — no manual override channel.** Note this does NOT mean no team news:
   FPL's own `news` and `chance_of_playing_next_round` fields are captured automatically in
   every snapshot. What is excluded is human judgment beyond what FPL publishes. Two
   consequences: Item 11/14 results are fully reproducible with no human input to account
   for, and Item 5 must extract as much as possible from those two fields, since they are
   now the only availability signal.

### Phase B — The forecasting engine

**Item 5. Minutes model**  — *DONE*
`models/minutes.py`. Three-bucket multiclass (0 / 1-59 / 60+) rather than a minutes
regression, because the FPL thresholds are discontinuous and the distribution is bimodal —
predicting 45 minutes when the truth is "cameo or full match" gives the least likely answer.

*Walk-forward results* (train on prior seasons, test on the next; `fpl minutes`):

| model | log loss | Brier | accuracy | ECE on P(60+) |
|---|---|---|---|---|
| minutes model | **0.4847** | **0.2690** | 0.8094 | **0.0118** |
| transition baseline | 0.5834 | 0.3086 | 0.7921 | 0.0196 |

Beats the baseline in all four test seasons on every metric. Calibration gaps stay within
±2pp across all ten probability bins, and the top bin is essentially exact (0.927 predicted
vs 0.927 observed) — which is the band captaincy decisions are made in.

Note accuracy barely moves (0.79 -> 0.81) while log loss improves 17%. That is the point:
always predicting "doesn't play" scores 58% accuracy and is worthless. Accuracy is not a
metric for this problem.

*Availability is deliberately not a training feature.* The archive has no history of
`chance_of_playing_next_round` or `status` — they exist only in our forward-looking snapshots
— so learning them would be train/serve skew in one direction or the other.
`apply_availability_gate()` instead applies the published percentage at inference as an
explicit multiplicative prior on playing, preserving the short/long ratio (a doubtful player
who would have started still starts *if* fit). It is already a probability, so it needs no
fitting, and keeping it outside the model makes its effect visible.

*Feature keying:* players are keyed by NAME, not `element` — FPL reassigns element ids each
season, so keying on the id would make every player a debutant each August and discard most
of the signal. `minutes_lag1` alone carries 66% of model gain.

*Data cleaning found:* `AM` rows (defunct Assistant Manager entries, permanently zero minutes)
must be dropped or they drag every base rate down; `GK`/`GKP` labels drift between seasons.

**Item 6. Team match model**  — *DONE*
`models/match_sim.py`. Dixon-Coles fitted by weighted MLE with exponential time decay,
blended with goal expectations recovered from the opening odds. Emits a full scoreline grid
per fixture and, from it, `P(clean sheet)`, the goals-conceded distribution, the expected
conceded penalty (-1 per 2), and expected goals for. On real data: home advantage 0.178,
rho -0.105, and the strength rankings are correct (City/Liverpool/Arsenal attack;
Arsenal/City defence).

*Blend weight, measured rather than assumed.* Rolling-origin over 2,085 out-of-sample
matches, refitting every 14 days:

| market weight | outcome log loss | clean-sheet ECE |
|---|---|---|
| 0.0 (Dixon-Coles alone) | 0.9864 | 0.0199 |
| 0.8 | 0.9644 | **0.0125** |
| 1.0 (market alone) | **0.9629** | 0.0128 |

**The market is at least as good as our model and the blend adds nothing significant** —
bootstrap CI on the log-loss difference between weight 1.0 and 0.8 is [-0.0036, +0.0001],
straddling zero. Weight 0.8 is kept anyway for a practical reason rather than a statistical
one: odds are not always available (no book has priced GW1 twelve days out, so those
forecasts run on Dixon-Coles alone), and a model that degrades gracefully beats one that
has nothing to say. Do not read the blend as an edge over the bookmaker; it is insurance.

*A bias investigated and dismissed.* Pooled clean-sheet predictions looked over-confident in
the top bin — 0.540 predicted against 0.44 observed. Splitting by season showed the gap
flips sign (+0.137 in 2024-25, -0.078 in 2021-22) on 32-88 matches per season, and
walk-forward isotonic recalibration moved mean ECE only 0.0313 -> 0.0286 while making
2024-25 worse. It is small-sample noise, not systematic bias, so **no recalibration layer was
added** — fitting one would have been fitting noise. Worth re-checking in Item 11 once more
seasons of forecasts exist.

*Promoted clubs* with no Premier League record are treated as exactly league-average
(attack and defence of zero under the sum-to-zero constraint) rather than dropped — the same
cold-start principle as the rate model.

**Item 7. Player attacking returns**  — *DONE*
`models/attack.py`. Built top-down: the match model already knows the team's expected goals
for this fixture, so the job is allocating them across players by shrunk rate x expected
minutes. Two properties come free that bottom-up would have to engineer — team totals stay
consistent with the match model, and fixture difficulty is inherited automatically.
Penalties are allocated separately from `penalties_order` because they are lumpy and
concentrated; averaging them across a squad would erase the edge on designated takers.

**Finding: finishing skill is not detectable in this data.** The spread in goals-per-xG
among forwards (weighted spread 46.0 for players with meaningful xG) is *smaller* than
Poisson noise alone would produce (54.8), so every player is correctly estimated at the
group conversion rate and the multiplier sits at ~1.0. This vindicates choosing the shrunk
multiplier over either fixed alternative: it TESTED the hypothesis instead of assuming an
answer, and the answer is "pure xG". `GammaPrior.skill_detected` now exposes this, because
a collapsed prior and a genuine signal look identical downstream.

**Two measurement bugs found here, both of which produced plausible-looking output.**
1. *Noise scale.* The Gamma-Poisson correction assumes counts. xG is a sum of probabilities
   and carries far less sampling noise — MEASURED by splitting each player's matches in half:
   `goals_scored 0.92`, `assists 0.83`, `expected_goals 0.29`, `expected_assists 0.17` of the
   Poisson expectation. Applying the full correction to xG collapsed every prior and handed
   the whole league an identical 0.397 xG/90. Now a per-stat `NOISE_SCALE` in `features/rates.py`.
2. *Detection threshold.* The noise correction is itself an estimate with sampling error of
   about `mean*sqrt(2n)`. A floor-based test for "is there real variation" fires on pure
   chance; `skill_detected` now requires the excess to clear two standard errors.

*SPRINT simplifications to revisit:* penalty-order shares (0.85/0.11/0.03) are assumed rather
than fitted, since the archive has no set-piece column to fit against; the assist rate (0.72
of goals) and penalties-per-match (0.11) are league constants rather than team-specific; and
xG is not decomposed into penalty and open-play components.

**Items 8, 9, 10, 12** — *DONE at SPRINT depth*

- **Item 8** `models/secondary.py` — saves, defensive contributions, cards, clean sheets.
  All threshold-based, so expectations are taken over the count distribution rather than
  applied to the mean: a keeper expected to make 2.8 saves scores 0 if you apply "1 per 3"
  to the average, but clears three saves over half the time. Saves scale with the opponent's
  expected goals, so save points and clean-sheet points correctly pull against each other.
- **Item 9** `models/bonus.py` — the crudest model in the system, deliberately. A proper
  version rebuilds BPS from components and simulates the within-match ranking; that cannot
  be calibrated yet because FPL retuned BPS for 2026/27, so every historical `bps` is on
  superseded rules. Uses realised bonus rates scaled by team strength instead.
- **Item 10** `models/points.py` — applies the configured scoring table to every component.
  Ordering is load-bearing: minutes gate everything, then team totals, then allocation.
- **Item 12** `optimise/squad.py` — exact MILP over squad/XI/captain. Solves in under a
  second, so there is no reason to approximate. Greedy selection fails here: it spends the
  budget on premiums and cannot then fill the remaining slots legally.

*Validated against reality* — model component shares vs the archive: appearance 59.4% vs
59.5%, goals 17.9% vs 16.5%, assists 7.6% vs 9.0%, bonus 7.1% vs 7.6%.

**Bug found here, and it is one this project built tooling to prevent.**
`decayed_totals` zero-fills missing values, so computing `defensive_contribution` rates over
2022-23 onward accumulated three seasons of minutes against zero counts and diluted the rate
several-fold. That is precisely the "a zero-filled column teaches the model nobody tackles"
trap `historical.coverage_report` exists to expose — and the pipeline walked into it anyway.
Rates are now computed per availability window (`pipeline._stat_seasons`).

**Open concern for Item 11/14: the model does not select Haaland.** At £15.5m he loses to
cheaper forwards. That may be correct, or it may indicate the top-down allocation caps elite
strikers too hard, or that the crude bonus model under-rewards them. It is the most
checkable disagreement between this system and FPL consensus, and it should be the first
thing the season simulator is pointed at.

> **Data constraint found in Item 2:** the DefCon components (CBI, tackles, recoveries) exist
> only from 2025/26 — ONE season, ~29.7k rows. Everything before that is genuinely missing,
> not zero (`fpl history` prints the coverage report; `interim/history_coverage`). So the
> DefCon model must be deliberately simple — a per-90 rate with heavy shrinkage toward
> position/role priors — because there is not enough data to support anything richer.
> Fitting a flexible model here would be the most likely place for this project to fool
> itself. The raw counts are unaffected by the 2026/27 BPS retune, so the one season is
> at least clean.

**Item 9. Bonus points**
Model BPS from its published components, then convert to 3/2/1 by simulating the within-match
BPS ranking. Bonus is ~8-10% of total points and is systematically underrated by simple models.

**Item 10. Points assembler**
Compose all components into a per-player, per-gameweek probability mass function over points.
Also emits mean, variance, `P(haul ≥ 10)` and captaincy-relevant tail statistics.

**Item 10 status — DONE.** `models/points.py` composes the mean; `models/distribution.py`
composes the full pmf by exact discrete convolution, conditional on the minutes bucket, and
convolves double gameweeks. Emits floor, median, ceiling, `P(blank)`, `P(haul ≥ 10)` and
`P(≥ 15)`. Mean agrees with the assembler at r=0.997 — two independent paths to one number.

Building it found and fixed a tail failure worth more than the layer itself: hauls were
under-predicted 2.5x because bonus was modelled independently of returns (it is 24x more
likely when a player returns) and counts were Poisson rather than overdispersed. See
`DECISIONS.md`.

**Item 10b status — DONE.** `backtest/repeat_sim.py`, `fpl repeat`. Replays seasons over
resampled outcomes so that comparisons carry a standard error instead of being single
deterministic numbers. Variants are scored on identical draws within each replay, so the
comparison is paired and the season-to-season swing cancels.

**Item 13b status — DONE.** `backtest/field_sim.py` rebuilt against an exact anchor: ownership
counts pin the average manager's squad points with no modelling in between, and the simulator
reproduces them within 0.3%. Managers now persist AND transfer (fixed uniforms against moving
ownership targets) and differ in skill, so the field has a top end to rank against. Rank is
measurable again — but it is far more sensitive to modelling choices than points are, and
should be read as indicative. See `DECISIONS.md`.

**Item 11 status — DONE.** `backtest/validate.py`, `fpl validate`. Measured on the full
2025-26 replay (30,575 player-gameweeks): expected-points Spearman **0.724**, minutes
Spearman 0.778, P(60+ minutes) ECE **0.0093** (predicted 0.263 vs observed 0.266),
P(any minutes) ECE 0.0127, top-20 precision 0.155 — roughly 6x random. Calibration is the
strong suit; top-N precision is low in absolute terms because weekly FPL hauls are close to
irreducibly noisy.

**Item 13 status — DONE (layer built, weight left at zero).** `optimise/risk.py`.
`lambda_rank` discounts expected points by effective ownership, so template players lose the
value that cancels against the field. `captain_choice` applies the same logic where the two
objectives diverge most sharply. **The weight stays 0 until it can be tuned against a
rank-aware simulator** — tuning it now would be guessing, and it needs live top-10k ownership
which does not exist until the season starts.

**Item 11. Validation & calibration**
Walk-forward backtest by gameweek (never random CV). Reliability diagrams for the probability
outputs, isotonic recalibration if needed. Compare against the baselines from Item 1.

*Decision — the closing line as a benchmark.* Closing odds are excluded from features
(they postdate the deadline) but used as an evaluation instrument, because they are a far
lower-variance quality signal than match outcomes. To detect a 7pp clean-sheet bias:
~158 fixtures (15.8 GWs) from outcomes, versus <1 gameweek from closing-line divergence —
19% vs 100% power after a single gameweek. Three weekly metrics:
1. signed divergence (ours − closing) — persistent drift means a bug, not edge;
2. head-to-head Brier/log-loss vs outcomes, ours against the closing line — the honest
   scoreboard for whether we add anything over the market;
3. largest per-fixture disagreements — usually the fingerprint of missed team news.

**Risk to manage:** watching this weekly and then tuning is researcher-mediated leakage —
the information reaches the model through our hands rather than through a column, and it
is invisible in the code. Contain it by treating CLV as monitoring with pre-committed
alert thresholds, and by judging every tuning decision against the Item 14 simulator
running on admissible data only.

### Phase C — Decision making

**Item 12. Optimiser**
Multi-period MILP over a rolling horizon (~5-8 GWs): binary variables for squad membership,
starting XI, captain, vice, transfers in/out per GW. Objective = time-decayed expected points
minus transfer hits. Constraints: £100m budget with correct selling-price rules, 2/5/5/3 squad,
valid formation, max 3 per club, free-transfer accumulation, bench-order autosub value.

**Item 13 status — chips DONE, risk layer built but untuned.**
`optimise/chips.py` values each chip and decides when to play it. Backtested across four
seasons: **+495 / +128 / 0 / +11 season points, mean +158** — positive or neutral in every
season.

*Chips are an optimal-stopping problem, not a valuation one.* Knowing a chip is worth 18
points this week is useless without knowing whether 24 is likely later and how many chances
remain. Each chip is a different quantity and conflating them is the usual error: Triple
Captain adds ONE extra copy of the captain's score (the armband already doubles), so valuing
it at 3x overstates it threefold.

**The finding that made it work.** The first policy played Bench Boost in GW2 of 2025-26 for
+21 when GW33 was worth +144, and chips overall cost **-73 points**. The cause was that the
stopping rule compared "this week" against the median observed week, and early in the season
it has almost no history — so its estimate of the future was anchored on the present.
Encoding that the BEST remaining week is worth roughly 2x a typical one for Bench Boost and
Triple Captain (a double gameweek: players play twice) and 2.5x for Free Hit (a blank: the
baseline collapses while the best available XI does not) turned -73 into +158 mean.

*Known weaknesses:* the multipliers proxy for fixture structure that a live planner could
read straight off the calendar, since double and blank gameweeks are announced weeks ahead —
a fixture-aware version would be strictly better. Free Hit still loses points when forced out
at expiry. And the wildcard rebuild uses current prices rather than selling prices, slightly
overstating its budget.

**Item 13. Chips & risk**
Chip timing as part of the same optimisation (bench boost and triple captain want double
gameweeks; free hit wants blanks).

*Decision taken:* the objective is expected points at the core, with a configurable
rank-aware layer (`optimise.objective.lambda_rank`, default 0). Rationale: maximising
E[points] and maximising P(high rank) are different problems, because rank depends on
`X - Y` (you minus the field) and the template largely cancels. A simulation of captaincy
alone showed a strategy with 38 FEWER expected points being 4x more likely to finish top 1%
and ~59x more likely to finish top 0.01%. Which objective is right depends on the target:
around top-100k, lambda=0 is near-optimal; chasing top-10k, the rank layer is mandatory.
Needs Item 2b (EO) and Item 10 (distributions) — not computable from point estimates.

**Item 14. Season simulation harness**  — *DONE, and built BEFORE deepening the sprint
components so that effort could be aimed by measurement rather than intuition*
`backtest/season_sim.py` + `backtest/historical_forecast.py`. Replays a full season using
only pre-deadline information, choosing the XI on forecast (never on outcome).

*2025-26, all 38 gameweeks, STRICT (minutes model retrained on prior seasons only):*

| | season points | per GW |
|---|---|---|
| model | **1901** | 50.0 |
| price-only baseline | 1116 | 29.4 |
| random legal squad | 282 | 7.4 |

> An earlier figure of 2086 was inflated by a leak: `simulate` loaded the saved minutes
> model, which is fitted on every season including the one being replayed. Worth 185 points
> (9%). Point-in-time discipline had been enforced on data but not on fitted models —
> a model is data too. Now retrained per season by default.

*Component ablation — season points lost when each component is removed:*

| removed | delta |
|---|---|
| appearance (minutes) | **-433** |
| clean sheets | **-374** |
| cards | -239 |
| bonus | -224 |
| defensive contributions | -205 |
| attacking returns | -180 |
| saves | -7 |

Reading: minutes and clean sheets dominate, exactly as the design assumed. **Attacking
returns rank LAST of the meaningful components** — counter to where most FPL analysis
focuses, and the strongest argument for having built the simulator first. Saves are
negligible and not worth further work. Cards score surprisingly high; the likely explanation
is that card rate proxies for playing style and minutes rather than being intrinsically
important, and that is worth confirming before investing in it.

**Important caveat on all of these numbers:** historical simulation runs WITHOUT the
availability gate, because `chance_of_playing_next_round` does not exist in the archive.
Live performance should exceed this, by an unknown margin.

*A "principled" improvement that measurably failed.* Bonus was changed to allocate exactly
six points per fixture — more faithful to the rules, since only six exist per match. It cost
**160 season points** (2086 -> 1926) and was reverted. Normalising within a match destroys
cross-match comparability, and the optimiser ranks players ACROSS fixtures: a strong player
was penalised for having strong team-mates. `bonus.allocate_match_bonus` is kept, documented
as tried and rejected.

### Phase D — Operations

**Item 15. Weekly pipeline & report**  — *DONE*
`reporting/gw_report.py`, `fpl report --gw N`. Markdown brief: XI, bench, captaincy with
`cost_vs_best` so overriding is priced, availability risks, differentials, and the standing
caveats about which components are weakest. Written as decision support, not an oracle —
a brief you cannot intelligently disagree with is not useful, since the model reads no
press conferences.

**Item 16. Monitoring & retraining**  — *DONE*
`backtest/monitor.py`, `fpl monitor`. Pre-committed thresholds rather than eyeballing, so
that reacting to a breach is a decision and not a rationalisation. Retraining runs on a fixed
6-gameweek cadence: weekly refits chase noise and make runs irreproducible, never refitting
discards a season of data.

The failure this guards against is not a crash but a model that keeps producing plausible
numbers while drifting — which happened three separate times during this build (collapsed
Kish prior, wrong xG noise scale, zero-filled DefCon column). Every one produced output that
looked entirely reasonable.

---

## 2b. Standing decisions (2026-08-09)

- **Pacing: sprint to end-to-end, then deepen.** Item 7 built properly; Items 8/9/10 and a
  working optimiser (12) built deliberately rough so a real GW1 squad recommendation exists
  by 2026-08-21. Depth is added afterwards. Anything knowingly crude is marked SPRINT in the
  code so it can be found and improved rather than forgotten.
- **Finishing skill: modelled, with heavy shrinkage.** A per-player conversion multiplier on
  xG, shrunk toward 1.0 with the same Gamma-Poisson machinery as the rate model. This nests
  the pure-xG case, so if finishing skill turns out not to persist the shrinkage handles it.
- **Set-piece duty: live API at inference, history for training.** Same pattern as the
  availability gate. `penalties_order` and the free-kick/corner orders exist in the live API
  (64/59/80 players ranked) but NOT in the archive, so they cannot be training features.
  Consequence to remember: the backtest cannot replicate this, so backtest results will
  understate live performance.

## 3. Suggested build order

Items 1-4 give a trustworthy dataset. Items 5, 6, 10 give a *working end-to-end forecast*
(minutes + match model + a crude attacking model is already competitive). Item 12 turns
forecasts into decisions. Then 7-9 and 13 sharpen the edges, and 14 tells you whether any of
it actually worked.

Deliberately deferred: neural sequence models, live in-play prediction, ownership/EO-based
rank optimisation beyond the simple risk objective, and a web UI.

---

## 4. Open questions to settle in Item 1

- Objective: maximise total points, or maximise probability of a top-X% overall rank?
  These give measurably different teams.
- Do you play chips at all, and do you want the model to recommend timing?
- How much manual override do you want — is this a decision-support tool or an autopilot?
- Available compute and how much time per week you want to spend on it.
