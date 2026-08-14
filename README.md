# FPL Expert

A points-forecasting and squad-optimisation system for Fantasy Premier League.

Two halves:

1. **Forecast** — a per-player, per-gameweek *probability distribution* over FPL points, built by
   modelling the underlying events (minutes, match scorelines, goal involvement, saves, bonus)
   rather than regressing directly on total points.
2. **Optimise** — a multi-gameweek mixed-integer program that turns those forecasts into
   decisions: squad, starting XI, captain, transfers and chip timing, subject to the real
   budget, formation, club-limit and transfer rules.

See [PLAN.md](PLAN.md) for the design and the numbered workstreams.

## Status

- **Item 1 — rules:** done. `config/scoring_rules.yaml` verified against the live game config.
- **Item 2 — data ingestion:** done, free sources only. 185,964 player-gameweek rows across
  7 seasons, plus 2,660 matches of bookmaker odds.
- **Item 2b — ownership:** built and tested. Uses prior-season rank to form a skilled
  cohort, so it works from GW1 rather than waiting for the top-10k table to stabilise.
  Runs once the season starts.
- **Item 3 — point-in-time snapshots:** done. Capture, enforcement, and scheduling.
- **Item 4 — feature engineering:** shrunk per-90 rates and fixture context done; the
  feature assembler is deferred to Item 5 so it follows real model requirements.
- **Item 5 — minutes model:** done. Walk-forward log loss 0.485 vs 0.583 for the
  transition baseline; calibration within ±2pp across all bins.
- **Item 6 — match model:** done. Dixon-Coles blended with odds-implied goal expectations;
  emits clean-sheet probabilities and goals-conceded distributions per team-fixture.
- **Item 7 — attacking returns:** done. Team expected goals allocated across players by
  shrunk xG/xA rates and expected minutes, with penalties handled separately.
- **Items 8-16 — done.** Secondary components, bonus, points assembler, squad MILP,
  validation, rank layer, season simulator, weekly report and monitoring.

**Full-season replay, strict walk-forward** (minutes model retrained on prior seasons only),
with MILP transfers, BB+TC chips, FPL's real selling-price rule, and — since
2026-08-12 — a **point-in-time planning horizon**, meaning each week's six-gameweek valuation
is assembled only from forecasts that could have been made that week:

| season | model | recent form | myopic | price-only | random |
|---|---|---|---|---|---|
| 2023-24 | **2352** | 1850 | 2202 | 1275 | 586 |
| 2024-25 | **2144** | 1874 | 2291 | 1336 | 605 |
| 2025-26 | **2071** | 1646 | 2041 | 1056 | 591 |

*(Ensemble means over 8 perturbed decision paths per season, which is why they differ slightly
from a single replay's 2349 / 2128 / 2060.)*

**Against recent form — the bar that means something — the model wins by +399 [+355, +443], in
every season, on all 24 of 24 paths.** Form buys whoever has been scoring over the last six
gameweeks, which is what an ordinary manager does, and it ranks players at 0.641 against
price's 0.381 (the full model: 0.692). The comparison is paired within each perturbation, and
it is the first result in this project to satisfy both adoption criteria — a paired interval
excluding zero *and* the sign holding in every season.

The **myopic** column is the uncomfortable one. It also beats form comfortably, and the gap
between it and the six-gameweek horizon is +11 ± 28 — indistinguishable. So the value
demonstrated here belongs to the *forecasts*, not to the multi-week optimisation built over
them.

> **The price-only column is not a usable baseline and is retained only for continuity.**
> Maximising a price-derived objective under a budget that binds at exactly £100m is close to
> degenerate: two solutions **0.8% apart in objective share 5 of 15 players and score 90 against
> 36** in the same gameweek. Which one the solver returns is decided by arbitrary detail. The
> figures above are one draw; running the same baseline through the ensemble gives
> 1275 / 1336 / 1056 instead.
>
> An earlier version of this file led with "the model does not beat the price-only baseline".
> That claim was built on a single draw from this degenerate set and is **withdrawn**. A
> non-degenerate baseline — recent form, say, whose objective is not collinear with the budget
> constraint — has to be built before the question can be asked at all.
>
> The simulator's own noise applies to every number here regardless: perturbing the forecast by
> 0.1% moves a season total with a standard deviation of 38 points, because one flipped transfer
> changes the squad path for the rest of the season. `DECISIONS.md` has the measurement.

**These replaced 2551 / 2535 / 2454, which were inflated by roughly 15%.** Until 2026-08-12 a
GW10 transfer was valued using the GW15 forecast — built from rates, form and a team model as
of GW15. No realised outcome was read, so every no-hindsight guard kept passing; the leak was
one level up, in the forecasts rather than the outcomes. Fixing it cost 392 points a season,
negative in all three. `DECISIONS.md` has the mechanism, which is more interesting than the
number: most of the loss came not from worse rankings but from the leaky horizon being 99.3%
*frozen* week to week, so the optimiser never had reason to trade.

**Simulated ranks are withdrawn.** They were computed against the inflated totals and have not
been re-derived. The ownership anchor is unaffected and remains the one exact check available:
every manager owns exactly fifteen players, so ownership counts pin what the average squad
scored with no modelling in between, and the simulated field reproduces it within 0.3% —
2014 / 2014, 1985 / 1991, 1958 / 1958 across the three seasons.

Overall points calibration is **0.995**, but that headline averages errors of opposite sign —
per season it is 0.985 / 0.997 / 1.004 — so treat the aggregate as less precise than it looks.
Note that calibration is a property of the FORECAST and is unaffected by any of the above; the
forecasts are well calibrated and the decisions built on them are what is in question.

Ablation, re-run on point-in-time forecasts — season points lost when each component is removed:

| variant | 2023-24 | 2024-25 | 2025-26 | mean |
|---|---|---|---|---|
| no_attack | −289 | −239 | −43 | **−190** |
| no_defcon | 0 | 0 | −143 | −48 |
| no_clean_sheet | −125 | +105 | −49 | −23 |
| no_saves | −133 | +45 | +55 | −11 |
| no_cards | −71 | −7 | +129 | +17 |
| no_appearance | −105 | +81 | +81 | +19 |
| no_bonus | −76 | +145 | +25 | +31 |

**Attack is the only component with a consistent sign**, and it is the only one clearly outside
the simulator's ~40-point-per-season noise floor. Everything else flips between seasons and
cannot be distinguished from zero — removing bonus, cards or appearance points *helps* on
average, which is a statement about noise rather than about those components. (`no_defcon` is
zero in two seasons because the statistic only exists in 2025-26.)

> An earlier version of this file claimed the opposite — that attack ranked *last* behind
> minutes and clean sheets. That ordering was measured under three double-gameweek bugs and on
> one deterministic replay per season. See `DECISIONS.md`; conclusions here are recorded with
> what refuted them.

> **Schedule the snapshot before 2026-08-21 17:30 UTC.** Pre-deadline state
> (injury news, prices, ownership) is overwritten in place and archived nowhere. A missed
> deadline means that gameweek can never be used for point-in-time evaluation.
> `powershell -ExecutionPolicy Bypass -File scripts\register_snapshot_task.ps1`

All data sources are free and keyless: the FPL API, the `vaastav` archive, and
football-data.co.uk. Understat and FBref are unreachable/blocked, but the FPL API already
supplies xG, xA and xGC per player per gameweek from 2022-23, which covers the need.

## Front end

```bash
fpl publish --gw 1      # build the serving bundle (~120KB) — run after each deadline
streamlit run app.py    # read it
```

The app **never runs the model**. `forecast_gameweek` loads the 185,000-row archive, rebuilds
minutes features over all of it and refits Dixon-Coles on every call, and `fpl squad` repeats
that once per horizon week — fine for a weekly command, hopeless per page view. `publish` does
it once and writes `data/serving/`: forecasts, the solved squad, a fixture grid and the brief,
with a manifest recording which gameweek it was built for and when.

That split is why the app deploys with `requirements-app.txt` — streamlit, pandas, pyarrow —
and needs neither lightgbm, nor pulp, nor the archive. The bundle is versioned, so a hosted
deployment needs no build step.

The bundle carries **two views of the same gameweek** — the standard forecast and one with
each club's expected minutes constrained to eleven players — and the app offers a switch. The
constraint improves the forecast (player MAE 15.02 to 14.84, team totals 970 to 990 against a
hard ceiling of 990) but its effect on season points flips sign between seasons (-90 / +77 /
+51), so neither view has earned the right to be the only one. On GW1 they share 7 of 15
players.

The bundle describes the **game, not your team**: no entry id, no held squad, no transfer plan,
which is what makes it safe to commit and publish. `fpl myteam --brief` covers the rest and
stays local.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Usage

```bash
fpl status     # where the season is, and what rules are loaded
fpl update     # pull the current season's core feeds from the FPL API
fpl history    # download past seasons + print the column-coverage report
fpl odds       # download bookmaker odds from football-data.co.uk
fpl snapshot   # capture pre-deadline state; --if-due makes it safe to run hourly
fpl snapshots  # list captures, and flag gameweeks with no pre-deadline snapshot
fpl ownership --gw N   # effective ownership among a skilled cohort (completed GWs only)
fpl minutes    # train + walk-forward score the minutes model, then save it
fpl match      # fit the match model and forecast clean sheets for upcoming fixtures
fpl squad --gw 1   # full pipeline: forecast every player, then solve for the best squad
fpl report --gw 1  # weekly decision brief (XI, captain, risks, differentials, caveats)
fpl simulate -s 2023-24 -s 2024-25 -s 2025-26   # replay seasons + component ablation
fpl repeat -s 2025-26 --draws 150   # replay many times over resampled outcomes, for error bars
fpl prices --season 2025-26   # price-change model: expected risers and fallers
fpl myteam --entry N          # pull your real squad and recommend transfers
fpl myteam --entry N --brief out.md   # ...as a full brief: transfers, chips, price moves
fpl validate       # component accuracy, calibration, top-N precision
fpl monitor        # standing health checks against pre-committed thresholds
```

`history` and `odds` accept `-s/--season` (repeatable) to refresh a single season; both are
partitioned per season, so a targeted refresh leaves the others untouched.

### Which commands recompute, and which read what is on disk

Worth knowing, because a model change reaches the two groups differently.

**Recompute from source every run** — `squad`, `report`, `match`, `myteam`. These rebuild
rates, minutes and match forecasts in-process, so a code change takes effect immediately with
no regeneration step. The one artefact they load is the saved minutes model
(`data/processed/models/minutes.txt`); re-run `fpl minutes` after any change to that model or
any refresh of history.

**Read stored tables** — `simulate` writes `sim_forecasts` and `sim_fixtures`; `validate`,
`monitor` and `repeat` read them. **Any change to the forecasting model needs `fpl simulate`
re-run before these mean anything.**

`fpl match` writes `processed/match_forecasts` as an output for inspection. Nothing reads it,
so a stale copy is misleading rather than harmful.

> **Partitions are refreshed one season at a time**, so a table can quietly hold two
> generations of the model at once — this has already caused one wrong conclusion. Check with
> `storage.partition_report("processed", "sim_forecasts")`, which warns when write times
> disagree.

### Before the GW1 deadline

The live commands are current in code but their inputs are only as fresh as the last pull.
Run `fpl update` (prices, availability, fixtures move daily) and `fpl snapshot` close to the
deadline.

## Data layers

| layer | contents | reproducible? |
|---|---|---|
| `data/raw/` | immutable gzipped API dumps, partitioned by pull timestamp | **no — this is the point-in-time record** |
| `data/interim/` | parsed, typed parquet per source | yes |
| `data/processed/` | model-ready feature tables | yes |

### Point-in-time discipline

Every feature used to predict a gameweek must be read through `PointInTime`:

```python
from fpl_expert.data.snapshot import PointInTime

pit = PointInTime.for_gameweek(7)   # latest snapshot taken BEFORE GW7's deadline
players = pit.players()             # prices, news, availability as known then
odds = pit.odds()                   # opening/deadline odds only — never closing
```

It raises `MissingSnapshotError` rather than quietly falling back to post-deadline state,
and it has no accessor for closing odds or results. Closing odds live in
`external/odds_eval` and are for Item 11's evaluation only — they postdate the deadline, so
they are the single most damaging thing that could leak into a feature.

`assert_no_post_deadline_columns(df)` guards assembled frames against the realistic
accident: a join that drags `p_home_close` along with the team names.
