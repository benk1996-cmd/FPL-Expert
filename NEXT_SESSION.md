# Next session — where to pick up

Updated 2026-08-24. **The season has started.** GW1 is played (one fixture still live at the
time of writing); GW2's deadline is **2026-08-28 17:30 UTC**.

Read `DECISIONS.md` first for *why* things are the way they are. This file is only *what to do
next*.

---

## Ground rules that have earned their place

These came out of things that went wrong. Breaking them has cost real work.

1. **Never conclude from a single deterministic replay — but check what the instrument is
   actually measuring.** Free Hit is the cautionary tale, and it has now been decided three
   times. Excluded on three consistent replays; readmitted at "+17 over 450 paired draws";
   then found that those draws came from the model's own pmf and estimated a tautology; then
   readmitted again at +7.2 on realised outcomes; and finally excluded when those realised
   outcomes were re-derived on a horizon that was not reading the future. Each instrument was
   better than the last and each overturned it.
2. **But a pooled confidence interval is not a per-season one.** Within a season every draw
   shares one decision path; only outcomes vary. A narrow pooled CI answers "how sure are we
   about these three seasons", not "will this hold next season" — for which n = 3. Check that
   the sign holds in every season before adopting anything.
3. **A bias that is real in a regression is not necessarily exploitable in a constrained
   optimiser.** FIVE principled fixes for measured biases have now made things worse or
   nothing: bonus match-allocation, ownership calibration, rank-aware captaincy, defcon
   overdispersion, and the transfer hit bar — the last resting on the most stable measurement
   in the project (margin overstated 2.3x, slope 0.436 in every season) and still failing.
4. **Quote points differences; treat ranks as indicative.** A 25-point change in our edge moved
   a median rank by an order of magnitude.
5. **Decisions never read realised outcomes.** Squad, transfers, captain, chips and the field's
   composition depend on forecasts only. This is what makes solve-once/rescore-many valid and
   made repeated simulation 360,000x cheaper.
6. **Fix causes, not symptoms.** The under-prediction was measured weeks before it was fixed,
   and the decision NOT to paper over it in `distribution.py` is what left the real bugs
   findable in `attack.py` and `pipeline.py`.
7. **Distrust defensive idioms that degrade silently.** `[c for c in group_cols if c in
   df.columns]` read as robustness and behaved as a no-op: it disabled position-grouped
   shrinkage entirely and gave goalkeepers outfield attacking rates for weeks. If a fallback
   changes the model, it must warn.
8. **Check `partition_report` before trusting a measurement that spans seasons.** Partitions
   are refreshed one at a time and a stale one is invisible in the concatenated table.
9. **A forecast has two dates: the week it describes and the week it was made in.** Conflating
   them leaks the future without ever touching a realised outcome, so every no-hindsight guard
   keeps passing. This cost 392 points a season and went unnoticed for the project's whole
   life. When adding any forward-looking quantity, ask which date each input carries.
10. **Suspect a result that is TOO consistent.** The horizon's justification was persuasive
   precisely because it was positive in all four seasons — and that consistency came from
   every season being assembled the same wrong way. Uniformity across seasons is evidence of a
   shared mechanism, which may be a shared bug.

---

## State of play

475 tests, clean lint. Three seasons, strict walk-forward, MILP transfers, BB+TC chips,
real selling prices, and — since 2026-08-12 — a **point-in-time planning horizon**. Figures
below are ENSEMBLE means over 8-10 perturbed decision paths per season, not single replays:

    season    model   recent form   myopic   price-only   ownership anchor
    2023-24    2352       1850       2202       1275         2014 / 2014
    2024-25    2144       1874       2291       1336         1985 / 1991
    2025-26    2071       1646       2041       1056         1958 / 1958

**These are not the old 2551 / 2535 / 2454.** The horizon lookahead was fixed and it cost 392
points a season, negative in all three. Read the DECISIONS entry before comparing anything to a
number written before 2026-08-12.

**The model beats recent form by +399 [+355, +443], every season, 24 of 24 paths.** That is the
project's headline result and the only one to satisfy both adoption criteria.

**Ranks are withdrawn** — they were computed against the inflated totals.

**The horizon is an unproven default**, +11 +/- 28 against myopic, and every parameter of it
has now been swept without finding an improvement.

**price-only is retired as a benchmark** — its optimum is degenerate. It sits 568 BELOW form,
so quoting it as a hard bar understated this system by ~550 points a season for months.

Everything in `PLAN.md` is built. What follows is what is left.

---

## 2026-08-24: the live path got its first real exercise, and it found things

Nothing here changes a backtest number. All of it is the live path, which until this week had
never been run against a started season.

**Two bugs, both invisible until now.**

* **Snapshot odds capture was silently discarding everything.** `normalise` returns a `date`
  column of dtype `datetime64[us]`, `to_dict("records")` yields `pd.Timestamp` objects, and
  `write_raw`'s `json.dumps` could not serialise them. Worse, it streamed into the open gzip
  handle, so the failure left a TRUNCATED dump on disk — 66 bytes ending mid-field — while the
  `except Exception` around the odds capture reported a clean snapshot holding zero rows. Not a
  regression: the feed returned `[]` on 9 and 16 August, so there was nothing to serialise and
  the bug could not fire until bookmakers began pricing. Fixed both halves — a narrow
  `_json_default` (not `default=str`, which would let an unexpected type into the evidence
  trail as a plausible string), and serialise-before-open so a failure leaves no file at all.
  Verified: `odds_rows 10`. The corrupt partition was deleted.
* **`data/serving/` is committed, and a `myteam` brief was sitting in it.** `--brief
  data/serving/myteam_gw2.md` put a real squad one `git add` from being deployed, against the
  boundary `test_the_bundle_carries_no_personal_data` exists to enforce. Now gitignored
  (`data/serving/myteam_*.md`), along with `.fpl_entry.json`. **Anything entry-specific must
  stay out of that directory** — it is the one data path that is versioned on purpose.

**One stale docstring corrected.** `recommend_transfers` still carried a prominent call to
re-derive the hit threshold. That work was done the very next day and the answer was no; the
docstring now records the rejection, the numbers that killed it, and a warning not to re-open
it on the strength of the 2.3x margin measurement alone. The superseded DECISIONS entry at
"needs re-deriving" was left alone deliberately — it is a dated entry in an append-only log
and the sweep entry below it supersedes it. **The stale-guidance risk is in docstrings, which
a reader meets with no date attached.**

**The front end grew two features and one hard limit.**

* `advice.py` — `analyse_entry()` returns an `EntryAdvice`, and both `fpl myteam` and the app
  call it. Extracted rather than duplicated because `report` and `myteam` already drifted once.
* A **My Team tab**, and a **gameweek navigator** (◀ ▶) across the published horizon. The
  navigator was nearly free: `_horizon_frame` already computed each week's forecast — the
  horizon valuation is the decayed sum of them — and threw them away once summed. They are now
  served as `forecasts.parquet` and the manifest carries `gameweeks`. Bundle 117KB -> 460KB,
  publish no slower. **The squad shown is always the published week's decision**; stepping
  forward answers "how does the squad I picked look next week", not "what would I pick", and
  the UI says so, because re-solving needs the model.
* **My Team cannot run on Streamlit Cloud, by design.** It needs `data/interim`,
  `data/processed`, `data/external` and `data/raw/snapshot` — all gitignored, which is exactly
  what lets the bundle deploy without the archive. The first attempt guarded `ImportError`,
  which never fires there: the packages install fine and it is the DATA that is absent, so a
  `FileNotFoundError` traceback reached the page. The tab now checks its inputs up front and
  explains, rather than offering a button that spins for twenty seconds and fails.

## 2026-08-25: the archive stopped at last May, and nothing was going to fix that

**The models were training on prior seasons only, a week into the new one.** `fpl history`
downloads COMPLETED seasons from the community archive; `fpl update` pulls the bootstrap,
which is current *state* and carries no per-gameweek results. So no `season=2026-27` partition
existed and none would have: every rate was built from players at their PREVIOUS clubs, and
the gap widened by one gameweek a week. Calvert-Lewin's xG rate was Everton's Calvert-Lewin.

`fpl results` closes it — 610 rows for GW1, one per player-fixture, written into
`interim/history` alongside the archive partitions.

**Why `element-summary` and not `event/{gw}/live`.** The live endpoint is one call per
gameweek instead of one per player, but it collapses a double gameweek into a single entry and
exposes per-fixture detail only for scoring identifiers — no per-fixture expected goals.
`element-summary/{id}/history` returns one row per FIXTURE with the full stat line, plus
`value` (price at the time) and `kickoff_time` for rest days. It matches the archive's shape,
so the partitions concatenate without harmonisation. ~610 polite requests, about ten minutes,
once a week.

**Only `finished AND data_checked` gameweeks are taken.** `finished` flips first, while bonus
is still provisional; a half-settled gameweek in the archive is indistinguishable from a real
one and would tell every rate that a striker played zero minutes that week.

### Two traps this sprang, both now guarded

1. **Lookahead, newly possible for the first time.** `as_of` is derived from the history
   MAXIMUM, not from the gameweek being forecast. That was safe only because the archive
   stopped last May. With the current season on disk, a result for GW n would sit at or before
   `as_of` when forecasting GW n and feed its own forecast — ground rule 9, exactly.
   `forecast_gameweek` now drops this season's rows from `gw` onward before computing rates,
   so the maximum is correct by construction whatever is on disk. Tested.
2. **The API sends expected goals as a STRING.** Nine columns differ in dtype from the archive
   — including `expected_goals` and `expected_assists`, the two most important rate inputs.
   Concatenation produced an object column that survived the write and failed on the first
   multiplication, deep in the rate calculation, with a traceback pointing at pandas. Coerced
   at ingestion, plus `schema_mismatches()` and a test comparing the written partition against
   the archive's dtypes — the check that would have caught it at the cause.

**Run `fpl results` after each gameweek is checked.** It is now the first step of the weekly
loop, before `fpl publish`.

## Everything previously listed here is now closed

Worked through 2026-08-11. Outcomes, so nobody re-opens them blind:

| item | outcome |
|---|---|
| Match model / residual under-prediction | **Cause found and fixed** — it was not the match model. Goalkeepers were being given outfield attacking rates (79 phantom goals). Overall calibration is now 1.0000. |
| Understand `lambda_rank` | **Closed.** Re-measured per season on corrected forecasts: worth nothing at 0.1 or 0.2, signs flip between seasons. Stays at 0. |
| Price model into the brief | **Done.** `_price_moves` feeds `price_section`; risers league-wide, fallers only among players you hold. |
| Rival budget / club limits | **Built, measured, off by default.** Repair works (47% illegal -> 0.7%) but costs the ownership anchor 1.4%. `enforce_legality=True` to switch on. |
| Goals/assists independence | **Measured (1.74x lift), declined.** An order of magnitude below the bonus coupling that was worth modelling, and would need a joint distribution where the code convolves independent ones. |

## Season-boundary minutes: investigated, and it was NOT a bug

Raised because `fpl squad` would not pick Haaland for 2026-27 GW1: the minutes model gave him
p_long 0.609 against Fernandes on 0.845, despite 60+ rates of 87% and 89% in the season just
finished. The EWM features have a five-gameweek half-life, so at an opener they are made
entirely of last May — which looked like end-of-season noise leaking across the boundary.

**Two tests said otherwise, and the second is decisive.**

Resetting the EWM at the season boundary, so an opener cannot see last May at all, made
openers WORSE in every test season:

    gw1 log loss     ewm crosses seasons   ewm reset per season
    mean                   0.7883                0.8018

And the empirical base rate settles it. Across five season boundaries, players with a strong
prior season (>=0.8) but weak closing form (EWM <=0.7):

    profile                            n     started 60+ in the next opener
    strong season, weak closing form   34              0.529
    strong season, strong closing form 281             0.772

`started_ewm` correlates with starting an opener at 0.535, ahead of the prior-season rate at
0.510. **Late-season minutes carry real information about the next campaign.** A p_long of
0.609 for a player rested twice in his final six is not an error — it is slightly generous
against a base rate of 0.529.

Caveat: n=34 for the Haaland-shaped group, so the point estimate is soft. The direction agrees
with the log-loss result, which does not depend on that subgroup.

**What was kept.** Season-level features (`season_games`, `season_started_rate`,
`prev_season_started_rate`, `prev_season_games`) went in anyway, because they are a small
consistent win in their own right — walk-forward log loss 0.4847 -> 0.4828 overall and
0.7902 -> 0.7876 on openers, better in all four test seasons. They moved Haaland 0.609 ->
0.617, which is the right size of correction: nearly none.

**Consequence for the Haaland question.** `fpl squad` omitting him is not a minutes bug. At
£15.5m with genuine rotation risk his horizon points are 6th in the game but 1.11 per £m, and
no budget justifies that. Whether the model is RIGHT about 2026-27 is unknowable until the
season starts — but it is not making an error the archive can detect.

## RESOLVED — the model DOES clear its bar. The alarm that stood here is withdrawn.

This section previously read "the model does not beat picking on price". That was wrong twice
over: the price baseline is degenerate (two solutions 0.8% apart in objective share 5 of 15
players and score 90 against 36), and it was being run on the leaky flat horizon while the
model ran point-in-time. Both are recorded in DECISIONS.

Against **recent form**, which is not collinear with the budget and ranks players at 0.641
against price's 0.381:

    paired vs form     mean_diff     se     95% CI         wins   adoptable
    horizon               +399.0   22.5   [+355, +443]     1.00      True
    myopic                +388.0   14.8   [+359, +417]     1.00      True
    price_only            -567.8   13.0   [-593, -542]     0.00      True

**+399, every season, 24 of 24 paths** — the first result in this project to satisfy both
adoption criteria. The forecasting layer earns its keep.

## What is still open: the DECISION layer earns nothing

Both `horizon` and `myopic` beat form by essentially the same margin, and the gap between them
is +11 +/- 28. The forecasts carry the whole result; the multi-week optimisation built over
them contributes nothing measurable.

Those leads have now been **swept on the ensemble and all of them failed** (2026-08-13):
9 variants x 3 seasons x 10 paths, paired. `hit_bar` 6/8/9/10/12, `decay` 0.70/0.92 and
`captaincy_weight` 0 all flip sign between seasons; four have pooled intervals excluding zero
and none survives ground rule 2. Every incumbent stays. See DECISIONS for the table.

Two things that closes:

* **The hit-bar correction is rejected**, not merely unresolved. It rested on the most stable
  finding available (margin overstated ~2.3x, slope 0.436 in every season) and still measured
  -38 / -42 / +45. That is the fifth principled bias fix to produce nothing.
* **The captaincy term is unproven too.** `no_captaincy_term` is -59 / -58 / +59. The earlier
  inference that the armband was carrying the horizon came from decomposing a horizon-minus-
  myopic difference; the direct toggle disagrees.

What is genuinely left on the decision layer:

1. **Decide whether to keep the horizon at all.** It is +11 +/- 28 against myopic, and no
   parameter of it can be improved. Keeping it is defensible (unproven is not disproven, and
   myopic never buys a premium — it owns Haaland 0 of 38); deleting it would simplify the
   backtest sevenfold. This is now a judgement call, not a measurement.
2. **Nothing else.** Every knob has been swept.

Also note the price gap itself is only about twice the noise floor in the two losing seasons,
and the model wins on the three-season mean by +89. The problem is real but it is not as stark
as a per-season sign count makes it look.

### DO NONE OF THOSE UNTIL THE INSTRUMENT IS FIXED — build the ensemble simulator

**Perturbing the forecast by 0.1% moves a season total with a standard deviation of 38 points
and a range of 90.** One flipped transfer decision changes the squad, and the squad changes
every decision after it. Transfer counts barely move (64, 64, 64, 64, 64, 65) while points
swing, so this is path dependence, not policy.

Almost every open question above is smaller than that floor:

    effect                                   per season          verdict
    horizon lookahead                 -200 / -518 / -459    real
    attack ablation                   -289 / -239 /  -43    real
    horizon vs myopic                 +111 / -174 /  +15    inside the noise
    valuation smoothing (hl=1)         -88 / +182 /  +78    inside the noise
    corrected hit bar (LOSO)           -54 /  +11 / +128    inside the noise
    Free Hit vs BB+TC                  -11 /  -16 /  +11    well inside the noise

**Build this:** run each variant over k perturbed decision paths per season and compare
distributions, not point estimates. At ~80s a path, k=10 across three seasons is ~40 minutes
per variant and cuts the standard error by about three. Perturb the forecast (or the solver's
tie-breaking); leave outcomes alone.

This is ground rule 1 applied to the DECISION path rather than to outcomes. `repeat_sim`
resamples outcomes while holding decisions fixed — the opposite axis, which is exactly why it
never exposed this.

Then, and only then, items 1-3 become answerable.

### A recommendation that was made and then measured away: adding 2022-23

Proposed as a route to n=4, then checked. **It does not work.** xG exists only FROM 2022-23, so
during that season the attacking rates are built from at most a partial season of xG and the
shrinkage pulls them to zero:

    xg_per90 among players with real exposure    p50      p90
    2022-23 GW10                              0.0000   0.0000
    2022-23 GW20                              0.0353   0.0765
    2022-23 GW30                              0.0691   0.2132
    2023-24 GW2                               0.0832   0.2874

At GW10 more than 90% of players with exposure have an attacking rate of exactly zero, and GW1
cannot be forecast at all. Since attack is the dominant component, 2022-23 would not be a
fourth sample of this system but a sample of a crippled one, guaranteed to underperform for
reasons unrelated to the decision layer. Any sign flip it produced would be an artefact.

n=3 was the wrong diagnosis anyway. The binding constraint is that each season is one draw from
a chaotic path, which is what the ensemble fixes.

## CLOSED — the horizon lookahead (was item 1)

Fixed 2026-08-12. `forecast_horizon` separates the gameweek being forecast from the gameweek
the forecast is made in. Worth **-392 season points, negative in all three seasons**; see
DECISIONS for the mechanism, which is more interesting than the number. It also closed the
double-gameweek minutes leak via `decision_state`, and that turned out to be a wash.

The reviewer's 70-500 point range was in fact roughly right, which its own caveats had
disclaimed. The sub-result "removing the chip planner's lookahead changes nothing (+3 / 0 / +9)"
did NOT hold — it degraded the planner while leaving the transfer horizon leaky.

## CLOSED — conclusions drawn through `repeat_sim` (was item 2)

Re-derived on `bootstrap_realised` with point-in-time forecasts.

- **Free Hit is out of `DEFAULT_CHIPS`**, now BB+TC. It went +7.2 [+5.6, +8.9] to
  **-5.5 [-6.9, -4.0]**, win rate 0.52 to 0.41. Playing chips is settled (`no chips` loses in
  all three at a 0.00 win rate); which chips is not resolvable at n=3, so the default retreats
  to the simpler set.
- **Ablation re-run lookahead-aware.** Attack is the only component with a consistent sign
  (-190 mean). Everything else flips; removing bonus/cards/appearance helps on average.
- `lambda_rank` and `flexibility_weight` were both measured at ~zero and are unlikely to change
  a decision; not re-derived.

### CLOSED — the market blend fires (was item 3)

Verified 2026-08-12: `attach_opening_odds` is wired into the forecast path and matches **90%**
of fixtures, so `market_weight=0.8` is real and README's "blended with odds-implied goal
expectations" is accurate. Every number dated 2026-08-12 or later describes a blended model.
Numbers from before the join was added still describe Dixon-Coles alone.

### 3. The ownership identity is not exact in blank gameweeks

Players without a fixture have no row, so `sum(selected)/15` understates the manager count and
inflates the anchor by 1.5-2.1% per season. The field is correspondingly ~30-40 points too
strong, which is conservative for our rank but means the calibration is against the wrong
target. Note the sampler itself is BETTER than advertised: 0.04% against the achievable
frame-restricted anchor, not 0.3%.

### Smaller, confirmed, unfixed

- **Auto-substitutions** are not modelled (~9 blanks, ~27 points a season, understates us).
- **Goals-conceded deduction** applies only to 60+ appearances in both paths; FPL applies it
  per 2 conceded while on the pitch.
- **Chip valuation uses market prices** while execution uses selling prices, so the planner
  over-values Free Hit and Wildcard relative to what it can afford.
- **`partition_report` warns on every read**, because partitions are always written
  sequentially. A warning that always fires is one nobody reads.
- ~~Minutes features leak within a double gameweek~~ — **fixed** by `decision_state`, which
  takes the earlier kickoff's row. All 1,766 rows. Measured as a wash (-58 / +34 / -8).

## What is actually left

Beyond the price-baseline problem at the top of this file:

0. **Weekly, now that the season is live:** `fpl snapshot` before each deadline (the scheduled
   task mostly handles it), then `fpl publish --gw N` after it, then `fpl myteam --entry
   3468852`. The published bundle is what the app serves, so a stale bundle is a stale page —
   it showed GW1 for three days after GW1 kicked off because nothing had republished.
1. **Simulated ranks need re-deriving.** Withdrawn, not replaced — they were computed against
   totals ~400 points too high. The field is anchored independently on the ownership identity,
   so only our side of the comparison moved, but it moved a long way.
2. **Saturation should be re-checked and may have resolved itself.** The old entry read "two
   seasons still saturate" after four sources of flattery had been removed. A fifth and much
   larger one has now gone, and at 2060-2349 against an anchor of ~1958-2014 the margin is far
   more plausible than it was. This may simply be closed.
3. **The distribution's joint structure** — goals/assists (1.74x) and clean-sheet/returns
   (1.37x) couplings, both measured and both declined as not worth the structure. Revisit only
   if the tail proves to matter after some other change.
4. **`fpl prices` is fitted on the whole archive** when used live. That is correct for
   predicting forward, but means its quoted log loss is the walk-forward figure from
   `fpl prices --season`, not a property of the live model.

## Explicitly NOT worth doing

- **Rebuilding bonus from BPS components.** Measured twice, before and after the attack fixes:
  bonus cannot be distinguished from zero in decision terms (-2.8, CI [-10.1, +4.6]). Neither
  can cards. A better bonus model would not move anything.
- **Negative-binomial defensive contributions.** Genuinely overdispersed, but modelling it
  improves defenders and worsens midfielders and forwards. Measured, rejected, recorded.
- **A team minutes budget.** Built, tested and rejected. It enforces a constraint that is
  arithmetically true (eleven players, ninety minutes; live frames run 399-1236 against a
  ceiling of 990) and improves minutes accuracy, but is worse on POINTS in all three seasons:
  MAE +0.003/+0.009/+0.001 and top-20 realised points -0.025/-0.050/-0.088. The rescale is
  proportional, so deep squads are scaled down hardest and that is where the scorers are.
  `balance_team_minutes` is kept, off and unused. See DECISIONS.
- **An opponent term on defensive contribution.** Looks like an obvious gap — clean sheets are
  fixture-dependent, defcon is not — but the effect is entirely BETWEEN teams and already
  captured by the per-player rate. Within a player, defcon/90 is 6.93 on a clean sheet and
  6.92 when conceding. Measured on 7,815 player-matches; see DECISIONS.
- **Tuning rank-aware captaincy.** Built, measured, negative. The mechanism is correct and
  tested; the situation does not call for it.
- **`flexibility_weight`.** At 0.1 it changes literally nothing (paired difference exactly
  zero); at 0.3 it is unresolved and loses 66% of draws.
- **Resetting the minutes EWM at the season boundary.** Measured worse on openers in all four
  test seasons; late-season minutes genuinely predict the next campaign.
- **Enforcing legality on rival squads.** Built and correct, but it trades the exact ownership
  calibration the field is anchored on for squad legality. Off by default; see `field_sim`.
- **Smoothing the horizon valuation across decision weeks.** Built as
  `simulate_season(smoothing=halflife)`, default 0.0. It targets the right mechanism — hits
  fall from 27 to 13 — but recovers only ~15% of the lookahead loss and the sign flips
  (-88 / +182 / +78). The control confirms the instrument: smoothing the LEAKY horizon
  degrades it monotonically, as it must, since that one is already frozen at 0.993. Kept in
  the code so the measurement can be repeated, not because it is close to worth enabling.

---

## Permanent limitation, worth restating

**Historical availability data does not exist.** `chance_of_playing_next_round`, `status` and
`news` are overwritten in place and archived by nobody, so every backtest runs without the
injury gate the live pipeline applies. This cannot be fixed retrospectively. It is why
`fpl snapshot` exists and is scheduled.

---

## SHELVED — a sentiment/NLP reviewer over the model's predictions (2026-08-16)

The idea: ingest qualitative team news and have it review or adjust the forecasts. It targets
the right gap — every remaining blind spot is a MINUTES problem (no depth-chart reasoning, so
"Saliba injured therefore Mosquera starts" is unavailable; no rotation awareness; no jobshare
detection), and minutes is the largest single driver of variance.

Shelved for three reasons, in order of weight:

1. **There is no corpus, and building one is gated on hardware.** `news`, `status` and
   `chance_of_playing_next_round` are overwritten in place, which is why `fpl snapshot` exists.
   Three snapshots exist. A season would give ~2,200 labelled examples (~60 flagged players x
   38 gameweeks) with free labels — the following week's minutes. But the scheduled task has
   `WakeToRun=False` and `DisallowStartIfOnBatteries=True`, and three of the five deadline
   checkpoints fall between 01:00 and 01:30 local, so the captures closest to each deadline
   will often be missed. Changing those settings was declined; the machine is not to be
   reconfigured. The task still runs opportunistically and costs nothing, so the corpus will
   accumulate partially rather than not at all.

   **Softened 2026-08-24.** The ladder self-heals — see the unblocked section above. A missed
   48h or 24h rung is captured at the next hourly run the machine is awake for, just further
   out; only the 2h and 0.5h rungs are lost for good. The corpus therefore accumulates in full
   at coarse resolution and is thin only at the freshest end. That is still the end where team
   news lands, so the argument stands, but the shortfall is narrower than "often missed" reads.

2. **It could not be validated.** No historical news exists, so there is nothing to backtest
   against. Six principled, well-reasoned improvements have now measured to nothing or worse —
   including one enforcing a constraint that was arithmetically TRUE — and an unvalidatable NLP
   layer would be the seventh with more moving parts than all of them combined.

3. **"Reviewer over the predictions" is the wrong shape anyway.** If the signal is real it
   belongs INSIDE the minutes model, not layered on the output: expected minutes propagate to
   clean sheets (conditional on 60+), the defcon threshold, goal allocation and team-goal
   conservation. Adjusting final expected points would break that consistency. A post-hoc
   adjustment layer also gets judged on whether its output looks plausible, which is exactly
   the self-grading trap `repeat_sim` fell into.

**What would unblock it**, cheapest first:

* Wire odds into the LIVE path. Bookmakers already ingest this qualitative information
  professionally and price it, they cover all 20 clubs, and the live path currently attaches no
  odds at all (`market_weight=0.8` is inert there). This is the cheap version of the same signal.
* Parse `news` with rules rather than learning. "Expected back 21 Aug" is a hard date the model
  ignores entirely — it reads only the numeric flag. A date parser is checkable within weeks,
  because the label arrives seven gameweeks-days later, and needs no corpus.
* A season of snapshots, for anything learned. Within-season walk-forward (train GW1-20, test
  GW21-38) by January; held-out-season validation, this project's standard, not before 2027-28.

**The larger prize, if snapshots do accumulate**, is not the reviewer. It is finally measuring
what the availability gate is WORTH. The backtest runs with no gate at all, so "every number
here is conservative by an unmeasurable amount" is literally true. One season of snapshots turns
that permanent caveat into a number, and needs no new modelling.

## UNBLOCKED — the season has started (this list was the blocker, and it is gone)

All three cleared on 2026-08-24. Recorded here rather than deleted, because two of them were
blocked for a *different reason* than this file claimed.

- **`fpl ownership --gw 1` is runnable now.** The gate is the DEADLINE, not completion:
  `entry/{id}/event/1/picks/` stops 404-ing once it passes, and picks froze at that moment, so
  a mid-flight gameweek is irrelevant. League 314 is populated. Two things to know before
  running it. First the cost: `top_n_managers: 10000` at `request_delay_seconds: 0.5` is ~10,000
  `entry/history` calls plus 1,000 picks calls, so 1.5-2 hours of polite requests. Second, and
  more important, **nothing consumes the output** — `ingest_ownership` is called only from the
  CLI, writes `interim/ownership`, and no other module reads it; `min_gw_for_top10k` in
  `config.py` is defined and never referenced. So this buys a MEASUREMENT, not a pipeline
  improvement. That measurement is worth having: it is an independent check on the simulated
  field anchor, which is known to be inflated 1.5-2.1% by the blank-gameweek identity error
  (see "the ownership identity is not exact in blank gameweeks" above). Not yet run.
- **`fpl myteam` works. The entry id is 3468852** ("beanchodeFC"). Everything else comes from
  public endpoints; there are no FPL credentials and none are wanted.
- **Snapshots: nothing to fix, and the shelved note over-stated the problem.** `snapshot_due`
  takes `target = min(passed)` and only skips when a held snapshot is already INSIDE that
  checkpoint, so a capture missed at 01:30 is taken at the next hourly run the machine is awake
  for — at 40h out rather than 48h. **The ladder self-heals.** Only the 2h and 0.5h captures are
  genuinely unrecoverable, since after the deadline `snapshot_due` returns early. For GW2 the 6h
  checkpoint falls at 19:30 local on a Friday, which is waking hours. The scheduled task is
  healthy (hourly, `LastTaskResult: 0`). No reconfiguration needed; the earlier "the captures
  closest to each deadline will often be missed" is true only of the last two rungs.

### `fpl myteam` has a prerequisite that is not a flag

It reads the target gameweek through the strict `PointInTime.for_gameweek`, not `for_planning`,
so it needs a pre-deadline snapshot for the gameweek being planned or it raises
`MissingSnapshotError`. **Run `fpl snapshot` before the deadline; do not loosen the accessor.**
Arguably it should use `for_planning` — planning the upcoming week from today's state is not a
lookahead violation, because a later snapshot cannot exist yet — but taking the capture
sidesteps the question and does not weaken a guard that exists to prevent one specific bug.

**Set-piece order is NOT blocked and was never blocked** — this section listed it as blocked,
wrongly, until 2026-08-14. The FPL API publishes `penalties_order`, `direct_freekicks_order` and
`corners_and_indirect_freekicks_order`; ingestion has always captured them; and
`pipeline.py` has always fed `penalty_share` into the allocator. 55 players carry a live
penalty share right now. What is blocked is only the BACKTEST, because the archive has no
set-piece column — which is why `penalty_share` is 0 on every historical row. That is an
asymmetry in our favour and is listed under the permanent limitation above, not here.

## Set-piece order: three findings, 2026-08-14

Verified against a published community takers list. **14 of 20 clubs agree exactly on the
primary taker**, including all four "nailed on" (Thiago, Palmer, Haaland, B.Fernandes) and
Fulham's Robinson, a £4.5m DEFENDER on penalties.

### 1. The model cannot represent a jobshare — the one real defect

    PENALTY_ORDER_SHARE = {1: 0.85, 2: 0.11, 3: 0.03}

Arsenal (Saka and Gyokeres, "whoever fancies it") and Sunderland (Le Fee and Diarra,
alternating) are true 50/50s. The table maps ordinal position to a fixed share, so two
co-takers listed 1 and 2 can never come out even: Saka is allocated 0.85 where ~0.5 is right,
a ~70% over-allocation of penalty value to a premium asset.

`penalties_text` is ingested and unused. It carries FPL's own free-text note, which is where a
jobshare is described — a route to detecting them without hardcoding anyone's list.

### 2. A designated taker's penalty xG is counted TWICE, and only in the live path

`xg_per90` is built from the archive's `expected_goals`, which includes penalty xG at 0.79 a
spot-kick. So a historical taker's rate already embeds his penalties. `allocate_team_goals`
then computes his open-play share FROM that inflated rate and adds `expected_penalty_goals` on
top. The team total is conserved by the rescale, but the taker is over-weighted against his
own team-mates. The explicit term is **9.5% of a taker's expected goals** (median, GW1).

**The backtest cannot detect this.** `penalty_share` is 0 on every historical row, so the
double count never occurs there — the 1.0000 calibration and the +399 result are both silent
on it. It is live-only and has never been measured.

Fixing it properly needs penalty xG netted out of the rate, and the archive records
`penalties_missed` but not penalties taken or scored, so it cannot be done exactly from what
is stored. An approximation is possible; validating it is not.

### 3. Free kicks and corners are ingested and unused — and should probably stay that way

`direct_freekicks_order` (59 players) and `corners_and_indirect_freekicks_order` (80) reach
`interim/players` and go no further. The obvious build is an uplift mirroring penalties.
**Do not** — it would compound finding 2. A regular corner taker's assists from corners are
already in his `xa_per90`, and a free-kick specialist's goals are already in his `xg_per90`;
an uplift on top double counts exactly as the penalty term does, for a much smaller effect
(direct FK conversion is ~5-8%). Set-piece order adds information only where duty has
CHANGED, and the archive has no historical duty to compare against.
