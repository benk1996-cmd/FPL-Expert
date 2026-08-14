# Next session — where to pick up

Updated 2026-08-13. GW1 deadline is **2026-08-21 17:30 UTC**.

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

448 tests, clean lint. Three seasons, strict walk-forward, MILP transfers, BB+TC chips,
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

## Blocked until the season starts (do not attempt now)

- Top-10k effective ownership (`fpl ownership`) — needs completed gameweeks.
- `fpl myteam` — needs the user's FPL entry id, still not supplied.
- Snapshot accumulation — the scheduled task fires before the GW1 deadline.

**Set-piece order is NOT blocked and was never blocked** — this list said it was, wrongly,
until 2026-08-14. The FPL API publishes `penalties_order`, `direct_freekicks_order` and
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
