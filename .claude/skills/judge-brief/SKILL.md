---
name: judge-brief
description: Adversarially review an `fpl myteam` brief (out.md, or any --brief output) before acting on it. Use when asked to judge, check, sanity-check, second-guess or sign off a transfer recommendation, a captaincy pick, a chip decision, or the weekly brief — and before making a real FPL transfer.
---

# Judging a weekly brief

The model is honest but not wise. It reports its own numbers faithfully and then recommends
things those numbers do not support, because its objective has known defects that were measured
and deliberately left in place. This skill is the standing check against that.

**Judge the recommendation against the brief's own figures first.** Almost every bad call this
project has produced was visible in the brief that proposed it — the numbers were printed
correctly and read past.

Run the checks in order. Stop and report as soon as one fails decisively; do not accumulate a
verdict from weak signals.

---

## 0. Is the brief stale?

```bash
stat -c 'generated: %y' out.md
fpl status
fpl snapshots | tail -3
python -c "from fpl_expert.data.historical import load_history; h=load_history(); c=h[h.season=='2026-27']; print(sorted(int(g) for g in c.GW.unique()))"
```

A brief is stale if it predates the newest ingested gameweek, the newest pre-deadline snapshot,
or a `fpl publish`. **Say so before judging anything else** — a stale brief has been overtaken,
and the rest of the review is about a decision nobody is making.

This has already happened: a brief generated nine minutes before `fpl results` finished was
reviewed as though it were current.

## 1. Does the XI actually change?

The most informative check, and the one that caught the GW4 recommendation.

The brief prints `Changes the XI: **X** in, **Y** out`. **Count the names.** Then compare the
alternatives:

```
do nothing         XI+capt  52.77
1 free transfer    XI+capt  54.34
2 transfers + hit  XI+capt  54.34   <- identical, and costs 4
```

If a plan taking a hit fields the same eleven as a cheaper plan, **the hit buys nothing this
week**, whatever its horizon figure says.

## 2. Do the incoming players start, or sit?

```python
from fpl_expert.backtest.season_sim import pick_xi
_, starters = pick_xi(post_transfer_squad, rules, "expected_points")
# is each transfers_in name in set(starters["web_name"])?
```

A signing who goes straight to the bench is bought on **bench value the XI never realises**.
`recommend_transfers` maximises the sum of FIFTEEN, so a fourth-choice defender counts as much
as the captain. This is a known, measured defect: the fix (`bench_aware`) was built and
REJECTED at -126 / -65 / +98 pooled -31, so the defect is still live and must be caught by eye.

## 3. This week against the horizon

The brief prints both. They are in different units and are not comparable:

```
this gameweek       52.77  ->   54.34   +1.58
over the horizon                        +17.54   net +13.54 after a 4pt hit
```

`decay=0.84` over 6 gameweeks is a multiplier of **4.054**; a hit is one-off. So the optimiser's
effective bar for a -4 is about **0.99 points per week** — anything worth a point a week clears
it. With the `max_transfers` cap lifted, the policy takes seven transfers and six hits and
scores it as a gain.

## 4. Calibrate the horizon claim

Multiply by **0.436** before believing it. Across 111 backtested transfer decisions, realised
return regresses on forecast gain with that slope, stable in every season. Correlation is only
0.27, so the margin is weak as well as biased.

```
+17.54 gross  ->  ~+7.6 realistic  ->  +3.6 after the hit
```

The brief states this in its own "What the move buys" section. **Apply it; do not just read it.**

## 5. The deferral test

**Could this move be made next week with a free transfer?** If so, taking it now buys only the
weeks before it would otherwise land — both branches hold an identical squad afterwards, so
every later term cancels:

```
take the hit iff (this gameweek's XI gain from the extra transfers) > 4
```

Exact when one transfer is brought forward. `defer_aware` implements it and measured
+60 [+21, +100] pooled — the strongest decision-layer result in the project — but the sign
flipped (-41 / +197 / +24), so it is NOT the default. Apply it as judgement, and say that is
what you are doing.

**The exception that matters:** when the outgoing player is a *zero* — injured, transferred
abroad, suspended — deferring costs a full week of a real asset rather than a marginal upgrade.
That is when a hit is right. It was right in GW2 (two dead assets; the week returned 101 points)
and wrong in GW3 and GW4 (bench upgrades).

## 6. Squad-level tells

- **Captaincy spread.** If `cost_vs_best` for the second choice is under ~1.5, the squad has no
  premium and the armband returns ~10 where 16+ is available. No single transfer fixes that —
  say so rather than optimising around it.
- **Bench boost value.** A high number (>12) is a tell, not a feature: capital is sitting idle.
  A recommendation that *raises* it is usually misallocating budget.
- **Club concentration.** Three from one club is the limit and a risk, especially a promoted side.

## 7. What the model cannot see — state the relevant ones every time

- **No press conferences.** `status` and `chance_of_playing_next_round` are the only availability
  signals and they lag reporting. Late team news is the user's to apply.
- **Ownership plays no part.** `lambda_rank` is 0.0, `rank_objective` is backtest-only, and
  `ingest_ownership` writes a table nothing reads. The optimiser maximises raw points, not rank.
  For a top-10k goal a 65%-owned player is nearly rank-neutral while a 30%-owned one cuts both
  ways. Flag it when a pick is unusually popular or unusually rare.
- **Promoted clubs rest on almost no data.** Dixon-Coles fits them on a handful of matches. A
  ridge of 0.5 was adopted 2026-09-02 after an unpenalised fit produced `defence 3.000` at the
  parameter bound and forecast Liverpool to score 0.10 at home to Hull. That fix cleared its
  evidential bar narrowly. **Treat any recommendation loading up on a promoted side as
  provisional.**
- **Bonus** is modelled from realised rates on a superseded BPS formula; **defensive
  contribution** has about two seasons behind it. Both appear in the brief's own caveats.
- **Auto-substitutions are not modelled** (~9 blanks, ~27 points a season — understates you).

---

## Verdict

Lead with the decision, then the single most decisive number. Do not bury it.

State plainly where the brief is **right**. A good free transfer deserves endorsing, and the
brief's self-reporting — printing the this-week figure beside the horizon figure, carrying the
calibration warning — is what makes judging it possible at all.

If several independent checks agree, say so. That is stronger than any one of them and unusual
enough to be worth naming.

Close with what would change the verdict: pending refreshes (`fpl results`, `fpl odds --season
<current>`, `fpl minutes`, `fpl publish`), how far away the deadline is, and which blind spot
the user must cover themselves.

**Never argue the recommendation is wrong merely because it disagrees with a correction this
project rejected.** `bench_aware`, `defer_aware` and the joint reformulation were all measured
and all failed ground rule 2. They are reasoning tools for a specific decision, not a better
model — and ground rule 1 says a single case is not evidence.
