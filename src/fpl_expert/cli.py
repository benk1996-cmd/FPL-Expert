"""Command line entry point: `fpl <command>`."""

from __future__ import annotations

import logging
import sys

import typer

from .config import load_config, load_scoring_rules

app = typer.Typer(add_completion=False, help="FPL Expert — forecasting and squad optimisation.")


def _force_utf8_output() -> None:
    """Make stdout UTF-8 whatever the console default is.

    Windows consoles default to cp1252, which cannot encode a good share of the Premier
    League: printing a squad containing Çalhanoğlu or Guimarães crashed the command outright
    with a UnicodeEncodeError, after the solve had already run. The archive is stored as UTF-8
    throughout; only the terminal was narrow.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):       # a redirected or closed stream
                pass


def _setup_logging(verbose: bool) -> None:
    _force_utf8_output()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def update(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Pull the current season's core feeds from the FPL API."""
    _setup_logging(verbose)
    from .data.fpl_api import ingest_core

    counts = ingest_core()
    for name, n in counts.items():
        typer.echo(f"  {name:<10} {n:>6,} rows")


@app.command()
def history(
    seasons: list[str] = typer.Option(None, "--season", "-s", help="Repeatable; default = config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download past seasons from the historical archive and report column coverage."""
    _setup_logging(verbose)
    from .data.historical import ingest_history

    report = ingest_history(list(seasons) if seasons else None)
    typer.echo(report.to_string(index=False))


@app.command()
def odds(
    seasons: list[str] = typer.Option(None, "--season", "-s", help="Repeatable; default = config"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download bookmaker odds from football-data.co.uk (free, no key)."""
    _setup_logging(verbose)
    from .data.odds import ingest_odds

    summary = ingest_odds(list(seasons) if seasons else None)
    typer.echo(summary.to_string(index=False))


@app.command()
def snapshot(
    if_due: bool = typer.Option(
        False, "--if-due", help="Only snapshot when a checkpoint before a deadline is due"
    ),
    checkpoints: str = typer.Option(
        "", "--checkpoints", help="Hours before deadline, comma separated (default 48,24,6,2,0.5)"
    ),
    reason: str = typer.Option("manual", "--reason"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Capture the game's mutable state before a deadline. Safe to schedule blindly."""
    _setup_logging(verbose)
    from .data.snapshot import DEFAULT_CHECKPOINTS, snapshot_due, take_snapshot

    if if_due:
        ladder = (
            tuple(float(c) for c in checkpoints.split(",")) if checkpoints else DEFAULT_CHECKPOINTS
        )
        due, why = snapshot_due(ladder)
        typer.echo(why)
        if not due:
            raise typer.Exit(0)

    manifest = take_snapshot(reason=reason)
    for key in ("stamp", "target_gw", "deadline", "taken_before_deadline",
                "hours_to_deadline", "players", "fixtures", "odds_rows"):
        typer.echo(f"  {key:<22} {manifest[key]}")
    if not manifest["taken_before_deadline"]:
        typer.secho(
            "  WARNING: taken after the deadline — recorded, but not usable as features.",
            fg=typer.colors.YELLOW,
        )


@app.command()
def ownership(
    gw: int = typer.Option(..., "--gw", help="A COMPLETED gameweek; picks 404 before a deadline"),
    max_past_rank: int = typer.Option(100_000, "--max-past-rank", help="Skill filter threshold"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build the skilled-manager cohort and compute effective ownership for a gameweek."""
    _setup_logging(verbose)
    from .data.ownership import ingest_ownership

    eo = ingest_ownership(gw, max_past_rank=max_past_rank)
    typer.echo(eo.head(20).to_string(index=False))


@app.command()
def minutes(
    evaluate: bool = typer.Option(True, "--evaluate/--no-evaluate", help="Walk-forward scoring"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train the minutes model, score it against the baseline, and save it."""
    _setup_logging(verbose)
    import warnings

    import pandas as pd

    from .backtest.metrics import compare
    from .backtest.walkforward import holdout_tail, season_splits
    from .config import project_root
    from .data.historical import load_history
    from .models.minutes import MinutesModel, TransitionBaseline, build_features

    warnings.filterwarnings("ignore")
    features = build_features(load_history())
    typer.echo(f"{len(features):,} player-gameweeks")

    if evaluate:
        rows = []
        for season, train, test in season_splits(features, min_train_seasons=3):
            fit_set, valid = holdout_tail(train)
            model = MinutesModel().fit(fit_set, valid)
            baseline = TransitionBaseline().fit(train)
            scored = compare(
                test["target"].values,
                {"minutes_model": model.predict_proba(test),
                 "baseline": baseline.predict_proba(test)},
            )
            scored.insert(0, "season", season)
            rows.append(scored)
        report = pd.concat(rows, ignore_index=True)
        typer.echo(report.round(4).to_string(index=False))
        typer.echo("\nmean across test seasons:")
        typer.echo(
            report.groupby("model")[["log_loss", "brier", "accuracy", "ece_60plus"]]
            .mean().round(4).to_string()
        )

    # Final model: everything we have.
    fit_set, valid = holdout_tail(features)
    final = MinutesModel().fit(fit_set, valid)
    path = project_root() / "data" / "processed" / "models" / "minutes.txt"
    final.save(path)
    typer.echo(f"\nsaved -> {path.relative_to(project_root())}")


@app.command()
def match(
    gw: int = typer.Option(None, "--gw", help="Restrict to one gameweek"),
    market_weight: float = typer.Option(0.8, "--market-weight"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fit the match model and forecast clean sheets and goals for upcoming fixtures."""
    _setup_logging(verbose)
    import warnings

    from .data.storage import read_table, write_table
    from .data.teams import normalise_series
    from .models.match_sim import DixonColes, build_match_forecasts

    warnings.filterwarnings("ignore")
    results = read_table("external", "odds_eval")
    model = DixonColes().fit(results)
    typer.echo(
        f"home advantage {model.home_advantage:.3f}  rho {model.rho:.4f}  "
        f"teams {len(model.teams)}"
    )

    fixtures = read_table("interim", "fixtures", season="2026-27")
    teams = read_table("interim", "teams", season="2026-27").set_index("id")["name"]
    fixtures = fixtures.assign(
        home_team=fixtures["team_h"].map(teams), away_team=fixtures["team_a"].map(teams)
    ).dropna(subset=["home_team", "away_team", "event"])
    if gw is not None:
        fixtures = fixtures[fixtures["event"] == gw]

    forecasts = build_match_forecasts(fixtures, model, market_weight=market_weight)
    normalise_series(forecasts["team"], strict=False)  # surface any naming drift early
    write_table(forecasts, "processed", "match_forecasts", season="2026-27")

    show = forecasts.nlargest(12, "p_clean_sheet")
    typer.echo(
        show[["gw", "team", "opponent", "is_home", "expected_goals_for",
              "expected_goals_against", "p_clean_sheet"]].round(3).to_string(index=False)
    )
    typer.echo(f"\n{len(forecasts)} team-fixtures -> processed/match_forecasts")


@app.command()
def squad(
    gw: int = typer.Option(1, "--gw"),
    budget: float = typer.Option(None, "--budget", help="Defaults to the configured 100.0"),
    bench_weight: float = typer.Option(0.10, "--bench-weight"),
    horizon: int = typer.Option(
        None, "--horizon", help="Gameweeks to value over; 1 is the old myopic behaviour"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Forecast a horizon of gameweeks and solve for the best legal squad, XI and captain.

    Valued over several gameweeks, not one, because a squad is held rather than rented. A
    single-gameweek objective divides each player's price by one week of return and therefore
    refuses every premium: measured in the season simulator, myopic valuation dropped Haaland
    from 34 starts in 38 to 2, and cost 296 season points. `--horizon 1` restores it.
    """
    _setup_logging(verbose)
    import warnings

    from .config import load_config, load_scoring_rules
    from .data.snapshot import MissingSnapshotError
    from .data.storage import write_table
    from .models.points import aggregate_gameweek
    from .optimise.squad import select_squad
    from .optimise.transfers import horizon_points
    from .pipeline import forecast_gameweek

    warnings.filterwarnings("ignore")
    cfg, rules = load_config(), load_scoring_rules()
    span = horizon if horizon is not None else cfg.optimise.horizon_gws

    forecasts = forecast_gameweek(gw)
    per_player = aggregate_gameweek(forecasts)
    write_table(forecasts, "processed", "player_forecasts", gw=gw)

    points_col = "expected_points"
    if span > 1:
        typer.echo(f"valuing over GW{gw}-{gw + span - 1} (--horizon {span})")
        by_gw = {gw: per_player}
        for future in range(gw + 1, gw + span):
            try:
                by_gw[future] = aggregate_gameweek(
                    forecast_gameweek(future, planning=True)
                )
            except (ValueError, KeyError, MissingSnapshotError) as exc:
                typer.echo(f"  GW{future}: unavailable ({exc}) — horizon truncated")
                break
        table = horizon_points(by_gw, decay=cfg.optimise.future_decay)
        per_player = per_player.merge(table, on="player_id", how="left")
        per_player["horizon_points"] = per_player["horizon_points"].fillna(
            per_player["expected_points"]
        )
        points_col = "horizon_points"

    solution = select_squad(
        per_player,
        budget=budget if budget is not None else rules["squad"]["budget"],
        squad_quota=rules["squad"]["positions"],
        formation=rules["squad"]["formation"],
        max_per_club=rules["squad"]["max_per_club"],
        bench_weight=bench_weight,
        points_col=points_col,
        double_captain=points_col == "expected_points",
    )

    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    show = ["web_name", "position", "team", "price", "expected_points"]
    if points_col != "expected_points":
        show.append(points_col)
    xi = solution.starting_xi.assign(_o=lambda d: d["position"].map(order)).sort_values(
        ["_o", "expected_points"], ascending=[True, False]
    )
    typer.echo("\nSTARTING XI")
    typer.echo(xi[show].round(2).to_string(index=False))
    typer.echo("\nBENCH")
    typer.echo(solution.bench[show].round(2).to_string(index=False))
    typer.echo("\n" + solution.summary())


@app.command()
def simulate(
    seasons: list[str] = typer.Option(["2025-26"], "--season", "-s", help="Repeatable"),
    gws: int = typer.Option(38, "--gws", help="How many gameweeks to replay"),
    ablation: bool = typer.Option(True, "--ablation/--no-ablation"),
    chips: bool = typer.Option(True, "--chips/--no-chips", help="Play chips in the replay"),
    lookahead: bool = typer.Option(
        True, "--lookahead/--no-lookahead",
        help="Build each gameweek's horizon from forecasts made AT that gameweek (default). "
             "--no-lookahead reuses each target week's own forecast, which is faster and "
             "reads data the decision could not have had.",
    ),
    strict: bool = typer.Option(
        True, "--strict/--reuse-model",
        help="Retrain the minutes model per season on PRIOR seasons only (default). "
             "--reuse-model loads the saved model, which has seen the test season.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Replay past seasons end-to-end and rank which components carry the decisions."""
    _setup_logging(verbose)
    import warnings

    import pandas as pd

    from .backtest.historical_forecast import (
        attach_actual_points,
        forecast_horizon,
        to_gameweek_level,
    )
    from .backtest.season_sim import (
        LOOKAHEAD_SPAN,
        ablate,
        benchmark_form,
        benchmark_price,
        benchmark_random,
        simulate_season,
    )
    from .backtest.walkforward import holdout_tail, prior_seasons
    from .config import project_root
    from .data.historical import load_history
    from .data.storage import read_table, write_table
    from .models.minutes import MinutesModel, build_features

    warnings.filterwarnings("ignore")
    history = load_history()
    results = read_table("external", "odds_eval")
    try:
        odds = read_table("external", "odds_features")
    except FileNotFoundError:
        odds = None
        typer.echo("no opening odds found — the match model will run unblended")
    features = build_features(history)

    summaries, ablations = [], []
    for season in seasons:
        if strict:
            # The saved model is trained on every season including this one. Reusing it
            # here would let the minutes model — the largest single driver of points — see
            # the future of the very gameweeks it is being scored on.
            prior = prior_seasons(features, season)
            if prior.empty:
                typer.echo(f"{season}: no prior seasons to train on — skipped")
                continue
            fit_set, valid = holdout_tail(prior)
            model = MinutesModel().fit(fit_set, valid)
            typer.echo(f"{season}: minutes model trained on {prior['season'].nunique()} "
                       f"prior season(s), {len(prior):,} rows")
        else:
            model = MinutesModel.load(
                project_root() / "data" / "processed" / "models" / "minutes.txt"
            )

        frames, forward = [], []
        for gw in range(1, gws + 1):
            # The transfer horizon spans six gameweeks and the chip planner looks one
            # further, so a decision week needs GW..GW+6 forecast from its own vantage point.
            targets = range(gw, min(gw + LOOKAHEAD_SPAN, gws + 1)) if lookahead else [gw]
            try:
                built = forecast_horizon(
                    history, results, model, features, season, gw, targets, odds=odds
                )
            except (ValueError, KeyError) as exc:
                typer.echo(f"  {season} GW{gw}: skipped ({exc})")
                continue
            frames.append(built.pop(gw))
            forward.extend(built.values())
        if not frames:
            continue

        # Fixture level is kept as well as gameweek level: the points DISTRIBUTION is built
        # per fixture and convolved, so collapsing first would lose the double gameweeks it
        # exists to handle.
        fixture_level = pd.concat(frames, ignore_index=True)
        write_table(fixture_level, "processed", "sim_fixtures", season=season)
        forecasts = attach_actual_points(to_gameweek_level(fixture_level), history)
        write_table(forecasts, "processed", "sim_forecasts", season=season)

        horizon = None
        if forward:
            # Realised points are deliberately NOT joined on: these frames exist only to be
            # valued, and the one thing a forward valuation must never see is an outcome.
            forward_level = to_gameweek_level(pd.concat(forward, ignore_index=True))
            write_table(forward_level, "processed", "sim_horizon", season=season)
            horizon = {
                as_of: {gw: block for gw, block in window.groupby("gw", sort=True)}
                for as_of, window in forward_level.groupby("as_of_gw", sort=True)
            }
            # The decision week's own frame carries realised points and is the one the
            # simulator scores against, so it comes from `forecasts`, not from `forward`.
            for as_of, window in horizon.items():
                window[as_of] = forecasts[forecasts["gw"] == as_of]
            typer.echo(
                f"  {season}: horizon built point-in-time over "
                f"{len(horizon)} decision weeks"
            )

        for result in (
            simulate_season(forecasts, label="model", use_chips=chips, lookahead=horizon),
            # `form` is the bar that means something: not collinear with the budget, and it
            # ranks players at 0.641 against price's 0.381. `price-only` is retained for
            # continuity only — its optimum is degenerate, so its score is one arbitrary draw
            # from a large set of near-ties and beating it establishes nothing.
            benchmark_form(forecasts, lookahead=horizon),
            benchmark_price(forecasts),
            benchmark_random(forecasts),
        ):
            row = result.summary()
            row["season"] = season
            summaries.append(row)

        if ablation:
            table = ablate(forecasts, lookahead=horizon)
            table["season"] = season
            ablations.append(table)

    if not summaries:
        raise typer.Exit(1)

    table = pd.DataFrame(summaries)
    typer.echo("\nSEASON RESULTS" + ("  (strict: no test-season leakage)" if strict else ""))
    typer.echo(
        table.pivot(index="season", columns="label", values="total_points").to_string()
    )
    typer.echo("\nper gameweek:")
    typer.echo(
        table.pivot(index="season", columns="label", values="points_per_gw").round(1).to_string()
    )

    if ablations:
        combined = pd.concat(ablations, ignore_index=True)
        typer.echo("\nCOMPONENT ABLATION — season points lost when each component is removed")
        typer.echo(
            combined.pivot(index="variant", columns="season", values="delta")
            .assign(mean=lambda d: d.mean(axis=1)).sort_values("mean").round(1).to_string()
        )


@app.command()
def prices(
    season: str = typer.Option("2025-26", "--season", help="Season to score, trained on prior"),
    top: int = typer.Option(15, "--top", help="How many risers and fallers to show"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fit the price-change model walk-forward and list expected risers and fallers."""
    _setup_logging(verbose)
    import warnings

    from .backtest.metrics import log_loss
    from .backtest.walkforward import prior_seasons
    from .data.historical import load_history
    from .models.prices import NoChangeBaseline, PriceModel, build_features

    warnings.filterwarnings("ignore")
    features = build_features(load_history())
    train, test = prior_seasons(features, season), features[features["season"] == season]
    if train.empty or test.empty:
        typer.echo(f"no prior seasons to train on for {season}")
        raise typer.Exit(1)

    model, baseline = PriceModel().fit(train), NoChangeBaseline().fit(train)
    y = test["target"].to_numpy()
    model_loss, base_loss = log_loss(y, model.predict_proba(test)), log_loss(
        y, baseline.predict_proba(test)
    )
    typer.echo(
        f"{len(train):,} training rows -> {len(test):,} scored\n"
        f"log loss {model_loss:.4f} vs base-rate {base_loss:.4f} "
        f"({1 - model_loss / base_loss:+.1%})"
    )

    latest = test[test["GW"] == test["GW"].max()].copy()
    latest["expected_change"] = model.expected_change(latest)
    probabilities = model.predict_proba(latest)
    latest["p_rise"], latest["p_fall"] = probabilities[:, 2], probabilities[:, 0]
    show = ["name", "value", "selected", "net_fraction", "p_rise", "p_fall", "expected_change"]

    typer.echo(f"\nGW{int(latest['GW'].iloc[0])} — most likely to RISE")
    typer.echo(latest.nlargest(top, "expected_change")[show].round(4).to_string(index=False))
    typer.echo("\nmost likely to FALL")
    typer.echo(latest.nsmallest(top, "expected_change")[show].round(4).to_string(index=False))


# Policy variants worth comparing, all of them currently unresolved on single replays.
ENSEMBLE_VARIANTS: dict[str, dict] = {
    "horizon": {},
    "myopic": {"horizon": 0},
    "smoothed": {"smoothing": 1.0},
    # The forecast margin behind a transfer is overstated ~2.3x (slope 0.436, stable in every
    # season), so the optimiser compares an inflated gain against a nominal 4-point hit. These
    # sweep the bar it actually demands. 4.0 is the incumbent and was never itself measured.
    **{f"hit_bar_{bar:g}": {"hit_bar": float(bar)} for bar in (6, 8, 9, 10, 12)},
    # Does the armband term carry the horizon on its own? The 2024-25 diagnosis found the
    # captaincy contribution consistently positive (+41 / +48) while transfer aggression was
    # consistently negative, which is two effects the single `horizon` switch conflates.
    "decay_0.70": {"decay": 0.70},
    "decay_0.92": {"decay": 0.92},
    "free_hit": {"allowed_chips": ("bench_boost", "triple_captain", "free_hit")},
    "no_captaincy_term": {"captaincy_weight": 0.0},
    # The bar the model has to clear. Decisions read `price_score`, so the perturbation lands
    # there too and the baseline explores its own decision paths on the same footing.
    #
    # It keeps the DEFAULT horizon, matching `season_sim.benchmark_price`. Setting horizon=0
    # here on the reasoning that "price barely changes, so a horizon over it is meaningless"
    # was wrong by 551 points on 2023-24 (1370 against 1921): the horizon column also carries
    # the captaincy uplift and drives the transfer MILP, so it changes the policy even when
    # the underlying quantity is near-constant. A baseline must be given its best form.
    "price_only": {"points_col": "price_score"},
    # The bar that actually means something. Unlike price, form is not collinear with the
    # budget constraint, so its optimum is a real one rather than one of a huge set of ties.
    "form": {"points_col": "form_score"},
}


@app.command()
def ensemble(
    seasons: list[str] = typer.Option(
        ["2023-24", "2024-25", "2025-26"], "--season", "-s", help="Repeatable"
    ),
    variants: list[str] = typer.Option(
        ["horizon", "myopic"], "--variant", "-v",
        help=f"Repeatable. One of: {', '.join(ENSEMBLE_VARIANTS)}",
    ),
    baseline: str = typer.Option("myopic", "--baseline"),
    paths: int = typer.Option(10, "--paths", help="Perturbed decision paths per season"),
    jitter: float = typer.Option(0.001, "--jitter", help="Relative forecast perturbation"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Compare policies over many perturbed DECISION paths instead of one replay each.

    A season replay is one draw from a chaotic process: a 0.1% forecast perturbation moves a
    season total with a standard deviation of 38 points, because one flipped transfer changes
    the squad and the squad changes every later decision. Most effects this project measures
    are smaller than that, which is why single replays kept producing sign flips.

    Requires `fpl simulate` to have been run first — it reads the cached forecasts and,
    when present, the point-in-time horizon.
    """
    _setup_logging(verbose)
    import warnings

    import pandas as pd

    from .backtest.ensemble import ensemble_season, per_season_paired, resolvable
    from .backtest.season_sim import attach_form
    from .data.storage import read_table, write_table

    warnings.filterwarnings("ignore")
    unknown = [v for v in [*variants, baseline] if v not in ENSEMBLE_VARIANTS]
    if unknown:
        typer.echo(f"unknown variant(s): {unknown}. Choose from {list(ENSEMBLE_VARIANTS)}")
        raise typer.Exit(1)
    chosen = list(dict.fromkeys([*variants, baseline]))

    runs = []
    for season in seasons:
        try:
            forecasts = read_table("processed", "sim_forecasts", season=season)
        except FileNotFoundError:
            typer.echo(f"{season}: no cached forecasts — run `fpl simulate` first")
            continue
        try:
            horizon = read_table("processed", "sim_horizon", season=season)
            lookahead = {
                int(a): {int(g): b for g, b in w.groupby("gw", sort=True)}
                for a, w in horizon.groupby("as_of_gw", sort=True)
            }
            for as_of, window in lookahead.items():
                window[as_of] = forecasts[forecasts["gw"] == as_of]
        except FileNotFoundError:
            lookahead = None
            typer.echo(f"{season}: no point-in-time horizon — decisions will read ahead")

        # Baselines score on derived columns, which every frame must carry before any variant
        # runs — including the horizon views, or the lookahead merge drops them. `attach_form`
        # gives those views the DECISION week's form, keeping the baseline point-in-time too.
        forecasts, lookahead = attach_form(forecasts, lookahead)
        forecasts = forecasts.assign(price_score=forecasts["price"])
        if lookahead is not None:
            lookahead = {
                as_of: {gw: f.assign(price_score=f["price"]) for gw, f in window.items()}
                for as_of, window in lookahead.items()
            }

        for name in chosen:
            table = ensemble_season(
                forecasts, label=name, paths=paths, jitter=jitter,
                lookahead=lookahead, use_chips=True, **ENSEMBLE_VARIANTS[name],
            )
            table["season"] = season
            runs.append(table)
            typer.echo(
                f"  {season} {name:18s} mean {table['points'].mean():7.0f}  "
                f"sd {table['points'].std(ddof=1):5.1f}"
            )

    if not runs:
        raise typer.Exit(1)
    combined = pd.concat(runs, ignore_index=True)
    # Persisted because an ensemble is expensive — 24 paths is roughly 40 minutes — and the
    # per-path totals support analyses the printed summary does not, per-season paired
    # intervals above all.
    write_table(combined, "processed", "ensemble_runs")
    typer.echo(f"\nper-path totals written to `ensemble_runs` ({len(combined)} rows)")

    typer.echo(f"\nPAIRED against '{baseline}', differenced within each shared path")
    typer.echo(resolvable(combined, baseline).round(2).to_string(index=False))

    if combined["season"].nunique() > 1:
        # Pooling mixes path noise with genuine between-season difference, so a large effect
        # that changes sign between seasons averages to nothing and reads as "no effect".
        # Only this cut tells those two situations apart, and they call for opposite responses.
        typer.echo("\nPER SEASON, paired within each shared path")
        typer.echo(per_season_paired(combined, baseline).round(2).to_string(index=False))

    typer.echo(
        "\n`adoptable` needs BOTH: a paired interval excluding zero (the effect is real "
        "across\nthese seasons) AND the sign agreeing in every season (ground rule 2). "
        "Path averaging\nsharpens the first; it cannot substitute for the second."
    )


@app.command()
def repeat(
    seasons: list[str] = typer.Option(["2025-26"], "--season", "-s", help="Repeatable"),
    draws: int = typer.Option(30, "--draws", help="Resampled seasons per variant"),
    baseline: str = typer.Option("bb_tc", "--baseline", help="Variant to difference against"),
    field: int = typer.Option(20_000, "--field", help="Simulated rival managers"),
    seed: int = typer.Option(0, "--seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Replay seasons many times over resampled outcomes, to put error bars on comparisons.

    Requires `fpl simulate` to have been run first — it reads the forecasts that produced.
    """
    _setup_logging(verbose)
    import warnings

    import pandas as pd

    from .backtest.field_sim import attach_ownership
    from .backtest.repeat_sim import STANDARD_VARIANTS, paired_comparison, repeat_season
    from .data.historical import load_history
    from .data.storage import read_table

    warnings.filterwarnings("ignore")
    history = load_history()

    combined = []
    for season in seasons:
        forecasts = attach_ownership(
            read_table("processed", "sim_forecasts", season=season), history
        )
        fixtures = read_table("processed", "sim_fixtures", season=season)
        typer.echo(f"\n{season}: {draws} draws x {len(STANDARD_VARIANTS)} variants")

        result = repeat_season(
            forecasts, fixtures, n_draws=draws, seed=seed, field_size=field, season=season
        )
        typer.echo(result.summary().to_string())

        comparison = paired_comparison(result.draws, baseline)
        typer.echo(f"\npaired against {baseline!r} — same outcomes within each draw:")
        typer.echo(comparison.to_string(index=False))

        frame = result.draws.assign(season=season)
        combined.append(frame)

    if len(combined) > 1:
        # Seasons pool as extra draws: each is a different world, and the paired structure
        # holds within each one.
        pooled = pd.concat(combined, ignore_index=True)
        pooled["draw"] = pooled["season"] + "#" + pooled["draw"].astype(str)
        typer.echo("\nPOOLED ACROSS SEASONS")
        typer.echo(paired_comparison(pooled, baseline).to_string(index=False))


@app.command()
def report(
    gw: int = typer.Option(1, "--gw"),
    out: str = typer.Option(None, "--out", help="Write markdown here instead of stdout"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Produce the weekly decision brief: squad, captain, risks and differentials."""
    _setup_logging(verbose)
    import warnings
    from pathlib import Path

    from .config import load_scoring_rules, project_root
    from .models.points import aggregate_gameweek
    from .optimise.squad import select_squad
    from .pipeline import forecast_gameweek
    from .reporting.gw_report import build_report

    warnings.filterwarnings("ignore")
    rules = load_scoring_rules()
    forecasts = aggregate_gameweek(forecast_gameweek(gw))
    solution = select_squad(
        forecasts,
        budget=rules["squad"]["budget"],
        squad_quota=rules["squad"]["positions"],
        formation=rules["squad"]["formation"],
        max_per_club=rules["squad"]["max_per_club"],
    )
    markdown = build_report(forecasts, solution, gw)

    path = Path(out) if out else project_root() / "data" / "processed" / f"gw{gw}_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    typer.echo(markdown if out is None else f"written -> {path}")


@app.command()
def validate(
    season: str = typer.Option("2025-26", "--season"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Score stored simulation forecasts: component accuracy, calibration, top-N precision."""
    _setup_logging(verbose)
    from .backtest.validate import calibration_summary, component_accuracy, top_n_precision
    from .data.storage import read_table

    forecasts = read_table("processed", "sim_forecasts", season=season)
    typer.echo("COMPONENT ACCURACY")
    typer.echo(component_accuracy(forecasts).to_string(index=False))
    typer.echo("\nCALIBRATION")
    typer.echo(calibration_summary(forecasts).to_string(index=False))
    precision = top_n_precision(forecasts)
    if not precision.empty:
        typer.echo(f"\nTOP-20 PRECISION  mean {precision['precision'].mean():.3f}")


@app.command()
def monitor(
    season: str = typer.Option("2025-26", "--season"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the standing health checks against pre-committed thresholds."""
    _setup_logging(verbose)
    from .backtest.monitor import check, retraining_due
    from .data.snapshot import list_snapshots
    from .data.storage import read_table

    forecasts = read_table("processed", "sim_forecasts", season=season)
    results = check(forecasts)
    typer.echo(results.to_string(index=False))
    if (results["status"] == "BREACH").any():
        typer.secho("\nBREACH detected — investigate before trusting this week's picks.",
                    fg=typer.colors.RED)
    _, why = retraining_due(list_snapshots())
    typer.echo(f"\nretraining: {why}")


@app.command()
def myteam(
    entry: int = typer.Option(..., "--entry", help="Your FPL manager id (from your team URL)"),
    gw: int = typer.Option(None, "--gw", help="Gameweek to plan for; defaults to the next"),
    horizon: int = typer.Option(None, "--horizon", help="Gameweeks to plan over"),
    max_transfers: int = typer.Option(2, "--max-transfers"),
    brief: str = typer.Option(
        None, "--brief", help="Write a full markdown brief here (transfers, chips, prices)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Pull your real squad and recommend transfers for the coming gameweek."""
    _setup_logging(verbose)
    import warnings

    from .config import load_config, load_scoring_rules
    from .data.fpl_api import FplApi, next_gameweek
    from .data.my_team import bank, current_squad, fetch_entry, free_transfers
    from .data.snapshot import PointInTime
    from .models.points import aggregate_gameweek
    from .optimise.transfers import horizon_points, recommend_transfers
    from .pipeline import forecast_gameweek

    warnings.filterwarnings("ignore")
    cfg, rules = load_config(), load_scoring_rules()
    api = FplApi()
    target = gw if gw is not None else next_gameweek(api.bootstrap_static())
    span = horizon if horizon is not None else cfg.optimise.horizon_gws

    profile = fetch_entry(api, entry)
    typer.echo(f"{profile.get('name', '?')} — GW{target}, planning over {span} gameweek(s)")

    players = PointInTime.for_gameweek(target).players()
    squad = current_squad(api, entry, target, players)
    available = free_transfers(api, entry, target, rules["transfers"]["max_banked"])
    in_bank = bank(profile)
    typer.echo(f"squad of {len(squad)}, £{in_bank:.1f}m banked, {available} free transfer(s)")

    # Future gameweeks are PLANNING, not evaluation: no pre-deadline snapshot exists for
    # them yet, so the horizon is built from the latest known state.
    per_gw = {
        g: aggregate_gameweek(forecast_gameweek(g, planning=g != target))
        for g in range(target, target + span)
    }
    horizon_table = horizon_points(per_gw, decay=cfg.optimise.future_decay)

    latest = per_gw[target].merge(horizon_table, on="player_id", how="left")
    latest["horizon_points"] = latest["horizon_points"].fillna(0.0)
    held = latest[latest["player_id"].isin(squad["id"])].merge(
        squad[["id", "selling_price"]].rename(columns={"id": "player_id"}), on="player_id"
    )

    plan = recommend_transfers(
        held, latest, bank=in_bank, free_transfers=available,
        max_per_club=rules["squad"]["max_per_club"], max_transfers=max_transfers,
    )
    typer.echo("\n" + plan.summary())
    if plan.n_transfers:
        show = ["web_name", "position", "team", "price", "horizon_points"]
        typer.echo("\nOUT")
        typer.echo(plan.transfers_out[
            [c if c != "price" else "selling_price" for c in show]
        ].round(2).to_string(index=False))
        typer.echo("\nIN")
        typer.echo(plan.transfers_in[show].round(2).to_string(index=False))

    if brief:
        _write_myteam_brief(brief, latest, held, squad, plan, target, span, rules)


def _price_moves(latest, held):
    """Expected overnight price moves: risers anywhere, fallers among players you hold.

    Asymmetric on purpose. A rise you do not own is an opportunity cost you may want to act
    on before it lands; a fall you do not own costs you nothing. The reverse for fallers —
    only your own matter, because that is where team value actually leaks.

    Returns `(None, None)` if the live feed has no transfer flow yet, which is the case
    before the first gameweek of a season — the brief simply omits the section.
    """
    import pandas as pd

    from .data.historical import load_history
    from .models.prices import PriceModel, build_features

    try:
        # Fitted on the whole archive, not walk-forward: this predicts the coming week rather
        # than scoring a held-out season, so every past gameweek is legitimately in the past.
        model = PriceModel().fit(build_features(load_history()))
    except (ValueError, RuntimeError, KeyError) as exc:
        logging.getLogger(__name__).info(
            "price model unavailable, omitting the section: %s", exc
        )
        return None, None

    needed = {"transfers_balance", "selected", "transfers_in", "transfers_out"}
    if not needed <= set(latest.columns):
        return None, None

    scored = latest.copy()
    probabilities = model.predict_proba(scored)
    scored["p_fall"], scored["p_rise"] = probabilities[:, 0], probabilities[:, 2]
    scored["expected_change"] = model.expected_change(scored)

    risers = scored.nlargest(6, "expected_change")
    mine = scored[scored["player_id"].isin(held["player_id"])]
    fallers = mine.nsmallest(6, "expected_change") if not mine.empty else pd.DataFrame()
    return risers, fallers


def _write_myteam_brief(path, latest, held, squad, plan, gw, span, rules) -> None:
    """The weekly brief for a squad you actually own — transfers, chips and prices together.

    Kept out of `report`, which solves for an ideal squad from scratch. Chip advice and
    transfer advice are only meaningful against a squad you hold, so they belong here.
    """
    from pathlib import Path

    import pandas as pd

    from .backtest.season_sim import pick_xi
    from .optimise.chips import bench_boost_value, triple_captain_value
    from .optimise.squad import SquadSolution
    from .reporting.gw_report import build_report

    ranked, starters = pick_xi(held, rules, "expected_points")
    bench = ranked.drop(index=starters.index)
    captain = starters.nlargest(1, "expected_points")

    solution = SquadSolution(
        squad=held, starting_xi=starters, bench=bench,
        captain=captain.iloc[0] if not captain.empty else pd.Series(dtype=object),
        vice_captain=(
            starters.nlargest(2, "expected_points").iloc[-1]
            if len(starters) > 1 else pd.Series(dtype=object)
        ),
        total_cost=float(held["price"].fillna(0).sum()),
        expected_points=float(starters["expected_points"].sum()),
        status="held squad",
    )

    # Only the two chips that do not require rebuilding the squad are valued here. Wildcard
    # and free hit are worth what a from-scratch rebuild would gain, which is a different
    # question and belongs with the transfer plan rather than beside it.
    values = {
        "bench_boost": bench_boost_value(
            held, set(bench["player_id"]), "expected_points"
        ),
        "triple_captain": triple_captain_value(
            starters,
            captain["player_id"].iloc[0] if not captain.empty else None,
            "expected_points",
        ),
    }

    risers, fallers = _price_moves(latest, held)
    markdown = build_report(
        latest, solution, gw, plan=plan, chip_values=values,
        risers=risers, fallers=fallers,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(markdown, encoding="utf-8")
    typer.echo(f"\nbrief written -> {path}")


@app.command()
def snapshots(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """List snapshots taken so far, and flag gameweeks with no pre-deadline capture."""
    _setup_logging(verbose)
    from .data.snapshot import list_snapshots

    df = list_snapshots()
    if df.empty:
        typer.echo("no snapshots yet — run `fpl snapshot`")
        raise typer.Exit(0)
    cols = ["stamp", "target_gw", "hours_to_deadline", "taken_before_deadline", "reason"]
    typer.echo(df[cols].to_string(index=False))

    covered = set(df.loc[df["taken_before_deadline"], "target_gw"].dropna().astype(int))
    if covered:
        missing = sorted(set(range(1, max(covered) + 1)) - covered)
        if missing:
            typer.secho(f"\nno pre-deadline snapshot for GW {missing}", fg=typer.colors.YELLOW)


@app.command()
def status(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Show where the season is and what the loaded rules say."""
    _setup_logging(verbose)
    from .data.fpl_api import FplApi, current_gameweek, next_gameweek, parse_events

    cfg, rules = load_config(), load_scoring_rules()
    bootstrap = FplApi().bootstrap_static()
    events = parse_events(bootstrap)
    nxt = next_gameweek(bootstrap)

    typer.echo(f"season          {cfg.project.get('season')}")
    typer.echo(f"rules verified  {rules.get('verified')} ({rules.get('verified_on')})")
    typer.echo(f"managers        {bootstrap['total_players']:,}")
    typer.echo(f"current GW      {current_gameweek(bootstrap)}")
    if nxt is not None:
        deadline = events.loc[events["id"] == nxt, "deadline_time"].iloc[0]
        typer.echo(f"next GW         {nxt}  deadline {deadline:%Y-%m-%d %H:%M UTC}")
    typer.echo(f"target rank     {cfg.optimise.objective.target_rank:,}")


if __name__ == "__main__":
    app()
