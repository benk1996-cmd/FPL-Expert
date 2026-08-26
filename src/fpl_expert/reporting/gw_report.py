"""Weekly decision brief.

Written as decision support rather than an oracle. Every recommendation carries the cost of
deviating from it, so a manager who knows something the model does not — and the model reads
no press conferences — can overrule it while seeing exactly what that costs. A brief that
only says "do this" is impossible to disagree with intelligently.

Deliberately surfaced alongside the picks:
  * the cost of the next-best alternative at each position, so overriding is priced;
  * which components each recommendation rests on, since bonus and DefCon are known to be
    the weakest models in the system;
  * where the forecast disagrees most with crowd ownership, which is where rank is won.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..models.bonus import bonus_confidence_note

log = logging.getLogger(__name__)

DISPLAY = ["web_name", "position", "team", "price", "expected_points"]


def _fmt(df: pd.DataFrame, columns: list[str] | None = None, rows: int = 10) -> str:
    present = [c for c in (columns or df.columns) if c in df.columns]
    return df.head(rows)[present].round(2).to_string(index=False)


def captain_options(forecasts: pd.DataFrame, squad_ids: set | None = None, top: int = 5):
    """Best armband candidates, with the gap to the next option."""
    pool = forecasts
    if squad_ids:
        pool = pool[pool["player_id"].isin(squad_ids)]
    ranked = pool.nlargest(top, "expected_points").copy()
    ranked["captain_points"] = 2 * ranked["expected_points"]
    ranked["cost_vs_best"] = ranked["captain_points"].max() - ranked["captain_points"]
    return ranked


def bench_risk(forecasts: pd.DataFrame, squad_ids: set) -> pd.DataFrame:
    """Squad members at real risk of not playing — the most common avoidable loss."""
    held = forecasts[forecasts["player_id"].isin(squad_ids)].copy()
    return held[held["p_appear"] < 0.7].sort_values("p_appear")


def transfer_section(plan) -> list[str]:
    """The transfer recommendation, in the brief rather than in a separate command.

    A brief that shows a squad but not how to reach it from the one you actually own is only
    half a decision. `net_gain` is already net of any hit, so a plan that survives to here has
    paid for itself on the planning horizon.
    """
    if plan is None:
        return []
    if plan.n_transfers == 0:
        return ["", "## Transfers", "**Roll.** No move clears the cost of making it."]

    show = ["web_name", "position", "team", "price", "horizon_points"]
    hit = f", {plan.hits} hit(s) costing {plan.hits * 4}" if plan.hits else ", no hit"
    return [
        "",
        "## Transfers",
        (f"**{plan.n_transfers} transfer(s)**{hit}  ·  "
         f"net gain **{plan.net_gain:+.2f}** points over the horizon"),
        "```",
        "OUT",
        _fmt(plan.transfers_out, [c if c != "price" else "selling_price" for c in show], 5),
        "",
        "IN",
        _fmt(plan.transfers_in, show, 5),
        "```",
    ]


def chip_section(chip: str | None, reason: str = "", values: dict | None = None) -> list[str]:
    """What the chip planner thinks about this week.

    Reported even when the answer is "hold", because the useful information is usually the
    margin — a bench boost worth 12 against a typical 11 is a different decision from one
    worth 20, and only the first is worth arguing about.
    """
    lines = ["", "## Chips"]
    if chip:
        lines.append(f"**Play {chip.replace('_', ' ').title()}** — {reason}")
    else:
        lines.append("**Hold.** Nothing this week beats what the remaining windows offer.")
    if values:
        lines += [
            "```",
            "\n".join(f"{name.replace('_', ' '):<16} {value:6.1f}" for name, value in values.items()),
            "```",
        ]
    return lines


def price_section(risers: pd.DataFrame | None, fallers: pd.DataFrame | None) -> list[str]:
    """Expected price moves among players you hold or might buy.

    Team value is slow and compounding, so it never justifies a transfer on its own — but a
    move you were making anyway is worth timing around it.
    """
    if risers is None or fallers is None or (risers.empty and fallers.empty):
        return []
    columns = ["web_name", "team", "price", "p_rise", "p_fall", "expected_change"]
    lines = ["", "## Price moves", "Overnight changes are not worth a transfer by themselves."]
    if not risers.empty:
        lines += ["```", "LIKELY RISERS", _fmt(risers, columns, rows=6), "```"]
    if not fallers.empty:
        lines += ["```", "LIKELY FALLERS (in your squad)", _fmt(fallers, columns, rows=6), "```"]
    return lines


def xi_value(squad: pd.DataFrame, rules: dict, points_col: str = "expected_points") -> float:
    """This gameweek's score for a squad: the best legal XI plus the armband again.

    The armband is counted twice because the captain scores twice — the same convention the
    manifest and `SquadSolution.expected_points` use, so the two are comparable.
    """
    from ..backtest.season_sim import pick_xi

    if squad.empty:
        return 0.0
    _, starters = pick_xi(squad, rules, points_col)
    if starters.empty:
        return 0.0
    return float(starters[points_col].sum() + starters[points_col].max())


def current_team_section(squad: pd.DataFrame, rules: dict) -> list[str]:
    """The team you hold RIGHT NOW, before any transfer — the do-nothing baseline.

    Worth printing even when a transfer is recommended: it is what you field if you decide to
    roll instead, and it is the number the gain below is measured against.
    """
    from ..backtest.season_sim import pick_xi

    if squad.empty:
        return []
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    ranked, starters = pick_xi(squad, rules, "expected_points")
    bench = ranked.drop(index=starters.index)
    starters = starters.assign(_o=lambda d: d["position"].map(order)).sort_values(
        ["_o", "expected_points"], ascending=[True, False]
    )
    return [
        "",
        "## Your team as it stands",
        (f"No transfer made. Expected **{xi_value(squad, rules):.1f}** points this "
         "gameweek, captain included."),
        "```",
        _fmt(starters, DISPLAY, rows=11),
        "```",
        "*bench*",
        "```",
        _fmt(bench.sort_values("expected_points", ascending=False), DISPLAY, rows=4),
        "```",
    ]


def move_section(before: float, after: float, plan, rules: dict, entering, leaving) -> list[str]:
    """What the recommended move is actually worth, split into its two honest halves.

    They answer different questions and are not comparable. The gameweek figure is what you
    gain THIS week from a better eleven. The horizon figure is what the optimiser chose on —
    six decayed gameweeks — and is the one that justified paying any hit.
    """
    if plan is None or not getattr(plan, "n_transfers", 0):
        return []
    hit = getattr(plan, "hit_cost", 0.0)
    lines = [
        "",
        "## What the move buys",
        "```",
        f"this gameweek      {before:6.2f}  ->  {after:6.2f}   {after - before:+.2f}",
        f"over the horizon                        {plan.gain:+.2f}"
        + (f"   net {plan.net_gain:+.2f} after a {hit:.0f}pt hit" if hit else "   (no hit)"),
        "```",
    ]
    if entering or leaving:
        lines.append(
            f"Changes the XI: **{', '.join(sorted(entering)) or 'nobody'}** in, "
            f"**{', '.join(sorted(leaving)) or 'nobody'}** out."
        )
    lines.append(
        "The horizon margin behind a transfer is measurably overstated — a fit of realised on "
        "forecast gain across 111 decisions gives a slope of 0.436, stable in every season. "
        "Treat a small positive as closer to zero than it reads."
    )
    return lines


def build_report(
    forecasts: pd.DataFrame,
    solution,
    gw: int,
    *,
    ownership: pd.DataFrame | None = None,
    plan=None,
    chip: str | None = None,
    chip_reason: str = "",
    chip_values: dict | None = None,
    risers: pd.DataFrame | None = None,
    fallers: pd.DataFrame | None = None,
    current_squad: pd.DataFrame | None = None,
    rules: dict | None = None,
    move: tuple | None = None,
) -> str:
    """Assemble the brief as markdown.

    `current_squad` and `move` are supplied by `myteam`, where a transfer plan exists and the
    XI below is the one you would field AFTER taking it. `report` solves an ideal fifteen from
    scratch, has no "before", and passes neither.
    """
    squad_ids = set(solution.squad["player_id"])
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    xi = solution.starting_xi.assign(_o=lambda d: d["position"].map(order)).sort_values(
        ["_o", "expected_points"], ascending=[True, False]
    )

    heading = "Starting XI" + (" — after the transfer" if move else "")
    header = [
        f"# Gameweek {gw} brief",
        "",
        (f"**Squad cost** £{solution.total_cost:.1f}m  ·  "
         f"**expected points** {solution.expected_points:.1f}  ·  "
         f"**captain** {solution.captain.get('web_name', '?')} "
         f"(vice {solution.vice_captain.get('web_name', '?')})"),
    ]
    team_sheet = [
        "",
        f"## {heading}",
        "```",
        _fmt(xi, DISPLAY, rows=11),
        "```",
        "",
        "## Bench",
        "```",
        _fmt(solution.bench, DISPLAY, rows=4),
        "```",
        "",
        "## Captaincy",
        "`cost_vs_best` is what you give up by overriding the recommendation.",
        "```",
        _fmt(
            captain_options(forecasts, squad_ids),
            ["web_name", "team", "expected_points", "captain_points", "cost_vs_best"],
        ),
        "```",
    ]

    # `myteam` reads as a decision in the order you actually make it: what you hold, what to
    # change, what that buys, then the team sheet to field. `report` has no "before" and keeps
    # the plain ordering.
    if current_squad is not None and rules is not None:
        lines = header + current_team_section(current_squad, rules)
        lines += transfer_section(plan)
        if move is not None:
            lines += move_section(*move[:2], plan, rules, *move[2:])
        lines += team_sheet
    else:
        lines = header + team_sheet + transfer_section(plan)
    lines += chip_section(chip, chip_reason, chip_values)

    risky = bench_risk(forecasts, squad_ids)
    if not risky.empty:
        lines += [
            "",
            "## Availability risk",
            "Squad members the minutes model rates below 70% to appear.",
            "```",
            _fmt(risky, ["web_name", "position", "team", "p_appear", "expected_minutes"]),
            "```",
        ]

    if ownership is not None and not ownership.empty:
        merged = forecasts.merge(ownership, on="player_id", how="left")
        from ..optimise.risk import differential_score

        merged["differential"] = differential_score(
            merged["expected_points"], merged.get("eo", pd.Series(0, index=merged.index))
        )
        lines += [
            "",
            "## Differentials",
            "Highest expected points relative to how much of the field already owns them.",
            "```",
            _fmt(
                merged.nlargest(8, "differential"),
                ["web_name", "position", "team", "expected_points", "eo", "differential"],
            ),
            "```",
        ]

    lines += price_section(risers, fallers)

    lines += [
        "",
        "## Caveats",
        f"- {bonus_confidence_note()}",
        ("- Defensive contributions are modelled on a single season (2025-26) with Poisson "
         "counts. They are measurably overdispersed, but a negative binomial improved "
         "defenders and worsened midfielders and forwards, so it was not applied."),
        "- The model reads no press conferences. Late team news is yours to apply.",
    ]
    return "\n".join(lines)
