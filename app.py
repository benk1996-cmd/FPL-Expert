"""Streamlit front end. Reads a precomputed bundle for everything about the GAME.

    fpl publish --gw N     # build data/serving/ (tens of seconds, weekly)
    streamlit run app.py   # read it (instant, per page view)

The separation is not tidiness. `forecast_gameweek` loads the 185,000-row archive, rebuilds
minutes features over all of it and refits Dixon-Coles on every call, and `fpl squad` repeats
that once per horizon week. Doing it per page view would be tens of seconds and hundreds of
megabytes; the bundle it produces is 117KB.

**The My Team tab is the one exception, and it is opt-in.** Advice about a squad someone
actually owns cannot come from the bundle, because the bundle is committed and deployed and
must therefore describe the game and never a user. So that tab runs the real pipeline — but
only when a button is pressed, cached on the entry id, and importing `fpl_expert.advice`
lazily inside the call. Every other tab still reads the bundle and this file still imports
nothing heavy at module scope, so a deployment without lightgbm, pulp or the archive serves
the whole app and fails only inside that one tab, with an explanation.

That tab reports on ONE entry, `ENTRY`, and offers no way to ask about another. It is not a
lookup tool for other people's teams.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from fpl_expert.serving import read_bundle

# Overridable so the deployment can point elsewhere and the tests can point at a fixture,
# rather than either depending on whatever `fpl publish` last wrote to the working copy.
BUNDLE = Path(
    os.environ.get("FPL_SERVING_DIR", Path(__file__).parent / "data" / "serving")
)
POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


def _entry_from_env() -> int | None:
    """`FPL_ENTRY`, from the environment or a local `.env`, or None if unset.

    Parsed here rather than with `python-dotenv` so the front end keeps its three-package
    dependency list — it reads a precomputed bundle and must install without the modelling
    stack. The CLI uses `fpl_expert.config.entry_id`, which does use dotenv; both read the
    same variable from the same file, so there is still one place to change the id.
    """
    raw = os.environ.get("FPL_ENTRY", "").strip()
    if not raw:
        env_file = Path(__file__).parent / ".env"
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("=")
                if key.strip() == "FPL_ENTRY" and not line.lstrip().startswith("#"):
                    raw = value.strip().strip("\"'")
                    break
        except OSError:
            return None
    return int(raw) if raw.isdigit() else None


# The one entry this app reports on. Previously a text box that accepted any id, with the last
# one remembered on disk; both are gone. A single owner needs no picker, and an arbitrary-entry
# box on a public deployment invites looking up other people's squads through a page that
# carries this project's recommendations — which would read as advice about them.
ENTRY = _entry_from_env()

st.set_page_config(page_title="FPL Expert", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner=False)
def load(path: str, stamp: float):
    """Cached on the manifest's mtime, so republishing refreshes without a restart."""
    return read_bundle(path)


# What `forecast_gameweek` reads that the deployed app does not have. Every one of these is
# gitignored on purpose — excluding them is what lets the bundle deploy without the archive —
# so My Team runs from a full local checkout and nowhere else. Checked by path rather than by
# import: on Streamlit Cloud the packages install fine and it is the DATA that is absent, which
# is why an ImportError guard alone let a FileNotFoundError traceback reach the page.
MODEL_INPUTS = {
    "the current season's teams and fixtures": "data/interim",
    "the trained minutes model": "data/processed/models/minutes.txt",
    "historical odds for the match model": "data/external",
    "a pre-deadline snapshot": "data/raw/snapshot",
}


def _populated(path: Path) -> bool:
    """Present and non-empty. An empty directory is as useless as an absent one here."""
    try:
        if not path.exists():
            return False
        return any(path.iterdir()) if path.is_dir() else path.stat().st_size > 0
    except OSError:
        return False


def missing_model_inputs() -> list[str]:
    """Which inputs are absent here. Empty means the model can run.

    Deliberately does not call `config.path()`, which creates the directory it resolves and
    would therefore report every layer as present.
    """
    root = Path(__file__).parent
    return [name for name, rel in MODEL_INPUTS.items() if not _populated(root / rel)]


@st.cache_data(show_spinner=False, ttl=3600)
def analyse(entry: int, gw: int | None):
    """Run the real pipeline for one entry. Cached, because it is ~20 seconds of work.

    Returns plain picklable pieces rather than the `EntryAdvice` dataclass so the cache never
    has to reason about a `TransferPlan`. The heavy imports live in here so that a deployment
    without the model still serves every other tab.
    """
    from fpl_expert.advice import analyse_entry
    from fpl_expert.cli import myteam_brief
    from fpl_expert.config import load_scoring_rules

    result = analyse_entry(entry, gw=gw)
    plan = result.plan
    return {
        "team_name": result.team_name,
        "manager_name": result.manager_name,
        "gameweek": result.gameweek,
        "span": result.span,
        "bank": result.bank,
        "free_transfers": result.free_transfers,
        "gw_points": result.gw1_points,
        "overall_rank": result.overall_rank,
        "squad": result.held,
        "summary": plan.summary(),
        "n_transfers": plan.n_transfers,
        "transfers_out": plan.transfers_out,
        "transfers_in": plan.transfers_in,
        "brief": myteam_brief(result, load_scoring_rules()),   # path=None: writes nothing
    }


def players_for(players, forecasts, gw: int, published_gw: int):
    """The player table as it stands in gameweek `gw`.

    Identity, price and squad membership come from `players.parquet` and do not vary: the
    fifteen were chosen ONCE, on the horizon valuation made at the published gameweek. Only
    the per-week quantities are swapped. Stepping forward therefore answers "how does the
    squad I picked look next week", not "what would I pick next week" — a different question
    the bundle cannot answer, because re-solving needs the model.
    """
    if forecasts is None or gw == published_gw:
        return players
    block = forecasts[forecasts["gw"] == gw].drop(columns=["gw"])
    if block.empty:
        return players
    static = [c for c in players.columns if c not in block.columns or c == "player_id"]
    return players[static].merge(block, on="player_id", how="inner")


def freshness(built_at: str) -> tuple[str, str]:
    """How old the advice is. A stale page must look stale rather than merely be stale."""
    age = datetime.now(UTC) - datetime.fromisoformat(built_at)
    hours = age.total_seconds() / 3600
    if hours < 24:
        return f"{hours:.0f}h old", "normal"
    return f"{hours / 24:.0f} days old", "inverse"


try:
    manifest_file = BUNDLE / "manifest.json"
    bundle = load(str(BUNDLE), manifest_file.stat().st_mtime)
except FileNotFoundError:
    st.error("No serving bundle found.")
    st.code("fpl publish --gw 1", language="bash")
    st.caption(
        "The app deliberately does not run the model — it reads what `publish` wrote. "
        "Forecasting loads the whole archive and refits the match model, which is far too "
        "slow to do per page view."
    )
    st.stop()

manifest = bundle["manifest"]
all_players = bundle["players"]
age, tone = freshness(manifest["built_at"])

# A bundle CAN carry alternative views of the same gameweek, and the switch below renders only
# when it does. Nothing currently publishes more than one: a minutes-budget view was offered
# here and withdrawn, because it was worse on points in every season tested even though it
# improved minutes. Offering a choice implies the evidence is balanced, and it was not.
variants = list(manifest.get("variants") or {"standard": manifest})
if "variant" in all_players.columns:
    variants = [v for v in variants if v in set(all_players["variant"])] or variants
    chosen = st.sidebar.radio("Prediction", variants, index=0) if len(variants) > 1 else variants[0]
    players = all_players[all_players["variant"] == chosen]
else:
    chosen = "standard"                       # a bundle published before variants existed
    players = all_players
summary = (manifest.get("variants") or {}).get(chosen, manifest)

# --- which gameweek of the horizon is on screen
published_gw = int(manifest["gameweek"])
served_gws = [int(g) for g in manifest.get("gameweeks") or [published_gw]]
if published_gw not in served_gws:                # a bundle published before forecasts.parquet
    served_gws = sorted({published_gw, *served_gws})

view_gw = int(st.session_state.get("view_gw", published_gw))
if view_gw not in served_gws:                     # republished over a different window
    view_gw = published_gw
index = served_gws.index(view_gw)

st.title("FPL Expert")

if len(served_gws) > 1:
    back, label, forward, spacer = st.columns([1, 2, 1, 8])
    if back.button("◀", disabled=index == 0, help="Previous gameweek",
                   use_container_width=True):
        st.session_state["view_gw"] = served_gws[index - 1]
        st.rerun()
    label.markdown(
        f"<div style='text-align:center;padding-top:0.35rem'><b>GW{view_gw}</b>"
        f"<br><span style='font-size:0.75rem;opacity:0.6'>{index + 1} of "
        f"{len(served_gws)}</span></div>",
        unsafe_allow_html=True,
    )
    if forward.button("▶", disabled=index == len(served_gws) - 1, help="Next gameweek",
                      use_container_width=True):
        st.session_state["view_gw"] = served_gws[index + 1]
        st.rerun()
    if view_gw != published_gw:
        spacer.warning(
            f"Forecasts for GW{view_gw}. The squad, captain and brief below are the "
            f"GW{published_gw} decision — the fifteen were chosen once, on the whole horizon.",
            icon="⏭",
        )

players = players_for(players, bundle.get("forecasts"), view_gw, published_gw)

top = st.columns(5)
top[0].metric(
    "Gameweek", view_gw,
    delta=(None if view_gw == published_gw else f"+{view_gw - published_gw} ahead"),
    delta_color="off",
)
if view_gw == published_gw:
    xi_points = float(summary["expected_points"])
else:
    # Recomputed rather than read from the manifest, which only ever describes the published
    # week. The armband is counted twice because the XI total includes it.
    xi = players[players["in_squad"] & players["is_starter"]]
    xi_points = float(
        xi["expected_points"].sum()
        + xi.loc[xi["is_captain"], "expected_points"].sum()
    )

top[1].metric(
    "Expected points", f"{xi_points:.1f}",
    delta=(round(xi_points - float(summary["expected_points"]), 2)
           if view_gw != published_gw
           else (None if chosen == "standard"
                 else round(summary["expected_points"] - manifest["expected_points"], 2))),
    help="This gameweek's XI plus the armband — not the multi-week objective the squad was "
         "chosen on. Stepping forward compares that week against the published one.",
)
top[2].metric("Squad cost", f"£{summary['squad_cost']:.1f}m")
top[3].metric("Captain", summary["captain"] or "—")
top[4].metric("Built", age, delta_color=tone)

st.caption(
    f"**{chosen}** view · valued over {manifest['horizon']} gameweeks · built "
    f"{manifest['built_at'].replace('T', ' ')} · {manifest['players']} players"
    + (f" · showing GW{view_gw} of GW{served_gws[0]}–{served_gws[-1]}"
       if len(served_gws) > 1 else "")
    + ". Advice is only as current as the bundle — republish after each deadline."
)

squad_tab, players_tab, fixtures_tab, brief_tab, myteam_tab = st.tabs(
    ["Squad", "Players", "Fixtures", "Brief", "My Team"]
)

with squad_tab:
    squad = players[players["in_squad"]].copy()
    squad["role"] = squad.apply(
        lambda r: "C" if r["is_captain"] else ("V" if r["is_vice"] else ""), axis=1
    )
    squad["_o"] = squad["position"].map(POSITION_ORDER)
    shown = ["role", "web_name", "position", "team", "price",
             "expected_points", "horizon_points", "p_long"]
    shown = [c for c in shown if c in squad.columns]

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Starting XI")
        st.dataframe(
            squad[squad["is_starter"]].sort_values(["_o", "expected_points"],
                                                   ascending=[True, False])[shown],
            hide_index=True, use_container_width=True,
        )
        st.subheader("Bench")
        st.dataframe(
            squad[~squad["is_starter"]].sort_values("expected_points", ascending=False)[shown],
            hide_index=True, use_container_width=True,
        )
    with right:
        st.subheader("Where the points come from")
        parts = [c for c in players.columns if c.startswith("pts_")]
        if parts:
            xi = squad[squad["is_starter"]]
            breakdown = (
                xi[parts].sum().rename("points").rename_axis("component").reset_index()
            )
            breakdown["component"] = breakdown["component"].str.replace("pts_", "", regex=False)
            st.bar_chart(
                breakdown.set_index("component").sort_values("points", ascending=False)
            )
            st.caption(
                "Attacking returns dominate — the only ablation component whose sign holds "
                "in every backtested season."
            )

with players_tab:
    st.subheader("All players")
    filters = st.columns(4)
    positions = filters[0].multiselect(
        "Position", sorted(players["position"].dropna().unique()), default=[]
    )
    teams = filters[1].multiselect(
        "Team", sorted(players["team"].dropna().unique()), default=[]
    )
    max_price = filters[2].slider(
        "Max price", float(players["price"].min()), float(players["price"].max()),
        float(players["price"].max()), step=0.5,
    )
    min_minutes = filters[3].slider("Min P(60+ mins)", 0.0, 1.0, 0.0, step=0.05)

    view = players.copy()
    if positions:
        view = view[view["position"].isin(positions)]
    if teams:
        view = view[view["team"].isin(teams)]
    view = view[view["price"] <= max_price]
    if "p_long" in view.columns:
        view = view[view["p_long"] >= min_minutes]

    sort_col = "horizon_points" if "horizon_points" in view.columns else "expected_points"
    columns = [c for c in ["web_name", "position", "team", "price", "expected_points",
                           "horizon_points", "p_long", "expected_minutes",
                           "expected_goals", "expected_assists"] if c in view.columns]
    st.dataframe(
        view.sort_values(sort_col, ascending=False)[columns].head(200),
        hide_index=True, use_container_width=True,
    )
    st.caption(f"{len(view)} players match. Showing the top 200 by {sort_col}.")

with fixtures_tab:
    grid = bundle.get("fixtures")
    if grid is None or grid.empty:
        st.info("No fixture data in this bundle.")
    else:
        st.subheader(
            f"GW{manifest['gameweek']}–{manifest['gameweek'] + manifest['horizon'] - 1}"
        )
        grid = grid.copy()
        grid["label"] = grid["opponent"] + grid["is_home"].map({True: " (H)", False: " (A)"})
        # `aggfunc=", ".join` rather than `first`: a double gameweek has two fixtures in one
        # cell and dropping one would hide exactly the weeks worth planning around.
        wide = grid.pivot_table(
            index="team", columns="gw", values="label", aggfunc=", ".join
        ).fillna("—")
        wide.columns = [f"GW{c}" for c in wide.columns]
        st.dataframe(wide, use_container_width=True)
        st.caption("— means a blank gameweek. Two entries in a cell is a double.")

with brief_tab:
    if view_gw != published_gw:
        st.caption(
            f"The brief is written once per publish and describes GW{published_gw}. "
            f"Stepping to GW{view_gw} changes the forecasts on the other tabs, not this."
        )
    st.markdown(bundle["brief"] or "_No brief in this bundle._")

with myteam_tab:
    st.subheader("Your squad")
    st.caption(
        "Everything else on this page is the ideal squad. This tab starts from the 15 you "
        "actually own, which makes it a marginal question — never *what is the best squad?* "
        "but *is this specific change worth what it costs?*"
    )

    absent = missing_model_inputs()
    if ENTRY is None:
        st.warning("No entry id configured, so there is no squad to report on.", icon="🔧")
        st.markdown(
            "Set `FPL_ENTRY` to the number in your team URL — either in the environment, or "
            "in a `.env` file at the repository root (copy `.env.example`). The `fpl myteam` "
            "command reads the same variable."
        )
        st.code("FPL_ENTRY=1234567", language="bash")
    elif absent:
        st.warning(
            "This tab cannot run here — it is the one part of the app that needs the model, "
            "and this deployment carries only the published bundle.",
            icon="🔒",
        )
        st.caption("Missing: " + "; ".join(absent) + ".")
        st.markdown(
            "Every other tab works because `fpl publish` precomputed it. Advice about a "
            "squad you own cannot be precomputed for everyone, so it needs the archive, the "
            "trained minutes model and a pre-deadline snapshot — all deliberately excluded "
            "from the repository, which is what keeps this deployment small. Run it from a "
            "full local checkout instead:"
        )
        st.code("fpl myteam --entry 1234567 --brief myteam.md", language="bash")
    else:
        controls = st.columns([1, 4])
        if controls[0].button("Analyse", type="primary", use_container_width=True):
            st.session_state["myteam_entry"] = ENTRY
        controls[1].caption(f"Entry **{ENTRY}**.")

        active = st.session_state.get("myteam_entry")
        if active is None:
            st.info(f"Press Analyse to work out this week's move for entry {ENTRY}.")
            st.caption(
                "Behind a button because this is the one tab that runs the model rather than "
                "reading the published bundle — advice about a squad you own cannot be "
                "precomputed. Expect roughly 20 seconds the first time; then it is cached."
            )
        else:
            try:
                with st.spinner(f"Forecasting {manifest['horizon']} gameweeks for entry {active}…"):
                    me = analyse(active, None)
            except (ImportError, FileNotFoundError) as exc:
                # The guard above catches this before anyone presses the button. Kept as a
                # backstop because it is the failure a deployment actually hits — the packages
                # install fine and the DATA is absent — and an ImportError-only guard let a
                # bare traceback reach the page.
                st.error("This deployment cannot run the model, only read the published bundle.")
                st.caption(
                    f"Missing: {exc}. The My Team tab needs the archive, the trained minutes "
                    "model and a pre-deadline snapshot; run it from a full local checkout."
                )
            except Exception as exc:                # noqa: BLE001 - surface, never blank the tab
                message = str(exc)
                if "snapshot" in message.lower():
                    # The strict point-in-time accessor, doing its job. Do not loosen it: take
                    # the missing capture instead, which is only possible before the deadline.
                    st.error("No pre-deadline snapshot exists for this gameweek yet.")
                    st.code("fpl snapshot", language="bash")
                    st.caption(
                        "The target gameweek is read through the strict point-in-time accessor "
                        "on purpose. Capture the state before the deadline and re-run — after "
                        "it passes, that state is unrecoverable."
                    )
                elif "not found" in message.lower():
                    st.error(
                        f"The FPL API has no entry {active}. Set FPL_ENTRY if this app has "
                        "changed hands."
                    )
                else:
                    st.error(f"Could not analyse entry {active}.")
                    st.exception(exc)
            else:
                head = st.columns(5)
                head[0].metric("Team", me["team_name"])
                head[1].metric("Gameweek", me["gameweek"])
                head[2].metric("In the bank", f"£{me['bank']:.1f}m")
                head[3].metric("Free transfers", me["free_transfers"])
                head[4].metric(
                    "Overall rank",
                    f"{me['overall_rank']:,}" if me["overall_rank"] else "—",
                    help="Last published rank. Blank until the first gameweek is scored.",
                )

                st.markdown(f"**{me['summary']}**")
                if me["n_transfers"]:
                    out_cols = ["web_name", "position", "team", "selling_price", "horizon_points"]
                    in_cols = ["web_name", "position", "team", "price", "horizon_points"]
                    left, right = st.columns(2)
                    with left:
                        st.caption("OUT")
                        st.dataframe(
                            me["transfers_out"][
                                [c for c in out_cols if c in me["transfers_out"].columns]
                            ].round(2), hide_index=True, use_container_width=True,
                        )
                    with right:
                        st.caption("IN")
                        st.dataframe(
                            me["transfers_in"][
                                [c for c in in_cols if c in me["transfers_in"].columns]
                            ].round(2), hide_index=True, use_container_width=True,
                        )
                st.caption(
                    f"Judged on the discounted {me['span']}-week horizon, not on GW"
                    f"{me['gameweek']} alone — a transfer is a durable change, and judging one on "
                    "a single week systematically over-trades. A move beyond your free transfers "
                    "must also clear 4 points."
                )

                mine = me["squad"].copy()
                mine["_o"] = mine["position"].map(POSITION_ORDER)
                squad_cols = [c for c in ["web_name", "position", "team", "selling_price",
                                          "expected_points", "horizon_points", "p_long"]
                              if c in mine.columns]
                st.subheader("The 15 you hold")
                st.dataframe(
                    mine.sort_values(["_o", "expected_points"], ascending=[True, False])[squad_cols],
                    hide_index=True, use_container_width=True,
                )

                with st.expander("Full brief — captaincy, chips, price moves, caveats"):
                    st.markdown(me["brief"])
                st.caption(
                    "Personal to this entry, so it is rendered and never written into "
                    "`data/serving/`, which is committed. Use `fpl myteam --brief PATH` for a file."
                )

prices = bundle.get("prices")
if prices is not None and not prices.empty:
    with st.sidebar:
        st.subheader("Price moves tonight")
        for direction, label in (("rise", "Likely risers"), ("fall", "Your likely fallers")):
            block = prices[prices["direction"] == direction]
            if not block.empty:
                st.caption(label)
                st.dataframe(
                    block[[c for c in ["web_name", "price", "expected_change"]
                           if c in block.columns]],
                    hide_index=True, use_container_width=True,
                )

with st.sidebar:
    if len(variants) > 1:
        st.subheader("Views")
        st.caption(
            f"This bundle carries {len(variants)} forecasts of the same gameweek. They differ "
            "in how minutes or points are modelled, not in the outcome being predicted."
        )
    st.subheader("How to read this")
    st.caption(
        "**expected_points** is this gameweek. **horizon_points** is the discounted "
        f"{manifest['horizon']}-week valuation the squad was chosen on, including the "
        "captaincy premium — the two are not comparable and the horizon one is not a "
        "prediction of anything."
    )
    st.caption(
        "The horizon is an unproven default: measured against a myopic policy it is worth "
        "+11 ± 28 points a season. It is kept because switching it off would change 8–10 of "
        "the 15 players on evidence that cannot separate the options."
    )
    st.caption(
        "Backtested at +399 [+355, +443] points a season against a recent-form baseline, "
        "holding in every season tested."
    )
