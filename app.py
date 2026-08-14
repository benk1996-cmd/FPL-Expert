"""Streamlit front end. Reads a precomputed bundle; never runs the model.

    fpl publish --gw N     # build data/serving/ (tens of seconds, weekly)
    streamlit run app.py   # read it (instant, per page view)

The separation is not tidiness. `forecast_gameweek` loads the 185,000-row archive, rebuilds
minutes features over all of it and refits Dixon-Coles on every call, and `fpl squad` repeats
that once per horizon week. Doing it per page view would be tens of seconds and hundreds of
megabytes; the bundle it produces is 117KB.

It also means this file imports nothing from `fpl_expert` except the bundle reader, so the app
deploys without lightgbm, pulp or the archive.
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

st.set_page_config(page_title="FPL Expert", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner=False)
def load(path: str, stamp: float):
    """Cached on the manifest's mtime, so republishing refreshes without a restart."""
    return read_bundle(path)


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
players = bundle["players"]
age, tone = freshness(manifest["built_at"])

st.title("FPL Expert")
top = st.columns(5)
top[0].metric("Gameweek", manifest["gameweek"])
top[1].metric("Expected points", f"{manifest['expected_points']:.1f}", help="This gameweek's XI plus the armband — not the multi-week objective the squad was chosen on.")
top[2].metric("Squad cost", f"£{manifest['squad_cost']:.1f}m")
top[3].metric("Captain", manifest["captain"] or "—")
top[4].metric("Built", age, delta_color=tone)

st.caption(
    f"Valued over {manifest['horizon']} gameweeks · built "
    f"{manifest['built_at'].replace('T', ' ')} · {manifest['players']} players. "
    "Advice is only as current as the bundle — republish after each deadline."
)

squad_tab, players_tab, fixtures_tab, brief_tab = st.tabs(
    ["Squad", "Players", "Fixtures", "Brief"]
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
    st.markdown(bundle["brief"] or "_No brief in this bundle._")

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
