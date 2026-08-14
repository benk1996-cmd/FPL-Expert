"""Canonical team names, and mappings from external sources onto them.

FPL's short names are the canonical form, because the FPL feed is the one thing every model
must join back to. External sources disagree in small ways ("Man United" vs "Man Utd",
"Tottenham" vs "Spurs") and a silent mismatch is expensive: the fixture simply loses its
odds and the match model quietly falls back to a prior for that game. So the lookup is
strict by default and raises on anything it does not recognise.
"""

from __future__ import annotations

import pandas as pd

# football-data.co.uk -> FPL short name. Only genuine differences are listed; anything
# absent is assumed identical and validated by `normalise_series`.
FOOTBALL_DATA_TO_FPL = {
    "Man United": "Man Utd",
    "Tottenham": "Spurs",
    "Ipswich": "Ipswich Town",
    "Hull": "Hull City",
    "Coventry": "Coventry City",
    "Sheffield United": "Sheffield Utd",
    "Nott'm Forest": "Nott'm Forest",
}

# Names confirmed present in football-data E0 files for 2019-20..2025-26, plus the clubs
# promoted for 2026/27. Used to catch typos and unexpected new spellings early.
KNOWN_FOOTBALL_DATA_TEAMS = {
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Burnley", "Chelsea",
    "Coventry", "Crystal Palace", "Everton", "Fulham", "Hull", "Ipswich", "Leeds", "Leicester",
    "Liverpool", "Luton", "Man City", "Man United", "Newcastle", "Norwich", "Nott'm Forest",
    "Sheffield United", "Southampton", "Sunderland", "Tottenham", "Watford", "West Brom",
    "West Ham", "Wolves",
}


# The historical archive's team names drift between seasons and do not always match the
# current FPL short names. Left unmapped, a club looks like a newly promoted side with no
# history at all: "Ipswich" (2024-25 archive) vs "Ipswich Town" (2026/27 FPL) made a club
# with a full Premier League season look completely cold.
ARCHIVE_TO_FPL = {
    "Ipswich": "Ipswich Town",
    "Hull": "Hull City",
    "Coventry": "Coventry City",
}

KNOWN_ARCHIVE_TEAMS = {
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Burnley", "Chelsea",
    "Crystal Palace", "Everton", "Fulham", "Ipswich", "Leeds", "Leicester", "Liverpool",
    "Luton", "Man City", "Man Utd", "Newcastle", "Norwich", "Nott'm Forest", "Sheffield Utd",
    "Southampton", "Spurs", "Sunderland", "Watford", "West Brom", "West Ham", "Wolves",
}


class UnknownTeamError(KeyError):
    """Raised when a source uses a team name we have no mapping for."""


def normalise_team(name: str, mapping: dict[str, str] | None = None, *, strict: bool = True) -> str:
    """Map one external team name onto its FPL short name."""
    mapping = FOOTBALL_DATA_TO_FPL if mapping is None else mapping
    if name in mapping:
        return mapping[name]
    if strict and name not in KNOWN_FOOTBALL_DATA_TEAMS:
        raise UnknownTeamError(
            f"unrecognised team name {name!r}. Add it to FOOTBALL_DATA_TO_FPL in "
            f"fpl_expert/data/teams.py — do not let it through, or this fixture will "
            f"silently lose its odds."
        )
    return name


def normalise_series(
    names: pd.Series, mapping: dict[str, str] | None = None, *, strict: bool = True
) -> pd.Series:
    """Vectorised `normalise_team`, reporting every unknown name at once."""
    mapping = FOOTBALL_DATA_TO_FPL if mapping is None else mapping
    known = KNOWN_ARCHIVE_TEAMS if mapping is ARCHIVE_TO_FPL else KNOWN_FOOTBALL_DATA_TEAMS
    if strict:
        unknown = sorted(set(names.dropna()) - set(mapping) - known)
        if unknown:
            raise UnknownTeamError(
                f"unrecognised team names {unknown}. Add them to the relevant mapping in "
                f"fpl_expert/data/teams.py."
            )
    return names.map(lambda n: mapping.get(n, n) if pd.notna(n) else n)


def normalise_archive_teams(names: pd.Series, *, strict: bool = True) -> pd.Series:
    """Map historical-archive team names onto current FPL short names."""
    return normalise_series(names, ARCHIVE_TO_FPL, strict=strict)
