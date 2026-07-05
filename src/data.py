"""Load and normalise football-data.co.uk Premier League season CSVs.

Each raw CSV is a football-data.co.uk "E0" export with ~60-106 columns. We keep
only the columns the pipeline needs, parse dates, tag each row with its season,
concatenate, and encode the full-time result into an integer label.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

# Full-time result -> integer label. Ordered so the label index lines up with
# CLASS_NAMES below (used everywhere for reports and confusion matrices).
RESULT_MAP = {"H": 0, "D": 1, "A": 2}
CLASS_NAMES = ["Home Win", "Draw", "Away Win"]

# Columns kept from each raw CSV. Odds columns feed the bookmaker baseline only
# (never the models). B365 (Bet365) is present in every season we use.
CORE_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS_COLS = ["B365H", "B365D", "B365A"]


def _season_from_path(path: Path) -> str:
    """Infer a season label like '2021-2022' from the file stem."""
    return path.stem


def _parse_dates(raw: pd.Series) -> pd.Series:
    """Parse football-data dates (day-first: dd/mm/yyyy, sometimes dd/mm/yy)."""
    parsed = pd.to_datetime(raw, format="%d/%m/%Y", errors="coerce")
    # Fall back for any rows that used a 2-digit year.
    missing = parsed.isna()
    if missing.any():
        parsed[missing] = pd.to_datetime(raw[missing], dayfirst=True, errors="coerce")
    return parsed


def load_season(path: str | Path) -> pd.DataFrame:
    """Load one season CSV into a normalised frame."""
    path = Path(path)
    df = pd.read_csv(path)

    missing = [c for c in CORE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    keep = CORE_COLS + [c for c in ODDS_COLS if c in df.columns]
    df = df[keep].copy()
    df["Date"] = _parse_dates(df["Date"])
    df["Season"] = _season_from_path(path)

    # Drop rows we cannot use (unparseable date or unexpected result code).
    df = df[df["Date"].notna()]
    df = df[df["FTR"].isin(RESULT_MAP)]
    df["target"] = df["FTR"].map(RESULT_MAP).astype(int)
    return df


def load_seasons(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Load and concatenate multiple seasons, sorted chronologically.

    Sort key is (Date, Season) so ties on a date stay grouped by season, and the
    global order is a valid timeline for the leakage-free rolling features.
    """
    frames = [load_season(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["Date", "Season"]).reset_index(drop=True)
    return combined
