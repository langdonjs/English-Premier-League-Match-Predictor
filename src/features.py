"""Leakage-free, pre-match feature engineering.

Every feature for a given match is computed using ONLY matches that kicked off
before it. Concretely we rely on ``shift(1)`` before every rolling/expanding
window so the current match's own result can never enter its own features. The
test suite (tests/test_features.py) enforces this: perturbing a match's score
must not change that match's feature row.

Design choices (documented so the methodology is honest):
  * form5_*        : rolling over a team's last 5 matches, spanning seasons.
                     Only the first few matches in the whole dataset are cold.
  * std_ppg        : points-per-game season-to-date; RESETS each season.
  * *_winrate      : venue-specific win rate season-to-date; RESETS each season.
  * h2h_home_winrate: home team's win rate in all prior meetings vs this
                      opponent, spanning seasons.
Cold-start rows (a team/season/matchup with no prior history) are left as NaN
here and imputed downstream inside each model's pipeline (median of the TRAIN
set only), so the split — not the feature builder — owns the imputed value.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

# Final model input columns, in a fixed order.
FEATURE_COLS = [
    "home_form5_ppg", "away_form5_ppg",
    "home_form5_gf", "away_form5_gf",
    "home_form5_ga", "away_form5_ga",
    "home_form5_gd", "away_form5_gd",
    "home_std_ppg", "away_std_ppg",
    "home_home_winrate", "away_away_winrate",
    "h2h_home_winrate",
    "diff_form5_ppg", "diff_form5_gd", "diff_std_ppg",
]

_FORM_WINDOW = 5


def _team_match_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Explode each match into two rows: one per team's perspective."""
    home = pd.DataFrame({
        "match_id": matches.index,
        "Date": matches["Date"].values,
        "Season": matches["Season"].values,
        "team": matches["HomeTeam"].values,
        "venue": "H",
        "gf": matches["FTHG"].values,
        "ga": matches["FTAG"].values,
    })
    away = pd.DataFrame({
        "match_id": matches.index,
        "Date": matches["Date"].values,
        "Season": matches["Season"].values,
        "team": matches["AwayTeam"].values,
        "venue": "A",
        "gf": matches["FTAG"].values,
        "ga": matches["FTHG"].values,
    })
    long = pd.concat([home, away], ignore_index=True)
    long["points"] = np.where(long.gf > long.ga, 3, np.where(long.gf == long.ga, 1, 0))
    long["win"] = (long.gf > long.ga).astype(int)
    # Chronological within each team so shift(1) means "the previous match".
    long = long.sort_values(["team", "Date", "match_id"]).reset_index(drop=True)
    return long


def _rolling_team_features(long: pd.DataFrame) -> pd.DataFrame:
    """Attach shifted rolling / expanding per-team features (no current match)."""
    by_team = long.groupby("team", sort=False)
    long["form5_ppg"] = by_team["points"].transform(
        lambda s: s.shift(1).rolling(_FORM_WINDOW, min_periods=1).mean())
    long["form5_gf"] = by_team["gf"].transform(
        lambda s: s.shift(1).rolling(_FORM_WINDOW, min_periods=1).mean())
    long["form5_ga"] = by_team["ga"].transform(
        lambda s: s.shift(1).rolling(_FORM_WINDOW, min_periods=1).mean())
    long["form5_gd"] = long["form5_gf"] - long["form5_ga"]
    # Number of matches this team has already played (0 for its first ever row).
    long["prior_games"] = by_team.cumcount()

    # Season-to-date points-per-game (resets each season).
    by_team_season = long.groupby(["team", "Season"], sort=False)
    long["std_ppg"] = by_team_season["points"].transform(
        lambda s: s.shift(1).expanding().mean())

    # Venue-specific win rate, season-to-date (resets each season).
    by_team_season_venue = long.groupby(["team", "Season", "venue"], sort=False)
    long["venue_winrate"] = by_team_season_venue["win"].transform(
        lambda s: s.shift(1).expanding().mean())
    return long


def _head_to_head(matches: pd.DataFrame) -> pd.Series:
    """Home team's win rate in prior meetings vs this opponent (spans seasons).

    Iterates in chronological order; each match is scored on history BEFORE it,
    then its own result is appended — so a match never sees itself.
    """
    history: dict[frozenset, list[str]] = defaultdict(list)
    values: dict = {}
    ordered = matches.sort_values(["Date"])
    for row in ordered.itertuples():
        home, away = row.HomeTeam, row.AwayTeam
        key = frozenset((home, away))
        prev = history[key]
        if prev:
            values[row.Index] = sum(1 for w in prev if w == home) / len(prev)
        else:
            values[row.Index] = np.nan
        if row.FTR == "H":
            winner = home
        elif row.FTR == "A":
            winner = away
        else:
            winner = "draw"
        history[key].append(winner)
    return pd.Series(values, name="h2h_home_winrate")


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Return ``matches`` with all engineered feature columns appended.

    Input must be the frame produced by ``data.load_seasons`` (has Date, Season,
    HomeTeam, AwayTeam, FTHG, FTAG, FTR, target). Output preserves the input
    index and row order.
    """
    long = _team_match_long(matches)
    long = _rolling_team_features(long)

    home_rows = long[long.venue == "H"].set_index("match_id")
    away_rows = long[long.venue == "A"].set_index("match_id")

    feat = matches.copy()
    feat["home_form5_ppg"] = home_rows["form5_ppg"]
    feat["away_form5_ppg"] = away_rows["form5_ppg"]
    feat["home_form5_gf"] = home_rows["form5_gf"]
    feat["away_form5_gf"] = away_rows["form5_gf"]
    feat["home_form5_ga"] = home_rows["form5_ga"]
    feat["away_form5_ga"] = away_rows["form5_ga"]
    feat["home_form5_gd"] = home_rows["form5_gd"]
    feat["away_form5_gd"] = away_rows["form5_gd"]
    feat["home_std_ppg"] = home_rows["std_ppg"]
    feat["away_std_ppg"] = away_rows["std_ppg"]
    feat["home_home_winrate"] = home_rows["venue_winrate"]
    feat["away_away_winrate"] = away_rows["venue_winrate"]
    feat["home_prior_games"] = home_rows["prior_games"]
    feat["away_prior_games"] = away_rows["prior_games"]

    feat["h2h_home_winrate"] = _head_to_head(matches)

    feat["diff_form5_ppg"] = feat["home_form5_ppg"] - feat["away_form5_ppg"]
    feat["diff_form5_gd"] = feat["home_form5_gd"] - feat["away_form5_gd"]
    feat["diff_std_ppg"] = feat["home_std_ppg"] - feat["away_std_ppg"]
    return feat
