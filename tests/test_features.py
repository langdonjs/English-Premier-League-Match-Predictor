"""Tests for the feature engineering — chiefly that it does not leak.

The central guarantee: a match's engineered features depend only on matches
that happened before it. We enforce it by perturbing a match's score and
checking (a) that same match's features are unchanged, and (b) that a later
match involving the same team DOES change (proving the features use history at
all, so the invariance in (a) is meaningful rather than vacuous).
"""

from pathlib import Path

import numpy as np
import pytest

from src.data import load_seasons
from src.features import FEATURE_COLS, build_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEASONS = ["2018-2019", "2019-2020", "2020-2021", "2021-2022"]


@pytest.fixture(scope="module")
def matches():
    return load_seasons([DATA_DIR / f"{s}.csv" for s in SEASONS])


@pytest.fixture(scope="module")
def features(matches):
    return build_features(matches)


def test_all_feature_columns_present(features):
    for col in FEATURE_COLS:
        assert col in features.columns
    assert features[FEATURE_COLS].shape == (len(features), len(FEATURE_COLS))


def test_first_match_per_team_has_no_form(features):
    # A team's very first match ever has no prior games -> NaN form features.
    first_ever = features[features["home_prior_games"] == 0]
    assert first_ever["home_form5_ppg"].isna().all()


def _pick_established_match(features):
    """A match where both teams and the matchup already have history."""
    mask = (
        (features["home_prior_games"] >= 5)
        & (features["away_prior_games"] >= 5)
        & features["h2h_home_winrate"].notna()
    )
    candidates = features[mask].index
    assert len(candidates) > 0
    return candidates[len(candidates) // 2]


def test_features_do_not_use_own_result(matches, features):
    idx = _pick_established_match(features)

    perturbed = matches.copy()
    # Flip this match to a 5-0 home win, whatever it was before.
    perturbed.loc[idx, ["FTHG", "FTAG", "FTR", "target"]] = [5, 0, "H", 0]
    refeat = build_features(perturbed)

    before = features.loc[idx, FEATURE_COLS].to_numpy(dtype=float)
    after = refeat.loc[idx, FEATURE_COLS].to_numpy(dtype=float)
    assert np.allclose(before, after, equal_nan=True), (
        "A match's own features must not depend on its own result."
    )


def test_history_actually_used(matches, features):
    # Perturbing a match SHOULD change a later match involving the same team.
    idx = _pick_established_match(features)
    home_team = matches.loc[idx, "HomeTeam"]
    match_date = matches.loc[idx, "Date"]

    later = matches[
        (matches["Date"] > match_date)
        & ((matches["HomeTeam"] == home_team) | (matches["AwayTeam"] == home_team))
    ]
    assert len(later) > 0, "expected a later match for this team"
    later_idx = later.index[0]

    perturbed = matches.copy()
    perturbed.loc[idx, ["FTHG", "FTAG", "FTR", "target"]] = [5, 0, "H", 0]
    refeat = build_features(perturbed)

    before = features.loc[later_idx, FEATURE_COLS].to_numpy(dtype=float)
    after = refeat.loc[later_idx, FEATURE_COLS].to_numpy(dtype=float)
    assert not np.allclose(before, after, equal_nan=True), (
        "A later match should reflect the perturbed history."
    )
