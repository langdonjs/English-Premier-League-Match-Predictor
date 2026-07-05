"""Experiment: how much do the betting odds help if we feed them to the models?

The main pipeline (main.py) keeps the betting odds OUT of the models and only
uses them for the bookmaker baseline. This script trains the same three models
twice, once on the engineered features alone and once with the Bet365 implied
probabilities added as features, and prints accuracy and weighted F1 for each.

Run from the repo root:  python odds_variant.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import load_seasons
from src.features import FEATURE_COLS, build_features

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_SEASONS = ["2018-2019", "2019-2020", "2020-2021"]
TEST_SEASON = "2021-2022"


def build_models() -> dict[str, Pipeline]:
    return {
        "Decision Tree": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=20, random_state=SEED)),
        ]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_leaf=10,
                n_jobs=-1, random_state=SEED)),
        ]),
        "MLP": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=1000,
                early_stopping=True, random_state=SEED)),
        ]),
    }


def main() -> None:
    np.random.seed(SEED)
    matches = load_seasons(
        [DATA_DIR / f"{s}.csv" for s in TRAIN_SEASONS + [TEST_SEASON]])
    feat = build_features(matches)

    # Bet365 odds -> normalised implied probabilities, added as features.
    implied = 1.0 / feat[["B365H", "B365D", "B365A"]].to_numpy(dtype=float)
    implied = implied / implied.sum(axis=1, keepdims=True)
    feat["odds_home"] = implied[:, 0]
    feat["odds_draw"] = implied[:, 1]
    feat["odds_away"] = implied[:, 2]
    cols_with_odds = FEATURE_COLS + ["odds_home", "odds_draw", "odds_away"]

    train = feat[feat["Season"].isin(TRAIN_SEASONS)]
    test = feat[feat["Season"] == TEST_SEASON]

    for label, cols in [
        ("Engineered features only", FEATURE_COLS),
        ("Engineered features + Bet365 odds", cols_with_odds),
    ]:
        X_train, y_train = train[cols], train["target"].to_numpy()
        X_test, y_test = test[cols], test["target"].to_numpy()
        print("=" * 60)
        print(label)
        print("=" * 60)
        for name, model in build_models().items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            wf1 = f1_score(y_test, pred, average="weighted")
            print(f"  {name:14s}  accuracy={acc:.3f}  weighted_F1={wf1:.3f}")
        print()


if __name__ == "__main__":
    main()
