"""EPL match-outcome prediction — training & evaluation pipeline.

Trains Decision Tree, Random Forest, and MLP classifiers on engineered,
leakage-free pre-match features and evaluates them on a held-out season against
three baselines (majority-class, uniform-random, and the bookmaker market).

Run from the repo root:  python main.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import CLASS_NAMES, load_seasons
from src.evaluate import (
    bookmaker_baseline,
    comparison_table,
    full_report,
    majority_and_random_baselines,
    metrics_row,
    save_confusion_matrix,
)
from src.features import FEATURE_COLS, build_features

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

TRAIN_SEASONS = ["2018-2019", "2019-2020", "2020-2021"]
TEST_SEASON = "2021-2022"


def build_models() -> dict[str, Pipeline]:
    """Three classifiers, each fronted by a median imputer (fit on train only).

    Cold-start feature rows arrive as NaN from the feature builder; the imputer
    fills them with the training median so the split owns the imputed value. The
    MLP additionally needs standardised inputs.
    """
    imputer = lambda: SimpleImputer(strategy="median")
    return {
        "Decision Tree": Pipeline([
            ("impute", imputer()),
            ("clf", DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=20, random_state=SEED)),
        ]),
        "Random Forest": Pipeline([
            ("impute", imputer()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_leaf=10,
                n_jobs=-1, random_state=SEED)),
        ]),
        "MLP": Pipeline([
            ("impute", imputer()),
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=1000,
                early_stopping=True, random_state=SEED)),
        ]),
    }


def class_distribution(y) -> str:
    counts = {name: int((y == i).sum()) for i, name in enumerate(CLASS_NAMES)}
    total = len(y)
    return "  ".join(f"{n}: {c} ({c / total:.1%})" for n, c in counts.items())


def main() -> None:
    np.random.seed(SEED)

    # ---- Load & engineer features on the full timeline, then split by season.
    train_paths = [DATA_DIR / f"{s}.csv" for s in TRAIN_SEASONS]
    test_path = DATA_DIR / f"{TEST_SEASON}.csv"
    matches = load_seasons(train_paths + [test_path])
    feat = build_features(matches)

    train = feat[feat["Season"].isin(TRAIN_SEASONS)]
    test = feat[feat["Season"] == TEST_SEASON]

    X_train, y_train = train[FEATURE_COLS], train["target"].to_numpy()
    X_test, y_test = test[FEATURE_COLS], test["target"].to_numpy()

    print("=" * 70)
    print("DATASET")
    print("=" * 70)
    print(f"Train seasons {TRAIN_SEASONS}: {len(train)} matches")
    print(f"   distribution: {class_distribution(y_train)}")
    print(f"Test season  {TEST_SEASON}: {len(test)} matches")
    print(f"   distribution: {class_distribution(y_test)}")
    print(f"Features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}")

    rows: dict[str, dict] = {}

    # ---- Baselines ------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("BASELINES (test season)")
    print("=" * 70)
    maj_pred, rand_pred = majority_and_random_baselines(
        X_train, y_train, X_test, seed=SEED)
    book_pred, book_mask = bookmaker_baseline(test)

    print("\n--- Majority-class baseline ---")
    print(full_report(y_test, maj_pred))
    rows["Baseline: Majority class"] = metrics_row(y_test, maj_pred)

    print("--- Uniform-random baseline ---")
    print(full_report(y_test, rand_pred))
    rows["Baseline: Random"] = metrics_row(y_test, rand_pred)

    n_book = int(book_mask.sum())
    print(f"--- Bookmaker (Bet365 implied) baseline [{n_book}/{len(test)} "
          "matches with odds] ---")
    print(full_report(y_test[book_mask], book_pred))
    rows["Baseline: Bookmaker (B365)"] = metrics_row(y_test[book_mask], book_pred)

    # ---- Models ---------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("MODELS (trained on prior seasons, evaluated on test season)")
    print("=" * 70)
    RESULTS_DIR.mkdir(exist_ok=True)
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f"\n--- {name} ---")
        print(full_report(y_test, pred))
        rows[name] = metrics_row(y_test, pred)
        out = RESULTS_DIR / f"confusion_{name.lower().replace(' ', '_')}.png"
        save_confusion_matrix(y_test, pred, f"{name} — {TEST_SEASON}", out)
        print(f"   confusion matrix saved: {out.relative_to(ROOT)}")

    # ---- Side-by-side summary ------------------------------------------- #
    print("\n" + "=" * 70)
    print("SUMMARY (higher is better; test season)")
    print("=" * 70)
    table = comparison_table(rows)
    print(table.to_string())
    (RESULTS_DIR / "metrics.txt").write_text(table.to_string() + "\n")
    print(f"\nMetrics table saved: {(RESULTS_DIR / 'metrics.txt').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
