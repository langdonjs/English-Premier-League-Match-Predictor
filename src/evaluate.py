"""Evaluation: baselines, classification reports, and confusion-matrix plots.

Everything here operates on the same held-out test set so models and baselines
are directly comparable. Nothing is estimated — every number comes from a real
prediction over the test rows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from .data import CLASS_NAMES

LABELS = [0, 1, 2]


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def majority_and_random_baselines(X_train, y_train, X_test, seed=42):
    """Return predictions for the majority-class and uniform-random baselines."""
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(X_train, y_train)
    random = DummyClassifier(strategy="uniform", random_state=seed)
    random.fit(X_train, y_train)
    return majority.predict(X_test), random.predict(X_test)


def bookmaker_baseline(test_df: pd.DataFrame):
    """Predict outcomes from Bet365 odds (implied prob argmax).

    Returns (predictions, mask) where mask marks test rows that actually had
    odds. Odds are inverted (1/odds) and normalised to strip the bookmaker's
    overround before taking the argmax.
    """
    cols = ["B365H", "B365D", "B365A"]
    have = test_df[cols].notna().all(axis=1)
    odds = test_df.loc[have, cols].to_numpy(dtype=float)
    implied = 1.0 / odds
    implied = implied / implied.sum(axis=1, keepdims=True)
    # Column order H, D, A already matches labels 0, 1, 2.
    preds = implied.argmax(axis=1)
    return preds, have.to_numpy()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def full_report(y_true, y_pred) -> str:
    """Full per-class classification report as printable text."""
    return classification_report(
        y_true, y_pred, labels=LABELS, target_names=CLASS_NAMES, zero_division=0
    )


def metrics_row(y_true, y_pred) -> dict:
    """Compact metrics used for the side-by-side comparison table."""
    rep = classification_report(
        y_true, y_pred, labels=LABELS, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": rep["macro avg"]["f1-score"],
        "Home Win F1": rep["Home Win"]["f1-score"],
        "Draw F1": rep["Draw"]["f1-score"],
        "Away Win F1": rep["Away Win"]["f1-score"],
        "Draw recall": rep["Draw"]["recall"],
    }


def comparison_table(rows: dict[str, dict]) -> pd.DataFrame:
    """Assemble labelled metric rows into a display DataFrame."""
    return pd.DataFrame(rows).T.round(3)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def save_confusion_matrix(y_true, y_pred, title: str, out_path: str | Path) -> Path:
    """Render and save a confusion-matrix PNG (raw counts)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
