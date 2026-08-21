"""Evaluation on the held-out test set.

Accuracy is deliberately NOT reported: with 6% fraud prevalence a model that
never flags anything scores 94% accuracy while catching zero fraud. We report
Precision, Recall, F1, ROC-AUC and PR-AUC (average precision), which is the
informative scalar under heavy imbalance.

Operating threshold
-------------------
The default 0.5 cutoff is meaningless for an imbalanced fraud screen. Missing
a fraud (false negative) costs the full claim payout; a false positive costs a
few minutes of analyst review. We therefore choose the threshold that
maximizes F2 (recall weighted 2x over precision) on *out-of-fold training
predictions* — the test set plays no part in choosing the threshold.
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

from src.preprocessing import PROCESSED_DIR, PROJECT_ROOT, TARGET
from src.train import CV, MODELS_DIR, REPORTS_DIR

FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

MODELS = {
    "Logistic Regression (balanced)": "baseline_logreg.pkl",
    "LightGBM (class weight)": "lgbm_weighted.pkl",
    "LightGBM (SMOTE)": "lgbm_smote.pkl",
}


def choose_threshold(model, X_train, y_train) -> float:
    """F2-optimal threshold from out-of-fold training predictions."""
    oof = cross_val_predict(model, X_train, y_train, cv=CV,
                            method="predict_proba", n_jobs=-1)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, oof)
    f2 = (5 * precision[:-1] * recall[:-1]) / \
         np.clip(4 * precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f2))])


def metrics_at(y_true, probs, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "f2": fbeta_score(y_true, preds, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": average_precision_score(y_true, probs),
    }


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    # Threshold chosen on the MAIN model, from training data only
    main_model = joblib.load(MODELS_DIR / "fraud_model.pkl")
    threshold = choose_threshold(main_model, X_train, y_train)
    (MODELS_DIR / "threshold.json").write_text(json.dumps(
        {"threshold": round(threshold, 4), "criterion": "max F2 (out-of-fold)"}))
    print(f"operating threshold (max F2, OOF): {threshold:.3f}")

    # ---- Test metrics for every candidate at that threshold ----
    rows = []
    for name, fname in MODELS.items():
        model = joblib.load(MODELS_DIR / fname)
        probs = model.predict_proba(X_test)[:, 1]
        rows.append({"Model": name, **{k: round(v, 4) for k, v in
                                       metrics_at(y_test, probs, threshold).items()}})
    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    table.to_csv(REPORTS_DIR / "test_metrics.csv", index=False)
    print(table.to_string(index=False))

    # ---- Figures for the main model ----
    probs = main_model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Oranges")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=16,
                fontweight="bold")
    ax.set_xticks([0, 1], ["Predicted clean", "Predicted fraud"])
    ax.set_yticks([0, 1], ["Actual clean", "Actual fraud"])
    ax.set_title(f"Confusion Matrix @ threshold {threshold:.2f} (test set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(recall, precision, color="#B85C38", lw=2)
    ax.axhline(y_test.mean(), color="grey", ls="--",
               label=f"Prevalence ({y_test.mean():.1%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve — PR-AUC {average_precision_score(y_test, probs):.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)

    print(f"figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
