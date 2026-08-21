"""Model training: baseline + main model + imbalance-strategy comparison.

Candidates
----------
1. Logistic Regression, class_weight='balanced'  — the honest baseline.
2. LightGBM with scale_pos_weight = n_neg/n_pos  — main model (class weighting).
3. LightGBM + SMOTE (oversampling inside CV folds via imblearn Pipeline) —
   same hyperparameters as (2) so the comparison isolates the imbalance
   strategy itself, not a different tuning budget.

Model selection uses cross-validated **average precision (PR-AUC)** — the right
scalar under 6% prevalence. The winner is refit on the full training set and
saved to models/fraud_model.pkl.
"""

import json

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import NUMERIC_OUTPUT, OHE_COLUMNS, FraudFeatureEngineer
from src.preprocessing import PROCESSED_DIR, PROJECT_ROOT, RANDOM_STATE, TARGET

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

LGBM_SEARCH_SPACE = {
    "model__n_estimators": [200, 400, 600],
    "model__learning_rate": [0.02, 0.05, 0.1],
    "model__num_leaves": [15, 31, 63],
    "model__min_child_samples": [10, 20, 40],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
    "model__reg_lambda": [0.0, 0.1, 1.0],
}


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), NUMERIC_OUTPUT),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), OHE_COLUMNS),
    ])


def build_baseline() -> Pipeline:
    """Logistic Regression with balanced class weights (scaled features)."""
    return Pipeline([
        ("features", FraudFeatureEngineer()),
        ("prep", _preprocessor(scale_numeric=True)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=RANDOM_STATE)),
    ])


def build_lgbm(scale_pos_weight: float) -> Pipeline:
    return Pipeline([
        ("features", FraudFeatureEngineer()),
        ("prep", _preprocessor(scale_numeric=False)),
        ("model", LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1,
                                 scale_pos_weight=scale_pos_weight, n_jobs=-1)),
    ])


def build_lgbm_smote(best_params: dict) -> ImbPipeline:
    """SMOTE variant: oversampling happens INSIDE each CV fold (imblearn
    pipeline), never before splitting — doing it before the split leaks
    synthetic copies of test-fold minority rows into training."""
    model_params = {k.replace("model__", ""): v for k, v in best_params.items()}
    return ImbPipeline([
        ("features", FraudFeatureEngineer()),
        ("prep", _preprocessor(scale_numeric=False)),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1,
                                 n_jobs=-1, **model_params)),
    ])


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    X, y = train_df.drop(columns=[TARGET]), train_df[TARGET]
    spw = float((y == 0).sum() / (y == 1).sum())
    print(f"training rows: {len(X)} | fraud: {y.mean():.2%} | scale_pos_weight: {spw:.1f}")

    results = {}

    # ---- 1. Baseline ----
    baseline = build_baseline()
    results["Logistic Regression (balanced)"] = cross_val_score(
        baseline, X, y, cv=CV, scoring="average_precision", n_jobs=-1).mean()
    print(f"baseline CV PR-AUC: {results['Logistic Regression (balanced)']:.4f}")

    # ---- 2. Main model: tuned LightGBM with class weighting ----
    search = RandomizedSearchCV(
        build_lgbm(spw), LGBM_SEARCH_SPACE, n_iter=15, cv=CV,
        scoring="average_precision", random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    results["LightGBM (class weight)"] = search.best_score_
    best_params = search.best_params_
    print(f"LightGBM CV PR-AUC: {search.best_score_:.4f} | params: {best_params}")

    # ---- 3. SMOTE comparison (same hyperparameters, different strategy) ----
    smote_pipe = build_lgbm_smote(best_params)
    results["LightGBM (SMOTE)"] = cross_val_score(
        smote_pipe, X, y, cv=CV, scoring="average_precision", n_jobs=-1).mean()
    print(f"LightGBM+SMOTE CV PR-AUC: {results['LightGBM (SMOTE)']:.4f}")

    # ---- Persist all three fitted on full training data ----
    baseline.fit(X, y)
    joblib.dump(baseline, MODELS_DIR / "baseline_logreg.pkl")
    smote_pipe.fit(X, y)
    joblib.dump(smote_pipe, MODELS_DIR / "lgbm_smote.pkl")
    joblib.dump(search.best_estimator_, MODELS_DIR / "lgbm_weighted.pkl")

    # The SERVED artifact is the CV winner (selection by CV, never by test)
    winner = max(results, key=results.get)
    winner_pipe = {"Logistic Regression (balanced)": baseline,
                   "LightGBM (class weight)": search.best_estimator_,
                   "LightGBM (SMOTE)": smote_pipe}[winner]
    joblib.dump(winner_pipe, MODELS_DIR / "fraud_model.pkl")
    (REPORTS_DIR / "cv_results.json").write_text(json.dumps({
        "cv_pr_auc": {k: round(v, 4) for k, v in results.items()},
        "winner": winner,
        "best_params": best_params,
        "scale_pos_weight": round(spw, 2),
    }, indent=2))
    print(f"winner by CV PR-AUC: {winner}")
    print(f"saved models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
