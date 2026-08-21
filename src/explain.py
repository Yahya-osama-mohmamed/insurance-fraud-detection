"""SHAP explainability: global drivers + per-claim reasons for the API.

The API's `top_reasons` field comes from `reason_strings()`: the top SHAP
contributions for one claim, translated into short human-readable sentences a
claims analyst can act on — the difference between "score 0.87" and an
explanation the reviewer can trust.
"""


import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.preprocessing import PROCESSED_DIR, PROJECT_ROOT, TARGET
from src.train import MODELS_DIR

FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Run raw claims through every step except the classifier.

    Resampler steps (e.g. SMOTE in the imblearn pipeline) only act during
    fit and have no transform() — they are skipped at inference, exactly as
    the imblearn pipeline itself does when predicting.
    """
    steps = [(n, s) for n, s in pipeline.steps[:-1] if hasattr(s, "transform")]
    transformer = Pipeline(steps)
    X_t = transformer.transform(X)
    names = list(pipeline.named_steps["prep"].get_feature_names_out())
    return pd.DataFrame(np.asarray(X_t), columns=names, index=X.index)


def get_explainer(pipeline: Pipeline) -> shap.TreeExplainer:
    return shap.TreeExplainer(pipeline.named_steps["model"])


def shap_values_for(pipeline: Pipeline, explainer, X: pd.DataFrame):
    X_t = transform_features(pipeline, X)
    sv = explainer.shap_values(X_t)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    return sv, X_t


# ---------------------------------------------------------------------------
# Human-readable reason templates (feature -> analyst language)
# ---------------------------------------------------------------------------
_REASON_TEMPLATES = {
    "num__short_gap_flag": "accident occurred within days of the policy start",
    "num__recent_address_change_flag": "address was changed shortly before the claim",
    "num__no_report_no_witness": "no police report filed and no witness present",
    "num__claim_month_mismatch": "claim was filed in a different month than the accident",
    "num__price_z_in_category": "vehicle price is unusual for its category ({value:+.1f} std devs)",
    "num__price_per_vehicle_year": "high vehicle value relative to its age",
    "num__PastNumberOfClaims_num": "history of prior claims ({value:.0f})",
    "num__Days_Policy_Accident_num": "short gap between policy start and accident ({value:.0f} days)",
    "num__Days_Policy_Claim_num": "claim filed {value:.0f} days after the accident",
    "num__AddressChange_Claim_num": "address change {value:.1f} years before the claim",
    "num__Age": "policy holder age ({value:.0f})",
    "num__AgeOfPolicyHolder_num": "policy holder age bracket (mid ~{value:.0f})",
    "num__make_freq": "uncommon vehicle make",
    "num__Deductible": "deductible level ({value:.0f})",
    "num__age_was_missing": "policy holder age missing from the record",
}


def _pretty(feature: str) -> str:
    name = feature.split("__", 1)[-1]
    return name.replace("_num", "").replace("_", " ")


# Binary features whose templates describe the value==1 state; when the value
# is 0 the template sentence would be FALSE for this claim, so skip it
_BINARY_FLAGS = {
    "num__short_gap_flag", "num__recent_address_change_flag",
    "num__no_report_no_witness", "num__claim_month_mismatch",
    "num__age_was_missing",
}


def reason_strings(shap_row: np.ndarray, feature_values: pd.Series,
                   top_n: int = 3) -> list:
    """Top-N risk-increasing SHAP contributions as analyst-readable strings.

    Only statements that are TRUE of this claim are emitted: a one-hot dummy
    or binary flag at value 0 can still carry positive SHAP (risk from the
    *absence* of a protective attribute), but phrasing it as if present
    ("police report filed = no" on a claim that HAS a report) would mislead
    the reviewer — those features are skipped in favor of the next driver.
    """
    order = np.argsort(shap_row)[::-1]
    reasons = []
    for idx in order:
        if shap_row[idx] <= 0 or len(reasons) >= top_n:
            break
        feature = feature_values.index[idx]
        value = feature_values.iloc[idx]

        # Skip absent dummies/flags — their template text would be false
        is_dummy = feature.startswith("cat__") or feature in _BINARY_FLAGS
        if is_dummy and float(value) < 0.5:
            continue

        template = _REASON_TEMPLATES.get(feature)
        if template:
            try:
                text = template.format(value=float(value))
            except (ValueError, TypeError):
                text = template
        elif feature.startswith("cat__"):
            # One-hot columns look like cat__Fault_Policy Holder
            raw = feature.split("__", 1)[-1]
            text = raw.replace("_", " = ", 1).lower()
        else:
            text = f"{_pretty(feature)} = {value}"
        reasons.append(text)
    return reasons


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = joblib.load(MODELS_DIR / "fraud_model.pkl")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    X_test = test_df.drop(columns=[TARGET])

    explainer = get_explainer(pipeline)
    sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    sv, X_t = shap_values_for(pipeline, explainer, sample)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv, X_t, plot_type="bar", show=False, max_display=15)
    plt.title("Global feature importance (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_global_importance.png", dpi=150)
    plt.close("all")

    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv, X_t, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=150)
    plt.close("all")

    # Waterfall for the highest-scoring claim in the sample
    probs = pipeline.predict_proba(sample)[:, 1]
    hot = int(np.argmax(probs))
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = np.atleast_1d(expected)[-1]
    explanation = shap.Explanation(
        values=sv[hot], base_values=expected,
        data=X_t.iloc[hot].values, feature_names=list(X_t.columns))
    plt.figure(figsize=(9, 7))
    shap.plots.waterfall(explanation, show=False, max_display=12)
    plt.title(f"Why this claim was flagged (p={probs[hot]:.2f})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_waterfall_flagged.png", dpi=150)
    plt.close("all")

    print("example reasons for the hottest claim:")
    for r in reason_strings(sv[hot], X_t.iloc[hot]):
        print("  -", r)
    print(f"figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
