"""Feature engineering for the fraud detection pipeline.

The raw dataset encodes most numeric quantities as ordered text bins
("20000 to 29000", "more than 30", ...). We convert them to numeric midpoints
so models can exploit their ordering, then add derived red-flag features that
mirror how a human fraud investigator reasons.

`FraudFeatureEngineer` is an sklearn transformer that lives INSIDE the model
pipeline: every statistic it needs (Age median, Make frequencies, per-category
price mean/std) is learned in fit() on training folds only — no leakage, and
identical behavior at API serving time.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Ordered-bin -> numeric midpoint maps (documented, deterministic)
# ---------------------------------------------------------------------------
ORDINAL_MAPS = {
    "VehiclePrice": {
        "less than 20000": 15000, "20000 to 29000": 24500, "30000 to 39000": 34500,
        "40000 to 59000": 49500, "60000 to 69000": 64500, "more than 69000": 80000,
    },
    "Days_Policy_Accident": {
        "none": 0, "1 to 7": 4, "8 to 15": 11.5, "15 to 30": 22.5, "more than 30": 45,
    },
    "Days_Policy_Claim": {
        "none": 0, "8 to 15": 11.5, "15 to 30": 22.5, "more than 30": 45,
    },
    "PastNumberOfClaims": {"none": 0, "1": 1, "2 to 4": 3, "more than 4": 6},
    "AgeOfVehicle": {
        "new": 0, "2 years": 2, "3 years": 3, "4 years": 4,
        "5 years": 5, "6 years": 6, "7 years": 7, "more than 7": 9,
    },
    "AgeOfPolicyHolder": {
        "16 to 17": 16.5, "18 to 20": 19, "21 to 25": 23, "26 to 30": 28,
        "31 to 35": 33, "36 to 40": 38, "41 to 50": 45.5, "51 to 65": 58,
        "over 65": 70,
    },
    "NumberOfSuppliments": {"none": 0, "1 to 2": 1.5, "3 to 5": 4, "more than 5": 7},
    "AddressChange_Claim": {
        # Years since the address changed relative to the claim.
        # "no change" gets a large sentinel (no recent change = low risk).
        "under 6 months": 0.4, "1 year": 1, "2 to 3 years": 2.5,
        "4 to 8 years": 6, "no change": 99,
    },
    "NumberOfCars": {
        "1 vehicle": 1, "2 vehicles": 2, "3 to 4": 3.5, "5 to 8": 6.5,
        "more than 8": 9,
    },
}

# Categoricals one-hot encoded downstream (low cardinality)
OHE_COLUMNS = [
    "Month", "DayOfWeek", "AccidentArea", "DayOfWeekClaimed", "MonthClaimed",
    "Sex", "MaritalStatus", "Fault", "PolicyType", "VehicleCategory",
    "PoliceReportFiled", "WitnessPresent", "AgentType", "BasePolicy",
]

# Numeric columns produced by the transformer (input order preserved)
NUMERIC_OUTPUT = [
    "Age", "WeekOfMonth", "WeekOfMonthClaimed", "RepNumber", "Deductible",
    "DriverRating", "Year",
    *(f"{c}_num" for c in ORDINAL_MAPS),
    "make_freq", "price_per_vehicle_year", "price_z_in_category",
    "short_gap_flag", "recent_address_change_flag", "claim_month_mismatch",
    "no_report_no_witness", "age_was_missing",
]


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Ordinal decoding + investigator-style derived features.

    Learned in fit (training data only): Age median, Make frequency table,
    per-VehicleCategory price mean/std.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FraudFeatureEngineer":
        X = X.copy()
        price = X["VehiclePrice"].map(ORDINAL_MAPS["VehiclePrice"])
        self.age_median_ = float(pd.to_numeric(X["Age"], errors="coerce").median())
        self.make_freq_ = (X["Make"].value_counts(normalize=True)).to_dict()
        grp = price.groupby(X["VehicleCategory"])
        self.cat_price_mean_ = grp.mean().to_dict()
        self.cat_price_std_ = grp.std().replace(0, 1).fillna(1).to_dict()
        self.global_price_mean_ = float(price.mean())
        self.global_price_std_ = float(price.std() or 1)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Ordinal decoding
        for col, mapping in ORDINAL_MAPS.items():
            df[f"{col}_num"] = df[col].map(mapping)

        # Age: numeric + missing flag + median imputation (median from fit)
        age = pd.to_numeric(df["Age"], errors="coerce")
        df["age_was_missing"] = age.isna().astype(int)
        df["Age"] = age.fillna(self.age_median_)

        # Frequency-encode Make (19 levels — too many for clean one-hot)
        df["make_freq"] = df["Make"].map(self.make_freq_).fillna(0.0)

        # --- Investigator-style derived features ---
        # Expensive car that is nearly new (price per year of vehicle age)
        df["price_per_vehicle_year"] = (
            df["VehiclePrice_num"] / (df["AgeOfVehicle_num"] + 1))
        # Price z-score within its vehicle category (stats from training fit)
        cat_mean = df["VehicleCategory"].map(self.cat_price_mean_).fillna(self.global_price_mean_)
        cat_std = df["VehicleCategory"].map(self.cat_price_std_).fillna(self.global_price_std_)
        df["price_z_in_category"] = (df["VehiclePrice_num"] - cat_mean) / cat_std
        # Accident within a week of policy start — classic red-flag pattern
        df["short_gap_flag"] = (df["Days_Policy_Accident_num"] <= 7).astype(int)
        # Address changed shortly before the claim
        df["recent_address_change_flag"] = (df["AddressChange_Claim_num"] <= 1).astype(int)
        # Claim filed in a different month than the accident
        df["claim_month_mismatch"] = (df["Month"] != df["MonthClaimed"]).astype(int)
        # Neither police report nor witness — unverifiable claim
        df["no_report_no_witness"] = (
            (df["PoliceReportFiled"] == "No") & (df["WitnessPresent"] == "No")
        ).astype(int)

        return df[NUMERIC_OUTPUT + OHE_COLUMNS]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(NUMERIC_OUTPUT + OHE_COLUMNS, dtype=object)
