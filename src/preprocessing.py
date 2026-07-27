"""Data loading, cleaning, and splitting for the fraud detection pipeline.

Cleaning decisions (each documented, none silent):
- PolicyNumber is dropped: it is a 1..N row index, not a feature. Left in, a
  tree model can memorize it (train rows sort chronologically by index) —
  a classic leakage trap in this Kaggle dataset.
- Age == 0 (320 rows, 2.1%) is physically impossible for a policy holder and
  is this dataset's missing-value marker. Converted to NaN; imputation happens
  downstream inside the model pipeline (median learned on training folds only).
- One corrupt row has '0' in DayOfWeekClaimed / MonthClaimed — dropped.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "fraud_oracle.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "FraudFound_P"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Load the raw Kaggle CSV."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented cleaning steps."""
    df = df.copy()

    # Leakage guard: PolicyNumber is a row index (1..15420), not information
    if "PolicyNumber" in df.columns:
        df = df.drop(columns=["PolicyNumber"])

    # '0' strings in the claimed-date fields mark one corrupt row
    n_before = len(df)
    df = df[(df["DayOfWeekClaimed"] != "0") & (df["MonthClaimed"] != "0")]
    dropped_corrupt = n_before - len(df)

    # Age 0 = missing marker -> NaN (imputed later inside the model pipeline)
    n_age_zero = int((df["Age"] == 0).sum())
    df.loc[df["Age"] == 0, "Age"] = pd.NA
    df["Age"] = df["Age"].astype("Float64")

    print(f"clean(): dropped {dropped_corrupt} corrupt row(s), "
          f"flagged {n_age_zero} Age=0 values as missing")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float = TEST_SIZE,
          random_state: int = RANDOM_STATE):
    """Stratified train/test split preserving the ~6% fraud rate."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = clean(load_raw())
    train_df, test_df = split(df)
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    print(f"train: {train_df.shape} ({train_df[TARGET].mean():.2%} fraud) | "
          f"test: {test_df.shape} ({test_df[TARGET].mean():.2%} fraud)")


if __name__ == "__main__":
    main()
