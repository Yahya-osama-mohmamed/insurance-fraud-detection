#!/usr/bin/env python
"""Fail the build if the shipped fraud model no longer clears its floor.

The bound that matters here is **recall**, not accuracy and not even PR-AUC:
the whole design premise is that a missed fraudulent claim costs the payout
while a false positive costs a few minutes of review. A retrain that traded
recall for precision would look fine on most dashboards and quietly break the
business case, so recall is the gate.

Reads the metrics table the evaluation step writes, plus the operating
threshold the API serves with.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "reports" / "test_metrics.csv"
THRESHOLD = ROOT / "models" / "threshold.json"

SERVED_MODEL = "LightGBM (SMOTE)"

CHECKS = [
    ("recall", "min", 0.85, "published 0.887 - the point of the model is catching fraud"),
    ("pr_auc", "min", 0.24, "published 0.265; PR-AUC is the selection metric under 6% prevalence"),
    ("roc_auc", "min", 0.82, "published 0.856"),
]


def main() -> int:
    if not METRICS.exists():
        print(f"FAIL: {METRICS} not found - run the evaluation before shipping.")
        return 1

    rows = {r["Model"]: r for r in csv.DictReader(METRICS.open(encoding="utf-8"))}
    if SERVED_MODEL not in rows:
        print(f"FAIL: no row for the served model {SERVED_MODEL!r} in {METRICS.name}")
        print(f"      found: {list(rows)}")
        return 1

    row = rows[SERVED_MODEL]
    print(f"served model: {SERVED_MODEL}")
    if THRESHOLD.exists():
        t = json.loads(THRESHOLD.read_text(encoding="utf-8"))
        print(f"threshold   : {t.get('threshold')} ({t.get('criterion')})")
    print()

    failures = []
    for metric, kind, bound, why in CHECKS:
        value = float(row[metric])
        ok = value >= bound if kind == "min" else value <= bound
        symbol = ">=" if kind == "min" else "<="
        print(f"{'PASS' if ok else 'FAIL'}  {metric:10} {value:<8} {symbol} {bound}   ({why})")
        if not ok:
            failures.append(f"{metric}={value} violates {symbol} {bound}")

    print()
    if failures:
        print("Model quality gate FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Model quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
