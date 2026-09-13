"""
evaluator.py — Validate agent output against the 25 gold-label sample requests.

CONCEPT (for learning):
  Before running on the full 250 requests, we test on the 25 KNOWN-CORRECT examples.
  This is called an "offline evaluation" or "hold-out validation".

  Metrics we check:
  - affordability_status accuracy (exact match, 4 classes)
  - recommended_payment_method accuracy (exact match, 5 classes)
  - amount_safe_to_pay accuracy (within ±5% tolerance)

  TOKEN USAGE: Zero — purely Python comparisons.
"""

import pandas as pd
from pathlib import Path


DATASET_DIR = Path(__file__).parent.parent / "dataset"


def evaluate(output_df: pd.DataFrame, sample_df: pd.DataFrame) -> dict:
    """
    Compare agent output against gold-label samples.
    Returns a dict of accuracy metrics.
    """
    # Merge on request_id
    merged = pd.merge(
        output_df,
        sample_df[["request_id", "affordability_status", "recommended_payment_method",
                    "amount_safe_to_pay", "earliest_date_for_full_payment"]],
        on="request_id",
        suffixes=("_pred", "_gold"),
    )

    if merged.empty:
        print("[evaluator] No overlapping request_ids between output and samples!")
        return {}

    # 1. Affordability status accuracy
    status_correct = (
        merged["affordability_status_pred"].str.strip() ==
        merged["affordability_status_gold"].str.strip()
    ).sum()
    status_acc = status_correct / len(merged)

    # 2. Payment method accuracy
    method_correct = (
        merged["recommended_payment_method_pred"].str.strip() ==
        merged["recommended_payment_method_gold"].str.strip()
    ).sum()
    method_acc = method_correct / len(merged)

    # 3. Amount accuracy (within ±5%)
    def amount_within_5pct(row):
        pred = float(str(row["amount_safe_to_pay_pred"]).replace(",", ""))
        gold = float(str(row["amount_safe_to_pay_gold"]).replace(",", ""))
        if gold == 0:
            return pred == 0
        return abs(pred - gold) / gold <= 0.05

    amount_ok = merged.apply(amount_within_5pct, axis=1).sum()
    amount_acc = amount_ok / len(merged)

    results = {
        "total_evaluated": len(merged),
        "affordability_status_accuracy": round(status_acc, 3),
        "payment_method_accuracy": round(method_acc, 3),
        "amount_accuracy_5pct": round(amount_acc, 3),
        "overall_score": round((status_acc + method_acc + amount_acc) / 3, 3),
    }

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS (vs 25 gold samples)")
    print("=" * 50)
    print(f"Samples evaluated:          {results['total_evaluated']}")
    print(f"Affordability accuracy:     {results['affordability_status_accuracy']:.1%}")
    print(f"Payment method accuracy:    {results['payment_method_accuracy']:.1%}")
    print(f"Amount accuracy (±5%):      {results['amount_accuracy_5pct']:.1%}")
    print(f"Overall score:              {results['overall_score']:.1%}")
    print("=" * 50)

    # Show mismatches for debugging
    status_wrong = merged[
        merged["affordability_status_pred"].str.strip() !=
        merged["affordability_status_gold"].str.strip()
    ][["request_id", "affordability_status_pred", "affordability_status_gold"]]

    if not status_wrong.empty:
        print("\nStatus mismatches:")
        print(status_wrong.to_string(index=False))

    method_wrong = merged[
        merged["recommended_payment_method_pred"].str.strip() !=
        merged["recommended_payment_method_gold"].str.strip()
    ][["request_id", "recommended_payment_method_pred", "recommended_payment_method_gold"]]

    if not method_wrong.empty:
        print("\nMethod mismatches:")
        print(method_wrong.to_string(index=False))

    return results


def run_evaluation(output_csv_path: str = None) -> dict:
    """Load output.csv and sample_requests.csv and run evaluation."""
    if output_csv_path is None:
        output_csv_path = Path(__file__).parent.parent / "output.csv"

    output_df = pd.read_csv(output_csv_path)
    sample_df = pd.read_csv(DATASET_DIR / "sample_requests.csv")

    return evaluate(output_df, sample_df)


if __name__ == "__main__":
    run_evaluation()
