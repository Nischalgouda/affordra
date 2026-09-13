"""
data_loader.py — Load and join all dataset CSVs into per-request context objects.

CONCEPT (for learning):
  This module is the "data pipeline" layer. Its job is to take raw CSV files and
  produce a clean, structured Python dict for every request — so the financial model
  never has to worry about file I/O, currency conversion, or joins.

  Think of it like a restaurant kitchen's prep station: everything is chopped, 
  measured, and ready before the chef (financial_model.py) starts cooking.
"""

import pandas as pd
from pathlib import Path
from datetime import date, timedelta
import re

DATASET_DIR = Path(__file__).parent.parent / "dataset"


def load_all_data():
    """Load every CSV once and return a dict of DataFrames."""
    data = {}

    data["profiles"] = pd.read_csv(DATASET_DIR / "financial_profiles.csv")
    data["events"] = pd.read_csv(DATASET_DIR / "financial_events.csv")
    data["exchange_rates"] = pd.read_csv(DATASET_DIR / "exchange_rates.csv")
    data["requests"] = pd.read_csv(DATASET_DIR / "requests.csv")
    data["payment_options"] = pd.read_csv(DATASET_DIR / "request_payment_options.csv")
    data["messages"] = pd.read_csv(DATASET_DIR / "messages.csv")
    data["images"] = pd.read_csv(DATASET_DIR / "images.csv")
    data["sample_requests"] = pd.read_csv(DATASET_DIR / "sample_requests.csv")

    # Parse date columns
    for col in ["event_date", "settlement_date"]:
        data["events"][col] = pd.to_datetime(data["events"][col], errors="coerce").dt.date

    data["exchange_rates"]["rate_date"] = pd.to_datetime(
        data["exchange_rates"]["rate_date"], errors="coerce"
    ).dt.date

    data["requests"]["request_date"] = pd.to_datetime(data["requests"]["request_date"]).dt.date
    data["requests"]["desired_completion_date"] = pd.to_datetime(
        data["requests"]["desired_completion_date"]
    ).dt.date

    for col in ["first_payment_date"]:
        data["payment_options"][col] = pd.to_datetime(
            data["payment_options"][col], errors="coerce"
        ).dt.date

    return data


def get_exchange_rate(exchange_rates_df, from_currency, to_currency, target_date):
    """
    Get the exchange rate for a given currency pair on or before a date.
    
    CONCEPT: The dataset has fixed rates per month. We find the rate whose
    date is closest to (but not after) our target date.
    """
    if from_currency == to_currency:
        return 1.0

    subset = exchange_rates_df[
        (exchange_rates_df["from_currency"] == from_currency)
        & (exchange_rates_df["to_currency"] == to_currency)
        & (exchange_rates_df["rate_date"] <= target_date)
    ]
    if subset.empty:
        # Try reverse direction
        subset_rev = exchange_rates_df[
            (exchange_rates_df["from_currency"] == to_currency)
            & (exchange_rates_df["to_currency"] == from_currency)
            & (exchange_rates_df["rate_date"] <= target_date)
        ]
        if not subset_rev.empty:
            rate = subset_rev.sort_values("rate_date").iloc[-1]["rate"]
            return 1.0 / rate
        return 1.0  # Fallback: no conversion (shouldn't happen)

    return subset.sort_values("rate_date").iloc[-1]["rate"]


def convert_amount(amount, from_currency, to_currency, exchange_rates_df, settlement_date):
    """Convert an amount to the target currency."""
    if from_currency == to_currency or pd.isna(amount):
        return amount
    rate = get_exchange_rate(exchange_rates_df, from_currency, to_currency, settlement_date)
    return float(amount) * rate


def detect_recurring_events(user_events_df):
    """
    Detect which settled expenses are recurring (rent, salary, subscriptions, etc.)
    and return a list of (category, direction, avg_amount, avg_gap_days) tuples.

    CONCEPT: We look for categories that appear multiple times with a regular cadence.
    'Cadence' = average days between events in that category.
    """
    settled = user_events_df[
        (user_events_df["status"] == "settled")
        & (user_events_df["direction"] == "debit")
        & (user_events_df["amount"].notna())
    ].copy()

    recurring = []

    for category, group in settled.groupby("category"):
        if len(group) < 2:
            continue

        group = group.sort_values("event_date")
        dates = list(group["event_date"])
        gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]

        if not gaps:
            continue

        avg_gap = sum(gaps) / len(gaps)
        avg_amount = group["amount_home"].mean()
        flexibility = group["flexibility"].iloc[-1] if "flexibility" in group.columns else "fixed"

        # A series is "recurring" if gap is reasonably consistent (CV < 0.5) and < 95 days
        if avg_gap <= 95 and avg_gap >= 3:
            recurring.append({
                "category": category,
                "direction": "debit",
                "avg_amount": avg_amount,
                "avg_gap_days": avg_gap,
                "last_date": dates[-1],
                "flexibility": flexibility,
                "event_ids": list(group["event_id"]),
            })

    # Also detect income (salary credits)
    income_events = user_events_df[
        (user_events_df["status"] == "settled")
        & (user_events_df["direction"] == "credit")
        & (user_events_df["category"] == "salary")
        & (user_events_df["amount"].notna())
    ].sort_values("event_date")

    if len(income_events) >= 1:
        dates = list(income_events["event_date"])
        if len(dates) >= 2:
            gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            avg_gap = sum(gaps) / len(gaps)
        else:
            avg_gap = 30  # Default monthly

        avg_salary = income_events["amount_home"].mean()
        recurring.append({
            "category": "salary",
            "direction": "credit",
            "avg_amount": avg_salary,
            "avg_gap_days": avg_gap,
            "last_date": dates[-1],
            "flexibility": "fixed",
            "event_ids": list(income_events["event_id"]),
        })

    return recurring


def build_request_context(request_row, data, image_amounts=None):
    """
    Build a complete context dict for one request.
    This is the single object passed to the financial model.

    CONCEPT: Instead of passing 8 DataFrames into the financial model, we pre-package
    everything relevant to this specific user + request into one clean dict.
    This is the "context object" pattern — common in AI agent design.
    """
    if image_amounts is None:
        image_amounts = {}

    user_id = request_row["user_id"]
    request_id = request_row["request_id"]

    # Ensure dates are proper date objects (may be strings from sample_requests)
    def to_date(val):
        if isinstance(val, date):
            return val
        if hasattr(val, 'date'):
            return val.date()
        try:
            return pd.to_datetime(str(val)).date()
        except Exception:
            return date.today()

    request_date = to_date(request_row["request_date"])
    desired_completion_date = to_date(request_row["desired_completion_date"])

    # --- 1. User financial profile ---
    profile = data["profiles"][data["profiles"]["user_id"] == user_id].iloc[0]
    home_currency = profile["home_currency"]

    # Parse pipe-separated preference lists
    def parse_list(val):
        if pd.isna(val) or str(val).strip() == "":
            return []
        return [x.strip() for x in str(val).split("|")]

    payment_methods = parse_list(profile["payment_methods_user_will_consider"])
    protected_categories = parse_list(profile["expense_categories_to_protect"])
    reducible_categories = parse_list(profile["expense_categories_user_is_willing_to_reduce"])
    stoppable_categories = parse_list(profile["expense_categories_user_is_willing_to_stop"])
    max_installment_months = (
        int(profile["max_installment_months"])
        if pd.notna(profile.get("max_installment_months", None))
        and str(profile.get("max_installment_months", "")).strip() != ""
        else None
    )

    # --- 2. User financial events (all time) ---
    user_events = data["events"][data["events"]["user_id"] == user_id].copy()

    # Convert amounts to home currency
    def to_home_currency(row):
        if pd.isna(row["amount"]):
            # Check if there's an image amount for this event
            ev_id = row["event_id"]
            if ev_id in image_amounts:
                amt = image_amounts[ev_id]
                if row["currency"] != home_currency:
                    sd = row["settlement_date"] if pd.notna(row["settlement_date"]) else request_date
                    return convert_amount(amt, row["currency"], home_currency, data["exchange_rates"], sd)
                return amt
            return None
        if row["currency"] == home_currency:
            return float(row["amount"])
        sd = row["settlement_date"] if pd.notna(row["settlement_date"]) else request_date
        return convert_amount(float(row["amount"]), row["currency"], home_currency, data["exchange_rates"], sd)

    user_events["amount_home"] = user_events.apply(to_home_currency, axis=1)

    # --- 3. Pending/scheduled events ---
    future_committed = user_events[
        (user_events["status"].isin(["pending", "scheduled"]))
        & (user_events["direction"] == "debit")
        & (user_events["amount_home"].notna())
    ][["event_id", "category", "settlement_date", "amount_home", "flexibility", "status"]].copy()

    future_income = user_events[
        (user_events["status"].isin(["pending", "scheduled"]))
        & (user_events["direction"] == "credit")
        & (user_events["amount_home"].notna())
    ][["event_id", "category", "settlement_date", "amount_home", "status"]].copy()

    # --- 4. Recurring patterns ---
    recurring = detect_recurring_events(user_events)

    # --- 5. Payment options for this request ---
    options = data["payment_options"][
        data["payment_options"]["request_id"] == request_id
    ].copy()

    # --- 6. Messages for this user/request ---
    msgs = data["messages"][
        (data["messages"]["user_id"] == user_id) |
        (data["messages"]["request_id"] == request_id)
    ].copy()

    # --- 7. Images for this user/request ---
    imgs = data["images"][
        (data["images"]["user_id"] == user_id) |
        (data["images"]["request_id"] == request_id)
    ].copy()

    return {
        "request_id": request_id,
        "user_id": user_id,
        "request_date": request_date,
        "request_type": request_row["request_type"],
        "requested_amount": float(request_row["requested_amount"]),
        "desired_completion_date": desired_completion_date,
        "allows_partial_payment": str(request_row.get("allows_partial_payment", "false")).lower() == "true",
        "request_text": request_row.get("request_text", ""),
        # Profile
        "home_currency": home_currency,
        "current_balance": float(profile["current_available_balance"]),
        "min_balance": float(profile["minimum_balance_to_keep"]),
        "payment_methods": payment_methods,
        "protected_categories": protected_categories,
        "reducible_categories": reducible_categories,
        "stoppable_categories": stoppable_categories,
        "max_installment_months": max_installment_months,
        # Events
        "user_events": user_events,
        "future_committed": future_committed,
        "future_income": future_income,
        "recurring": recurring,
        # Request data
        "payment_options": options,
        "messages": msgs,
        "images": imgs,
    }
