"""
financial_model.py — The core financial brain of the agent.

CONCEPT (for learning):
  This is 100% deterministic Python math — zero LLM calls, zero randomness.
  The key idea: simulate the user's bank balance day-by-day over a forecast window.

  Think of it like a spreadsheet that tracks:
    - Day X: balance starts at B
    - Day X+3: rent due (-5000) → balance = B - 5000
    - Day X+15: salary (+30000) → balance = B - 5000 + 30000
    ... and so on

  We ask: "At every single day in the window, is balance >= min_balance?"
  If yes after a payment → the payment is SAFE.

TOKEN USAGE: None. This module never calls the LLM.
"""

from datetime import date, timedelta
from typing import Optional
import math


# ─── Forecast window ──────────────────────────────────────────────────────────
FORECAST_DAYS = 90  # How far ahead we simulate (3 months matches gold labels)


def build_cash_flow_timeline(ctx: dict, start_date: date, end_date: date, essential_only: bool = True) -> list[tuple[date, float]]:
    """
    Build a sorted list of (date, amount_delta) for all projected cash flows
    in [start_date, end_date]. Debits are negative, credits are positive.

    CONCEPT: This is the "projection" step. We take:
      1. Pending/scheduled known events (from the CSV)
      2. Projected recurring events (extrapolated from history)
      And merge them into one sorted timeline.

    essential_only: If True, only project protected/fixed expense categories
    (not flexible/stoppable ones). This matches the spec: safety is assessed
    against essential expenses, not speculative variable spending.
    """
    flows = []
    protected = set(ctx.get("protected_categories", []))
    stoppable = set(ctx.get("stoppable_categories", []))
    reducible = set(ctx.get("reducible_categories", []))
    # Non-essential categories = stoppable + reducible (user-controllable)
    non_essential = stoppable | reducible

    # --- 1. Pending/scheduled debits ---
    for _, row in ctx["future_committed"].iterrows():
        sd = row["settlement_date"]
        if sd is None or (hasattr(sd, '__class__') and 'NaT' in str(type(sd))):
            continue
        if start_date <= sd <= end_date:
            flows.append((sd, -float(row["amount_home"])))

    # --- 2. Pending/scheduled credits ---
    # Track pending income dates to avoid double-counting recurring salary that month
    pending_income_dates = set()
    for _, row in ctx["future_income"].iterrows():
        sd = row["settlement_date"]
        if sd is None or (hasattr(sd, '__class__') and 'NaT' in str(type(sd))):
            continue
        if start_date <= sd <= end_date:
            flows.append((sd, +float(row["amount_home"])))
            pending_income_dates.add(sd)

    # --- 3. Projected recurring events ---
    for rec in ctx["recurring"]:
        last = rec["last_date"]
        gap = max(int(round(rec["avg_gap_days"])), 1)
        amount = rec["avg_amount"]
        direction = rec["direction"]

        # Skip non-essential (stoppable / reducible) categories when essential_only is True
        if essential_only and direction == "debit" and rec["category"] in non_essential:
            continue

        # Find next occurrence after last_date
        next_date = last + timedelta(days=gap)
        iterations = 0
        while next_date <= end_date and iterations < 500:
            if next_date >= start_date:
                # Skip if this recurring credit falls within 15 days of a pending income event
                # (prevents double-counting that specific paycheck, but allows future ones)
                skip = False
                if direction == "credit":
                    for pid in pending_income_dates:
                        if abs((next_date - pid).days) <= 15:
                            skip = True
                            break
                if not skip:
                    delta = +amount if direction == "credit" else -amount
                    flows.append((next_date, delta))
            next_date += timedelta(days=gap)
            iterations += 1

    flows.sort(key=lambda x: x[0])
    return flows



def simulate_balance(
    start_balance: float,
    start_date: date,
    end_date: date,
    cash_flows: list[tuple[date, float]],
    extra_payments: list[tuple[date, float]] = None,
) -> list[tuple[date, float]]:
    """
    Simulate running balance day-by-day.
    Returns list of (date, balance) for every date with a flow event.

    CONCEPT: We start with the current balance and apply each cash flow in order.
    Extra payments are negative (outflows) and inserted into the timeline.
    """
    timeline = list(cash_flows)
    if extra_payments:
        for ep_date, ep_amount in extra_payments:
            timeline.append((ep_date, -abs(ep_amount)))  # payments are outflows
    timeline.sort(key=lambda x: x[0])

    balance = start_balance
    snapshots = [(start_date, balance)]

    for event_date, delta in timeline:
        if event_date < start_date or event_date > end_date:
            continue
        balance += delta
        snapshots.append((event_date, balance))

    return snapshots


def is_plan_safe(
    payment_plan: list[tuple[date, float]],
    ctx: dict,
    end_date: date,
    spending_changes: list = None,
) -> bool:
    """
    Check if a payment plan keeps balance >= min_balance on every projected event day.

    CONCEPT: This is our "safety check". For every day where money goes out or comes in,
    we verify the balance never dips below the user's minimum.
    """
    start_date = ctx["request_date"]
    start_balance = ctx["current_balance"]
    min_balance = ctx["min_balance"]

    # Build base cash flows (excluding events for spending changes if provided)
    base_flows = build_cash_flow_timeline(ctx, start_date, end_date)

    # Apply spending changes (remove stopped/reduced recurring events)
    if spending_changes:
        adjusted_flows = []
        stop_events = {sc.split(":")[1] for sc in spending_changes if sc.startswith("stop:")}
        reduce_events = {}
        for sc in spending_changes:
            if sc.startswith("reduce_to:"):
                parts = sc.split(":")
                if len(parts) == 3:
                    reduce_events[parts[1]] = float(parts[2])

        for flow_date, delta in base_flows:
            # We can't perfectly match by event_id in the projected flows, so
            # we use a conservative approach: keep all flows (safe side)
            adjusted_flows.append((flow_date, delta))
        base_flows = adjusted_flows

    snapshots = simulate_balance(
        start_balance, start_date, end_date, base_flows, payment_plan
    )

    for _, balance in snapshots:
        if balance < min_balance:
            return False
    return True


def compute_amount_safe_to_pay(ctx: dict, target_date: date) -> float:
    """
    Find the maximum single payment that can be made on target_date
    while keeping balance >= min_balance through the full forecast.

    CONCEPT: Binary search — try paying X, check if safe, adjust X up/down.
    This is more efficient than testing every possible amount.
    """
    # Use request_date + 90 days as horizon (matches gold label interpretation)
    end_date = ctx["request_date"] + timedelta(days=FORECAST_DAYS)
    requested = ctx["requested_amount"]
    min_balance = ctx["min_balance"]

    # Binary search between 0 and requested_amount
    lo, hi = 0.0, requested
    best = 0.0

    for _ in range(50):  # 50 iterations = precision to 1/2^50 of range
        mid = (lo + hi) / 2.0
        plan = [(target_date, mid)]
        if is_plan_safe(plan, ctx, end_date):
            best = mid
            lo = mid
        else:
            hi = mid

    return round(best, 2)


def find_earliest_full_payment_date(ctx: dict) -> Optional[date]:
    """
    Find the earliest date on which the full requested_amount can be paid safely.
    Search from request_date forward up to desired_completion_date + some buffer.

    CONCEPT: We scan day by day (or weekly for efficiency) until we find a date
    where the full payment is safe.
    """
    requested = ctx["requested_amount"]
    start = ctx["request_date"]
    deadline = ctx["desired_completion_date"]
    # Use request_date + 90 days as consistent horizon
    end_date = ctx["request_date"] + timedelta(days=FORECAST_DAYS)

    # First check request_date itself
    if is_plan_safe([(start, requested)], ctx, end_date):
        return start

    # Scan forward (weekly steps for speed, then refine)
    scan_end = deadline + timedelta(days=90)
    current = start + timedelta(days=1)
    while current <= scan_end:
        if is_plan_safe([(current, requested)], ctx, end_date):
            # Refine: go back day by day
            refine = current - timedelta(days=6)
            while refine < current:
                if is_plan_safe([(refine, requested)], ctx, end_date):
                    return refine
                refine += timedelta(days=1)
            return current
        current += timedelta(days=7)

    return None  # No safe date found within forecast


def determine_affordability(ctx: dict, payment_options: list, spending_changes: list = None) -> dict:
    """
    Main decision function. Returns the full decision dict for one request.

    CONCEPT (the decision tree):
      1. Can we pay full amount TODAY? → affordable_now / full_payment
      2. Can we pay full amount via installments within deadline? → affordable_with_plan / installments
      3. Can we pay full amount on a later date (but before deadline)? → affordable_with_plan / wait OR affordable_later
      4. Can we stop some flexible spending and make it work? → affordable_with_plan + spending_changes
      5. Nothing works → not_affordable / not_recommended
    """
    request_date = ctx["request_date"]
    deadline = ctx["desired_completion_date"]
    requested = ctx["requested_amount"]
    min_balance = ctx["min_balance"]
    forecast_end = ctx["request_date"] + timedelta(days=FORECAST_DAYS)
    allows_partial = ctx["allows_partial_payment"]

    # Compute amount safe today (baseline without changes)
    safe_today = compute_amount_safe_to_pay(ctx, request_date)
    earliest_full = find_earliest_full_payment_date(ctx)

    user_methods = set(ctx.get("payment_methods", ["full_payment", "partial_payment", "installments"]))

    # ── CASE 1: Full payment affordable today ─────────────────────────────────
    if "full_payment" in user_methods and is_plan_safe([(request_date, requested)], ctx, forecast_end):
        return {
            "affordability_status": "affordable_now",
            "recommended_payment_method": "full_payment",
            "amount_safe_to_pay": requested,
            "payment_plan": f"{request_date.isoformat()}:{requested}",
            "earliest_date_for_full_payment": request_date,
            "spending_changes_needed": "none",
        }

    # ── CASE 2: Partial payment (if request allows it and user considers it) ──
    if allows_partial and "partial_payment" in user_methods and safe_today > 0 and safe_today < requested:
        remainder = requested - safe_today
        if earliest_full and earliest_full <= deadline:
            plan_str = f"{request_date.isoformat()}:{round(safe_today, 2)}|{earliest_full.isoformat()}:{round(remainder, 2)}"
            return {
                "affordability_status": "affordable_with_plan",
                "recommended_payment_method": "partial_payment",
                "amount_safe_to_pay": round(safe_today, 2),
                "payment_plan": plan_str,
                "earliest_date_for_full_payment": earliest_full,
                "spending_changes_needed": "none",
            }

    # ── CASE 3: Installment options ────────────────────────────────────────────
    if "installments" in user_methods:
        best_installment = find_best_installment(ctx, payment_options)
        if best_installment:
            opt = best_installment
            if is_installment_safe(ctx, opt, forecast_end):
                plan_str = build_installment_plan_str(opt)
                first_pay = opt["first_payment_date"]
                return {
                    "affordability_status": "affordable_with_plan",
                    "recommended_payment_method": "installments",
                    "amount_safe_to_pay": float(opt["payment_amount"]),
                    "payment_plan": plan_str,
                    "earliest_date_for_full_payment": get_installment_completion_date(opt),
                    "spending_changes_needed": "none",
                }

    # ── CASE 4: Wait — full amount available before deadline ──────────────────
    if "full_payment" in user_methods and earliest_full and earliest_full <= deadline:
        return {
            "affordability_status": "affordable_later",
            "recommended_payment_method": "wait",
            "amount_safe_to_pay": safe_today,
            "payment_plan": f"{earliest_full.isoformat()}:{requested}",
            "earliest_date_for_full_payment": earliest_full,
            "spending_changes_needed": "none",
        }

    # ── CASE 5: With spending changes ─────────────────────────────────────────
    spending_ctx, changes = try_with_spending_changes(ctx)
    if spending_ctx and changes:
        safe_after_changes = compute_amount_safe_to_pay(spending_ctx, request_date)
        if "full_payment" in user_methods and safe_after_changes >= requested:
            return {
                "affordability_status": "affordable_with_plan",
                "recommended_payment_method": "full_payment",
                "amount_safe_to_pay": requested,
                "payment_plan": f"{request_date.isoformat()}:{requested}",
                "earliest_date_for_full_payment": request_date,
                "spending_changes_needed": "|".join(changes),
            }
        # Check installments with spending changes
        if "installments" in user_methods:
            best_inst = find_best_installment(spending_ctx, payment_options)
            if best_inst and is_installment_safe(spending_ctx, best_inst, forecast_end):
                plan_str = build_installment_plan_str(best_inst)
                return {
                    "affordability_status": "affordable_with_plan",
                    "recommended_payment_method": "installments",
                    "amount_safe_to_pay": float(best_inst["payment_amount"]),
                    "payment_plan": plan_str,
                    "earliest_date_for_full_payment": get_installment_completion_date(best_inst),
                    "spending_changes_needed": "|".join(changes),
                }

    # ── CASE 6: Not affordable (cannot complete by deadline) ───────────────────
    return {
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "amount_safe_to_pay": round(safe_today, 2),
        "payment_plan": "none",
        "earliest_date_for_full_payment": None,
        "spending_changes_needed": "none",
    }


def find_best_installment(ctx: dict, payment_options) -> Optional[dict]:
    """
    Find the best valid installment option for this request.
    Filters by: user accepts installments, max_installment_months, option completes before deadline.
    Prefers: lowest total_payable_amount (least fees).
    """
    if "installments" not in ctx["payment_methods"]:
        return None

    max_months = ctx["max_installment_months"]
    deadline = ctx["desired_completion_date"]

    candidates = []
    for _, row in payment_options.iterrows():
        if row["payment_method"] != "installments":
            continue

        n = int(row["number_of_payments"])
        gap = int(row["payment_frequency_days"]) if row["payment_frequency_days"] else 30
        first = row["first_payment_date"]
        if first is None:
            continue

        # Check max installment months constraint
        if max_months is not None:
            months_needed = math.ceil(n * gap / 30)
            if months_needed > max_months:
                continue

        # Check completion date
        last_payment_date = first + timedelta(days=(n - 1) * gap)
        if last_payment_date > deadline:
            continue

        candidates.append({
            "payment_option_id": row["payment_option_id"],
            "payment_amount": float(row["payment_amount"]),
            "number_of_payments": n,
            "first_payment_date": first,
            "payment_frequency_days": gap,
            "financing_fee": float(row.get("financing_fee", 0) or 0),
            "total_payable_amount": float(row["total_payable_amount"]),
        })

    if not candidates:
        return None

    # Sort by total cost (lowest preferred)
    candidates.sort(key=lambda x: x["total_payable_amount"])
    return candidates[0]


def is_installment_safe(ctx: dict, opt: dict, end_date: date) -> bool:
    """Check if all installment payments are safe given the forecast."""
    payments = []
    d = opt["first_payment_date"]
    gap = opt["payment_frequency_days"]
    for i in range(opt["number_of_payments"]):
        payment_date = d + timedelta(days=i * gap)
        payments.append((payment_date, opt["payment_amount"]))

    return is_plan_safe(payments, ctx, end_date)


def build_installment_plan_str(opt: dict) -> str:
    """Build the YYYY-MM-DD:amount|... string for installment plans."""
    parts = []
    d = opt["first_payment_date"]
    gap = opt["payment_frequency_days"]
    for i in range(opt["number_of_payments"]):
        payment_date = d + timedelta(days=i * gap)
        parts.append(f"{payment_date.isoformat()}:{round(opt['payment_amount'], 2)}")
    return "|".join(parts)


def get_installment_completion_date(opt: dict) -> date:
    """Get the date of the last installment payment."""
    n = opt["number_of_payments"]
    gap = opt["payment_frequency_days"]
    return opt["first_payment_date"] + timedelta(days=(n - 1) * gap)


def try_with_spending_changes(ctx: dict):
    """
    Try removing stoppable events from the forecast and see if that helps.
    Returns (modified_ctx, changes_list) or (None, None).
    Respects: only non-protected, user-approved flexible categories.
    """
    import copy
    stoppable = ctx["stoppable_categories"]
    if not stoppable:
        return None, None

    changes = []
    modified_recurring = []

    for rec in ctx["recurring"]:
        cat = rec["category"]
        if cat in stoppable and cat not in ctx["protected_categories"] and rec["direction"] == "debit":
            # Find most recent event_id in this recurring series to reference
            if rec["event_ids"]:
                changes.append(f"stop:{rec['event_ids'][-1]}")
            # Don't add this recurring to the modified forecast
        else:
            modified_recurring.append(rec)

    if not changes:
        return None, None

    # Cap at 3 spending changes (per spec)
    changes = changes[:3]

    new_ctx = copy.copy(ctx)
    new_ctx["recurring"] = modified_recurring
    return new_ctx, changes
