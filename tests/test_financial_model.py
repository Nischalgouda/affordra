"""
tests/test_financial_model.py - Affordra unit tests (pytest)
Tests core financial decision logic without needing the full dataset.
"""
import sys
import os
from datetime import date, timedelta
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from financial_model import (
    determine_affordability,
    compute_amount_safe_to_pay,
    is_plan_safe,
    simulate_balance,
)


def _mock_ctx(
    balance=10000.0, min_balance=1000.0, requested=5000.0,
    request_date=None, deadline_offset_days=30,
    payment_methods=None, allows_partial=False,
    protected_categories=None, stoppable_categories=None, reducible_categories=None,
):
    if request_date is None:
        request_date = date.today()
    return {
        "user_id": "test_user", "request_id": "test_req",
        "home_currency": "USD", "current_balance": balance,
        "min_balance": min_balance, "requested_amount": requested,
        "request_date": request_date,
        "desired_completion_date": request_date + timedelta(days=deadline_offset_days),
        "payment_methods": payment_methods or ["full_payment", "partial_payment", "installments"],
        "allows_partial_payment": allows_partial,
        "max_installment_months": 6, "recurring": [],
        "future_committed": pd.DataFrame(columns=["settlement_date", "amount_home"]),
        "future_income": pd.DataFrame(columns=["settlement_date", "amount_home"]),
        "messages": pd.DataFrame(), "payment_options": pd.DataFrame(),
        "protected_categories": protected_categories or ["rent", "utilities", "groceries"],
        "stoppable_categories": stoppable_categories or [],
        "reducible_categories": reducible_categories or [],
        "request_text": "Test purchase",
    }


def test_affordable_now_sufficient_balance():
    ctx = _mock_ctx(balance=10000, min_balance=1000, requested=5000)
    result = determine_affordability(ctx, pd.DataFrame())
    assert result["affordability_status"] == "affordable_now"
    assert result["recommended_payment_method"] == "full_payment"
    assert result["amount_safe_to_pay"] == 5000


def test_min_balance_protected_when_tight():
    """Balance=6000, min=5500, requested=5000 -> only 500 buffer -> not affordable_now."""
    ctx = _mock_ctx(balance=6000, min_balance=5500, requested=5000)
    result = determine_affordability(ctx, pd.DataFrame())
    assert result["affordability_status"] != "affordable_now"


def test_safe_amount_never_breaches_min_balance():
    ctx = _mock_ctx(balance=5000, min_balance=2000, requested=8000)
    safe = compute_amount_safe_to_pay(ctx, date.today())
    assert ctx["current_balance"] - safe >= ctx["min_balance"] - 1


def test_zero_safe_when_balance_at_minimum():
    ctx = _mock_ctx(balance=1000, min_balance=1000, requested=500)
    safe = compute_amount_safe_to_pay(ctx, date.today())
    assert safe == 0.0


def test_not_affordable_when_below_minimum():
    ctx = _mock_ctx(balance=800, min_balance=1000, requested=500)
    result = determine_affordability(ctx, pd.DataFrame())
    assert result["affordability_status"] == "not_affordable"
    assert result["recommended_payment_method"] == "not_recommended"
    assert result["payment_plan"] == "none"


def test_zero_balance_not_recommended():
    """Balance below minimum - nothing can be paid."""
    ctx = _mock_ctx(balance=500, min_balance=1000, requested=200)
    result = determine_affordability(ctx, pd.DataFrame())
    assert result['recommended_payment_method'] == 'not_recommended'


def test_simulate_balance_decreases_with_outflow():
    today = date.today()
    flows = [(today, -1000.0)]
    snapshots = simulate_balance(5000, today, today + timedelta(days=1), flows)
    balances = [b for _, b in snapshots]
    assert min(balances) == 4000.0


def test_simulate_balance_increases_with_income():
    today = date.today()
    flows = [(today + timedelta(days=5), +2000.0)]
    snapshots = simulate_balance(3000, today, today + timedelta(days=10), flows)
    balances = [b for _, b in snapshots]
    assert max(balances) == 5000.0


def test_is_plan_safe_within_buffer():
    ctx = _mock_ctx(balance=10000, min_balance=1000)
    today = date.today()
    assert is_plan_safe([(today, 5000.0)], ctx, today + timedelta(days=90)) is True


def test_is_plan_unsafe_breaches_minimum():
    ctx = _mock_ctx(balance=3000, min_balance=2500)
    today = date.today()
    assert is_plan_safe([(today, 1000.0)], ctx, today + timedelta(days=90)) is False

