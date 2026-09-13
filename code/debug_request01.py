import sys; sys.path.insert(0, 'code')
from data_loader import load_all_data, build_request_context
from financial_model import is_plan_safe, compute_amount_safe_to_pay, build_cash_flow_timeline, simulate_balance
from datetime import date, timedelta
data = load_all_data()
req = data['sample_requests'][data['sample_requests']['request_id'] == 'request_01'].iloc[0]
ctx = build_request_context(req, data)
start = ctx['request_date']
deadline = ctx['desired_completion_date']
forecast_end = deadline + timedelta(days=90)

print(f"Request date: {start}, Deadline: {deadline}, Forecast end: {forecast_end}")
print(f"Balance: {ctx['current_balance']}, Min: {ctx['min_balance']}")

# Simulate with full payment
flows = build_cash_flow_timeline(ctx, start, forecast_end)
snaps = simulate_balance(ctx['current_balance'], start, forecast_end, flows, [(start, 25256)])
min_bal = min(b for d, b in snaps)
unsafe = [(d, b) for d, b in snaps if b < ctx['min_balance']]
print(f"Minimum balance with full payment: {min_bal:.2f}")
print(f"Unsafe days: {unsafe[:5]}")

safe = is_plan_safe([(start, 25256)], ctx, forecast_end)
print(f"is_plan_safe result: {safe}")
amt = compute_amount_safe_to_pay(ctx, start)
print(f"compute_amount_safe: {amt}")
