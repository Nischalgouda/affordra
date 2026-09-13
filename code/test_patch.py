import sys
sys.path.insert(0, 'code')
import financial_model
from datetime import timedelta

def patched_build(ctx, start_date, end_date, essential_only=True):
    flows = []
    protected = set(ctx.get('protected_categories', []))
    stoppable = set(ctx.get('stoppable_categories', []))
    reducible = set(ctx.get('reducible_categories', []))
    non_essential = stoppable | reducible

    for _, row in ctx['future_committed'].iterrows():
        sd = row['settlement_date']
        if sd is None or (hasattr(sd, '__class__') and 'NaT' in str(type(sd))):
            continue
        if start_date <= sd <= end_date:
            flows.append((sd, -float(row['amount_home'])))

    pending_income_dates = set()
    for _, row in ctx['future_income'].iterrows():
        sd = row['settlement_date']
        if sd is None or (hasattr(sd, '__class__') and 'NaT' in str(type(sd))):
            continue
        if start_date <= sd <= end_date:
            flows.append((sd, +float(row['amount_home'])))
            pending_income_dates.add(sd)

    for rec in ctx['recurring']:
        direction = rec['direction']
        if essential_only and direction == 'debit' and rec['category'] in non_essential:
            continue
        last = rec['last_date']
        gap = max(int(round(rec['avg_gap_days'])), 1)
        amount = rec['avg_amount']
        next_date = last + timedelta(days=gap)
        iterations = 0
        while next_date <= end_date and iterations < 500:
            if next_date >= start_date:
                skip = False
                if direction == 'credit':
                    for pid in pending_income_dates:
                        if abs((next_date - pid).days) <= 15:
                            skip = True
                            break
                if not skip:
                    delta = +amount if direction == 'credit' else -amount
                    flows.append((next_date, delta))
            next_date += timedelta(days=gap)
            iterations += 1

    flows.sort(key=lambda x: x[0])
    return flows

financial_model.build_cash_flow_timeline = patched_build

from data_loader import load_all_data, build_request_context
from financial_model import determine_affordability
data = load_all_data()
samples = data['sample_requests']

correct_s, correct_m = 0, 0
for _, req in samples.iterrows():
    ctx = build_request_context(req, data)
    pred = determine_affordability(ctx, ctx.get('payment_options', []))
    gold_s = req['affordability_status']
    gold_m = req['recommended_payment_method']
    if pred['affordability_status'] == gold_s: correct_s += 1
    if pred['recommended_payment_method'] == gold_m: correct_m += 1
    s_tag = 'OK' if pred['affordability_status'] == gold_s else 'FAIL'
    m_tag = 'OK' if pred['recommended_payment_method'] == gold_m else 'FAIL'
    print(f"{req['request_id']}: [{s_tag}] {pred['affordability_status']} (gold={gold_s}) | [{m_tag}] {pred['recommended_payment_method']} (gold={gold_m})")

print(f"Status: {correct_s}/{len(samples)} ({correct_s/len(samples):.1%}), Method: {correct_m}/{len(samples)} ({correct_m/len(samples):.1%})")
