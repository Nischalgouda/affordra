import sys
sys.path.insert(0, 'code')
from data_loader import load_all_data, build_request_context
from financial_model import determine_affordability

data = load_all_data()
samples = data['sample_requests']

correct_status = 0
correct_method = 0
total = len(samples)

for _, req in samples.iterrows():
    ctx = build_request_context(req, data)
    pred = determine_affordability(ctx, ctx.get('payment_options', []))
    gold_status = req['affordability_status']
    gold_method = req['recommended_payment_method']
    gold_safe = float(req['amount_safe_to_pay'])
    
    match_s = (pred['affordability_status'] == gold_status)
    match_m = (pred['recommended_payment_method'] == gold_method)
    if match_s: correct_status += 1
    if match_m: correct_method += 1
    status_icon = "MATCH" if match_s else "MISMATCH"
    method_icon = "MATCH" if match_m else "MISMATCH"
    print(f"{req['request_id']}: status={pred['affordability_status']} (gold={gold_status}, {status_icon}) | method={pred['recommended_payment_method']} (gold={gold_method}, {method_icon}) | amt={pred['amount_safe_to_pay']:.1f} vs gold={gold_safe:.1f}")

print(f"\nStatus accuracy: {correct_status}/{total} ({correct_status/total:.1%})")
print(f"Method accuracy: {correct_method}/{total} ({correct_method/total:.1%})")
