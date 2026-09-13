"""
main.py — Orchestrator: runs the full Affordra ⚡ agent pipeline.

CONCEPT (for learning):
  This is the "conductor" of the orchestra. It:
  1. Loads all data once (expensive I/O happens here, once)
  2. Processes each request through the pipeline
  3. Applies LLM messages interpretation + explanation
  4. Writes output.csv
  5. Logs to log.txt (required by AGENTS.md)

  PIPELINE DESIGN PATTERN: Each step is independent and testable.
  data_loader → financial_model → llm_agent → output
  
TOKEN BUDGET:
  - Image OCR: 16 images × ~500 tokens = ~8,000 tokens
  - Message interpretation: ~50 relevant messages × ~400 tokens = ~20,000 tokens  
  - Explanations: 250 requests × ~200 tokens = ~50,000 tokens
  - Total: ~78,000 tokens → well within free daily limit (1M tokens/day)
"""

import sys
import os
os.environ.setdefault('PYTHONUTF8', '1')
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import traceback

# Add code/ to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all_data, build_request_context
from financial_model import determine_affordability
from image_extractor import extract_all_image_amounts
from llm_agent import interpret_messages_for_context, generate_explanation, get_total_api_calls, _template_explanation
from evaluator import evaluate as run_evaluation_df

REPO_ROOT = Path(__file__).parent.parent
LOG_FILE = REPO_ROOT / "log.txt"
OUTPUT_FILE = REPO_ROOT / "output.csv"


def append_log(entry: str):
    """Append to log.txt (required by AGENTS.md §5)."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def log_session_start():
    ts = datetime.now(timezone.utc).isoformat()
    entry = f"""
## {ts} SESSION START

tool=Antigravity (Google DeepMind)
Repo Root: {REPO_ROOT}
Branch: main
Worktree: main
Parent Agent: none
Language: py
Time Remaining: ~4h (deadline 2026-09-13T18:00:00+05:30)
"""
    append_log(entry)


def log_turn(title: str, prompt: str, summary: str, actions: list):
    ts = datetime.now(timezone.utc).isoformat()
    actions_str = "\n".join(f"* {a}" for a in actions)
    entry = f"""
## {ts} {title}

User Prompt (verbatim, secrets redacted):
{prompt}

Agent Response Summary:
{summary}

Actions:
{actions_str}

Context:
tool=Antigravity (Google DeepMind)
branch=main
repo_root={REPO_ROOT}
worktree=main
parent_agent=none
"""
    append_log(entry)


def apply_message_overrides(ctx: dict, message_adjustments: dict) -> dict:
    """
    Apply LLM-extracted message facts to adjust the financial context.
    
    CONCEPT: Messages can contain facts that override CSV data.
    e.g., employer message says salary is now higher → update recurring salary forecast.
    """
    import copy
    if not message_adjustments:
        return ctx

    ctx = copy.copy(ctx)

    # Salary override
    if "salary_override" in message_adjustments:
        new_salary = float(message_adjustments["salary_override"])
        updated_recurring = []
        for rec in ctx["recurring"]:
            if rec["category"] == "salary" and rec["direction"] == "credit":
                rec = dict(rec)
                rec["avg_amount"] = new_salary
            updated_recurring.append(rec)
        ctx["recurring"] = updated_recurring

    # Salary date override — update future income
    if "salary_date_override" in message_adjustments:
        from datetime import date
        try:
            new_date = date.fromisoformat(message_adjustments["salary_date_override"])
            updated_recurring = []
            for rec in ctx["recurring"]:
                if rec["category"] == "salary" and rec["direction"] == "credit":
                    rec = dict(rec)
                    rec["last_date"] = new_date - __import__('datetime').timedelta(days=int(rec["avg_gap_days"]))
                updated_recurring.append(rec)
            ctx["recurring"] = updated_recurring
        except Exception:
            pass

    # Rent increase
    if "rent_increase_pct" in message_adjustments:
        pct = float(message_adjustments["rent_increase_pct"])
        updated_recurring = []
        for rec in ctx["recurring"]:
            if rec["category"] == "rent" and rec["direction"] == "debit":
                rec = dict(rec)
                rec["avg_amount"] = rec["avg_amount"] * (1 + pct)
            updated_recurring.append(rec)
        ctx["recurring"] = updated_recurring

    # Pending credit not available (e.g., freelance payout not yet withdrawable)
    if message_adjustments.get("pending_credit_not_available"):
        ctx["future_income"] = ctx["future_income"].iloc[0:0]  # Clear pending credits

    return ctx


def process_single_request(request_row, data, image_amounts: dict, use_llm: bool = True) -> dict:
    """
    Run the full pipeline for one request.
    Returns a dict with all output fields.
    """
    request_id = request_row["request_id"]

    try:
        # Step 1: Build context object
        ctx = build_request_context(request_row, data, image_amounts)

        # Step 2: Interpret messages (LLM) — only if there are relevant messages
        if use_llm and not ctx["messages"].empty:
            adjustments = interpret_messages_for_context(ctx["messages"], ctx)
            ctx = apply_message_overrides(ctx, adjustments)

        # Step 3: Run financial model (deterministic)
        decision = determine_affordability(ctx, ctx["payment_options"])

        # Step 4: Generate explanation (LLM or template)
        if use_llm:
            explanation = generate_explanation(ctx, decision)
        else:
            explanation = _template_explanation(ctx, decision)

        # Step 5: Format earliest_date_for_full_payment
        earliest = decision.get("earliest_date_for_full_payment")
        earliest_str = earliest.isoformat() if earliest else ""

        return {
            "request_id": request_id,
            "amount_safe_to_pay": decision.get("amount_safe_to_pay", 0),
            "affordability_status": decision.get("affordability_status", "not_affordable"),
            "recommended_payment_method": decision.get("recommended_payment_method", "not_recommended"),
            "payment_plan": decision.get("payment_plan", "none"),
            "earliest_date_for_full_payment": earliest_str,
            "spending_changes_needed": decision.get("spending_changes_needed", "none"),
            "decision_explanation": explanation,
        }

    except Exception as e:
        print(f"  ERROR processing {request_id}: {e}")
        traceback.print_exc()
        return {
            "request_id": request_id,
            "amount_safe_to_pay": 0,
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": "",
            "spending_changes_needed": "none",
            "decision_explanation": f"Unable to process request: {str(e)[:100]}",
        }


def main(use_llm: bool = False, run_eval: bool = True):
    """Main entry point."""
    print("=" * 60)
    print("AFFORDRA ⚡ — Autonomous AI Financial Decision Agent")
    print("=" * 60)

    log_session_start()

    # ── Step 1: Load all data ──────────────────────────────────────────────────
    print("\n[1/5] Loading datasets...")
    data = load_all_data()
    print(f"  Loaded {len(data['requests'])} requests, {len(data['events'])} events, "
          f"{len(data['profiles'])} profiles")

    # ── Step 2: Extract image amounts ──────────────────────────────────────────
    print("\n[2/5] Loading image amounts (from cache / Gemini Vision)...")
    image_amounts = extract_all_image_amounts(data["images"], data["events"])

    # ── Step 3: Process all requests ───────────────────────────────────────────
    print(f"\n[3/5] Processing {len(data['requests'])} requests...")
    results = []
    total = len(data["requests"])

    for i, (_, request_row) in enumerate(data["requests"].iterrows()):
        request_id = request_row["request_id"]
        print(f"  [{i+1}/{total}] {request_id} (user: {request_row['user_id']}, "
              f"amount: {request_row['requested_amount']:,.0f} {request_row['request_type']})")

        result = process_single_request(request_row, data, image_amounts, use_llm=use_llm)
        results.append(result)

        # Progress save every 25 requests
        if (i + 1) % 25 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
            print(f"  -> Checkpoint saved ({i+1}/{total})")

    # ── Step 4: Write output.csv ───────────────────────────────────────────────
    print("\n[4/5] Writing output.csv...")
    output_df = pd.DataFrame(results)

    # Ensure correct column order
    output_cols = [
        "request_id", "amount_safe_to_pay", "affordability_status",
        "recommended_payment_method", "payment_plan", "earliest_date_for_full_payment",
        "spending_changes_needed", "decision_explanation"
    ]
    output_df = output_df[output_cols]
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Written: {OUTPUT_FILE}")
    print(f"  Rows: {len(output_df)}")

    # -- Step 5: Evaluate against gold samples (run samples through pipeline too) ---
    if run_eval:
        print("\n[5/5] Running evaluation against gold samples...")
        # Run sample requests through the same pipeline
        sample_results = []
        for _, req_row in data["sample_requests"].iterrows():
            res = process_single_request(req_row, data, image_amounts, use_llm=False)
            sample_results.append(res)
        sample_output_df = pd.DataFrame(sample_results)
        run_evaluation_df(sample_output_df, data["sample_requests"])

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n[DONE] Total LLM API calls: {get_total_api_calls()}")
    print(f"   Output file: {OUTPUT_FILE}")

    # Log the run
    status_counts = output_df["affordability_status"].value_counts().to_dict()
    log_turn(
        title="Full pipeline run completed",
        prompt="Run the Affordra agent on all 250 requests",
        summary=(
            f"Ran full pipeline on {len(output_df)} requests. "
            f"Status distribution: {status_counts}. "
            f"Total LLM calls: {get_total_api_calls()}. "
            f"Output written to {OUTPUT_FILE}."
        ),
        actions=[
            f"Loaded {len(data['requests'])} requests from dataset/requests.csv",
            f"Processed {len(image_amounts)} image amounts",
            f"Generated {len(output_df)} output rows",
            f"Written output to {OUTPUT_FILE}",
        ],
    )

    return output_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Affordra ⚡ — AI Financial Decision Agent")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM calls (default: template explanations)")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation step")
    args = parser.parse_args()

    main(use_llm=args.use_llm, run_eval=not args.no_eval)
