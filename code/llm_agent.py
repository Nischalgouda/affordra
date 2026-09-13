"""
llm_agent.py — Gemini LLM integration for decision explanations + message analysis.

CONCEPT (for learning):
  The LLM has TWO jobs here:

  1. INTERPRET MESSAGES: Messages from employers, banks, service providers may contain
     facts that override what's in the CSV. We extract structured facts from them.
     e.g., "Your salary is reduced to EUR 1,422.85" -> update the salary forecast.

  2. GENERATE EXPLANATION: Produce the human-readable decision_explanation field.
     All numbers come from the deterministic model -- the LLM just writes the sentence.

TOKEN OPTIMIZATION STRATEGIES:
  - Use gemini-3.6-flash (cheapest, fastest model)
  - Compress context to only what's relevant (not the full CSVs)
  - Small focused prompts (~300 tokens each)
  - Template fallback if API fails
  - Rate limiting: 4 seconds between calls (free tier: 15 RPM)
"""

import os
import time
import json
import re
from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

_client = None
_call_count = 0
_last_call_time = 0
RATE_LIMIT_DELAY = 13.0  # seconds between calls (free tier: 5 RPM = 12s + buffer)


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _rate_limited_call(prompt: str) -> str:
    """Make a rate-limited API call to Gemini."""
    global _call_count, _last_call_time

    elapsed = time.time() - _last_call_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)

    try:
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        _call_count += 1
        _last_call_time = time.time()
        return response.text.strip()
    except Exception as e:
        print(f"  [llm_agent] API error: {e}")
        time.sleep(10)
        return ""


def interpret_messages_for_context(messages_df, ctx: dict) -> dict:
    """
    Read messages related to this user/request and extract financial facts
    that should override or supplement the financial model's assumptions.

    Returns a dict of adjustments, e.g.:
      {"salary_override": 42750000, "pending_credit_not_available": true}
    """
    if messages_df.empty:
        return {}

    msgs = messages_df[
        (messages_df["user_id"] == ctx["user_id"]) |
        (messages_df.get("request_id", messages_df["user_id"]) == ctx["request_id"])
    ]

    if msgs.empty:
        return {}

    msg_texts = []
    for _, row in msgs.iterrows():
        msg_texts.append(f"[{row.get('source_type', 'unknown')}] {row.get('message_text', '')}")

    combined = "\n".join(msg_texts[:5])  # Limit to 5 messages for token efficiency

    prompt = f"""You are a financial data extractor. Read these messages and extract ONLY factual financial updates.

Messages:
{combined}

User home currency: {ctx['home_currency']}

Extract any of these facts IF explicitly stated (do NOT infer or guess):
- salary_override: new confirmed salary amount (number, in home currency)
- salary_date_override: new confirmed salary payment date (YYYY-MM-DD)
- rent_increase_pct: rent increase as decimal (0.12 for 12%)
- pending_credit_not_available: true if a pending income is explicitly NOT yet available

Return ONLY a JSON object. If nothing found, return {{}}.
Example: {{"salary_override": 42750000}}"""

    raw = _rate_limited_call(prompt)
    if not raw:
        return {}

    try:
        json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass

    return {}


def generate_explanation(ctx: dict, decision: dict) -> str:
    """
    Generate the decision_explanation field using Gemini.
    All numbers are passed in -- the LLM only writes prose.
    """
    currency = ctx["home_currency"]
    amount = decision.get("amount_safe_to_pay", 0)
    status = decision.get("affordability_status", "")
    method = decision.get("recommended_payment_method", "")
    plan = decision.get("payment_plan", "none")
    changes = decision.get("spending_changes_needed", "none")
    min_bal = ctx["min_balance"]
    request_text = str(ctx.get("request_text", ""))[:150]

    prompt = f"""Write a concise 1-2 sentence financial recommendation.

Request: "{request_text}"
Currency: {currency}
Requested: {ctx['requested_amount']:,.2f}
Safe to pay now: {amount:,.2f} {currency}
Decision: {status} -> {method}
Payment plan: {plan}
Spending changes: {changes}
Min balance: {min_bal:,.2f} {currency}

Rules:
- Direct and specific (include actual amounts and dates from the plan)
- No markdown, no bullets
- Under 50 words
- Match the language of the user's request if not English"""

    explanation = _rate_limited_call(prompt)

    if not explanation:
        return _template_explanation(ctx, decision)

    return explanation


def _template_explanation(ctx: dict, decision: dict) -> str:
    """Fallback template-based explanation (no LLM, zero tokens)."""
    currency = ctx["home_currency"]
    requested = ctx["requested_amount"]
    amount = decision.get("amount_safe_to_pay", 0)
    status = decision.get("affordability_status", "")
    method = decision.get("recommended_payment_method", "")
    min_bal = ctx["min_balance"]
    earliest = decision.get("earliest_date_for_full_payment")

    if status == "affordable_now":
        return (f"Pay {currency} {requested:,.2f} today. "
                f"This leaves at least {currency} {min_bal:,.2f} available.")
    elif status == "affordable_with_plan":
        if method == "installments":
            return (f"Use the installment plan. "
                    f"This keeps at least {currency} {min_bal:,.2f} available throughout.")
        elif method == "partial_payment":
            return (f"Pay {currency} {amount:,.2f} today and the remainder when funds allow. "
                    f"This keeps the {currency} {min_bal:,.2f} minimum protected.")
        elif method == "wait":
            date_str = earliest.isoformat() if earliest else "a future date"
            return (f"Wait until {date_str}, then pay {currency} {requested:,.2f} in full. "
                    f"Paying earlier would put the {currency} {min_bal:,.2f} minimum at risk.")
        elif method == "full_payment":
            return (f"Stop flexible expenses, then pay {currency} {requested:,.2f}. "
                    f"This keeps at least {currency} {min_bal:,.2f} available.")
    elif status == "affordable_later":
        date_str = earliest.isoformat() if earliest else "a future date"
        return (f"Wait until {date_str}, then pay {currency} {requested:,.2f} in full. "
                f"Paying sooner would put the {currency} {min_bal:,.2f} minimum at risk.")
    else:
        return (f"Do not make this payment. "
                f"None of the available options keeps the {currency} {min_bal:,.2f} minimum protected.")


def get_total_api_calls() -> int:
    return _call_count
