# Affordra ⚡ — Autonomous AI Financial Decision & Affordability Agent

> **Know before you spend.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20VLM%20%2B%20Deterministic-green.svg)](#architecture)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Affordra** is an intelligent, autonomous financial decision agent that helps users make safe, personalized spending and purchasing decisions. Given an expense or purchase query, Affordra models 90 days of cash flow to determine whether to pay in full today, structure an optimal installment plan, wait for funds to settle, or hold off entirely.

Instead of relying solely on probabilistic LLMs—which suffer from arithmetic hallucinations and token bloat—Affordra employs a **Hybrid AI Architecture**: a **Vision-Language Model (Gemini Vision)** for parsing unstructured document receipts, combined with a **deterministic 90-day cash flow simulation engine** for mathematical precision.

---

## Key Features

- **Multimodal Document Parsing**: Automatically extracts transaction details, taxes, and amounts from raw invoice and receipt images using Gemini 3.6 Flash with local persistent caching.
- **Multi-Currency Support**: Ingests and normalizes transactions across USD, EUR, INR, IDR, and ZAR using fixed dated exchange rates.
- **90-Day Cash Flow Simulation**: Projects future balances day-by-day, accounting for recurring income cadences, essential commitments (rent, utilities, groceries), and user-defined minimum balance buffers.
- **Preference-Aware Decision Engine**: Evaluates payment avenues (`full_payment`, `partial_payment`, `installments`, `wait`) strictly adhering to user preferences, merchant options, and completion deadlines.
- **Sub-10s Execution**: Decoupled batch pipeline that processes 250 complex financial requests in under **10 seconds** with **$0 token inference cost**.

---

## Architecture

```
                    ┌────── Raw User Request & Financial Profile ──────┐
                    │                                                  │
           Unstructured Receipt?                               Has Messages?
              │                │                                 │         │
             Yes               No                               Yes        No
              │                │                                 │         │
              ▼                ▼                                 ▼         ▼
        Gemini Vision     Use Recorded CSV                 Entity Parser  Skip
       (VLM Extraction)        Amount                            │
              │                │                                 │
              └────────┬───────┘                                 │
                       │                                         │
                       └───────────────────┬─────────────────────┘
                                           ▼
                           Deterministic Simulation Engine
                        • 90-day forward balance projection
                        • Protection of essential spending categories
                        • Dynamic installment & payment plan solver
                                           │
                                           ▼
                        Grounded Decision & Output Generator
                   (affordability status, payment plan, explanation)
```

---

## Benchmark Performance

Evaluated against the ground-truth financial evaluation benchmark:

| Metric | Score | Notes |
|---|---|---|
| **Payment Method Accuracy** | **80.0%** | Accurately identifies full payment, installments, wait, or rejection |
| **Affordability Status Accuracy** | **72.0%** | Correctly classifies `affordable_now`, `affordable_with_plan`, etc. |
| **Pipeline Latency** | **~9.5s** | Evaluates all 250 requests end-to-end |
| **Inference Token Cost** | **$0.00** | Idempotent image cache + zero-token simulation engine |

---

## Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/<your-username>/buy-or-wait-ai-financial-agent.git
cd buy-or-wait-ai-financial-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r code/requirements.txt
```

### 2. Configuration (Optional)

If re-extracting new receipts from Gemini Vision:
```bash
cp .env.example .env
# Add your GEMINI_API_KEY inside .env
```
*(All 16 reference document amounts are pre-cached in `code/image_cache.json` for offline reproducibility).*

### 3. Run the Agent

To process all requests and generate `output.csv`:
```bash
python run.py
```

To run the evaluation benchmark:
```bash
python code/eval_samples.py
```

---

## Project Structure

```text
.
├── run.py                      # Root CLI entry point (delegates to code/main.py)
├── .env.example                # Sample environment configuration
├── .gitignore                  # Production exclusion rules
├── README.md                   # Project documentation
├── output.csv                  # Generated decision predictions (250 rows)
├── dataset/                    # Evaluation requests, profiles, events, and media
└── code/
    ├── main.py                 # Core pipeline orchestrator
    ├── data_loader.py          # Data ingestion, multi-currency normalization
    ├── financial_model.py      # 90-day cash flow simulation & decision tree
    ├── image_extractor.py      # Gemini Vision document entity extractor
    ├── image_cache.json        # Persistent disk cache for document amounts
    ├── llm_agent.py            # Natural language explanation generator
    ├── evaluator.py            # Benchmark scoring module
    └── requirements.txt        # Python package dependencies
```

---

## Output Schema

For each request, the agent outputs:
- `request_id`: Unique identifier
- `amount_safe_to_pay`: Maximum safe payment amount on evaluation date
- `affordability_status`: `affordable_now` | `affordable_with_plan` | `affordable_later` | `not_affordable`
- `recommended_payment_method`: `full_payment` | `partial_payment` | `installments` | `wait` | `not_recommended`
- `payment_plan`: Chronological dates and amounts (`YYYY-MM-DD:amount|...`) or `none`
- `earliest_date_for_full_payment`: First safe date for full lump-sum payment
- `spending_changes_needed`: Specific discretionary expenses to halt or reduce
- `decision_explanation`: Concise, grounded justification based on balance thresholds

---

## 🗺️ Future Roadmap

- [ ] **Interactive Web UI**: Streamlit / Next.js web application with live 90-day balance trajectory charts.
- [ ] **Receipt Dropzone**: Instant drag-and-drop receipt entity extraction and spending analysis.
- [ ] **OpenBanking Integration**: Real-time account balance and transaction syncing via Plaid API.
- [ ] **Proactive Notifications**: Webhook / SMS alerts when delayed purchases reach safe affordability windows.

---

## License

This project is licensed under the MIT License.
