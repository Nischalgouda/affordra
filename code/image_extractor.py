"""
image_extractor.py — Extract monetary amounts from PNG images using Gemini Vision.

CONCEPT (for learning):
  16 of the financial events have a BLANK amount field in financial_events.csv.
  The problem statement says: "When a financial event has a blank amount, use its
  event_id to find the matching image in images.csv, then extract the amount from that image."

  This is a Vision Language Model (VLM) task — we send the image to Gemini and ask it
  to read the number from what is likely a pay stub, receipt, or bank statement image.

TOKEN USAGE: ~500 tokens per image x 16 images = ~8,000 tokens total. Very low.
"""

import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from google import genai
from google.genai import types

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATASET_DIR = Path(__file__).parent.parent / "dataset"
IMAGES_DIR = DATASET_DIR / "media" / "images"

CACHE_FILE = Path(__file__).parent / "image_cache.json"

# Load cache from disk if available
_cache = {}
if CACHE_FILE.exists():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            _cache = json.load(f)
    except Exception:
        _cache = {}

_client = None


def _save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_cache, f, indent=2)
    except Exception:
        pass


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def extract_amount_from_image(image_id: str, currency: str):
    """
    Send an image to Gemini Vision and extract the monetary amount.
    Returns: (amount_float, currency_str) or (None, None) if extraction fails.
    """
    if image_id in _cache:
        return _cache[image_id]

    if not GEMINI_API_KEY:
        print(f"  [image_extractor] No API key, skipping {image_id}")
        _cache[image_id] = (None, None)
        return None, None

    image_path = IMAGES_DIR / f"{image_id}.png"
    if not image_path.exists():
        print(f"  [image_extractor] Image not found: {image_path}")
        _cache[image_id] = (None, None)
        return None, None

    try:
        client = _get_client()

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        prompt = (
            f"You are a financial document parser. Extract the total amount from this financial document image. "
            f"The document is associated with a transaction in {currency} currency. "
            f"Return ONLY a JSON object: {{\"amount\": <number>, \"currency\": \"<3-letter-code>\"}} "
            f"Rules: amount must be a plain number (no commas, no symbols). "
            f"Return the TOTAL or GROSS amount. "
            f"If you cannot find an amount, return {{\"amount\": null, \"currency\": null}}"
        )

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ],
        )
        raw = response.text.strip()

        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            amount = data.get("amount")
            curr = data.get("currency") or currency
            if amount is not None:
                result = (float(amount), str(curr))
                _cache[image_id] = result
                _save_cache()
                return result

    except Exception as e:
        print(f"  [image_extractor] Error processing {image_id}: {e}")

    _cache[image_id] = (None, None)
    return None, None


def extract_all_image_amounts(images_df, events_df):
    """
    Process all images in images.csv and return a dict mapping event_id -> amount.
    """
    image_amounts = {}

    print(f"\n[image_extractor] Processing {len(images_df)} images...")

    for _, img_row in images_df.iterrows():
        image_id = img_row["image_id"]
        event_id = img_row.get("related_event_id")

        if not event_id or str(event_id).strip() == "" or str(event_id) == "nan":
            continue

        event_rows = events_df[events_df["event_id"] == event_id]
        if event_rows.empty:
            continue

        event_row = event_rows.iloc[0]
        currency = str(event_row["currency"])

        print(f"  Processing {image_id} for {event_id} (currency: {currency})...")
        amount, extracted_currency = extract_amount_from_image(image_id, currency)

        if amount is not None:
            image_amounts[event_id] = amount
            print(f"  -> Extracted: {amount} {extracted_currency}")
        else:
            print(f"  -> Extraction failed, will use 0 as fallback")
            image_amounts[event_id] = 0.0

    print(f"[image_extractor] Done. Extracted {len(image_amounts)} amounts.\n")
    return image_amounts
