"""
Cabin Bag Checker — Automated Airline Data Updater

Fetches each airline's baggage policy page, uses an LLM to extract current
cabin bag dimensions, compares with stored data, and updates airlines.json.

Usage:
    python scripts/update_airlines.py                       # dry-run (print changes only)
    python scripts/update_airlines.py --apply               # write changes to airlines.json
    python scripts/update_airlines.py --apply --rebuild      # also regenerate index.html

Requires:
    pip install -r requirements.txt

Environment variables:
    GOOGLE_API_KEY   — Gemini API key  (preferred, free tier available)
        OR
    OPENAI_API_KEY   — OpenAI API key  (fallback)
"""

import json, os, sys, time, re, argparse, hashlib
from pathlib import Path
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "airlines.json"
CHANGELOG = ROOT / "CHANGELOG.md"

# ── Config ─────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 20          # seconds per HTTP request
DELAY_BETWEEN   = 2           # seconds between airline fetches (be polite)
MAX_PAGE_CHARS  = 12_000      # truncate page text sent to LLM

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ── LLM extraction prompt ─────────────────────────────────────────────
EXTRACTION_PROMPT = """You are a data-extraction assistant. I will give you text from an airline's cabin/carry-on baggage policy page.

Extract ONLY the **main cabin carry-on bag** (the overhead bin bag) dimensions and weight limit.
Do NOT extract the personal item / underseat bag dimensions.
Do NOT extract checked baggage dimensions.

Return EXACTLY this JSON (no markdown, no explanation):
{{
  "length_cm": <int>,
  "width_cm": <int>,
  "height_cm": <int>,
  "weight_kg": <number>,
  "note": "<any important caveat, or empty string>"
}}

Rules:
- If dimensions are in inches, convert to centimetres (1 in = 2.54 cm) and round to nearest integer.
- If weight is in pounds, convert to kg (1 lb = 0.4536 kg) and round to 1 decimal.
- If you cannot find clear cabin bag dimensions, return: {{"error": "not found"}}
- length = the longest side, height = the shortest side.

Airline: {airline_name}
Page text:
---
{page_text}
---"""


# ═══════════════════════════════════════════════════════════════════════
# 1. Fetch & clean page text
# ═══════════════════════════════════════════════════════════════════════

def fetch_page_text(url: str) -> str | None:
    """Fetch a URL and return visible text, or None on failure."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        print(f"    ⚠ fetch failed: {e}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    # Remove scripts, styles, nav, footer
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "img"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:MAX_PAGE_CHARS]


# ═══════════════════════════════════════════════════════════════════════
# 2. LLM extraction (Gemini or OpenAI)
# ═══════════════════════════════════════════════════════════════════════

def extract_with_gemini(airline_name: str, page_text: str) -> dict | None:
    """Use Google Gemini API to extract bag limits."""
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = EXTRACTION_PROMPT.format(airline_name=airline_name, page_text=page_text)
    resp = model.generate_content(prompt)
    return _parse_llm_json(resp.text)


def extract_with_openai(airline_name: str, page_text: str) -> dict | None:
    """Use OpenAI API to extract bag limits."""
    from openai import OpenAI
    client = OpenAI()
    prompt = EXTRACTION_PROMPT.format(airline_name=airline_name, page_text=page_text)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return _parse_llm_json(resp.choices[0].message.content)


def _parse_llm_json(raw: str) -> dict | None:
    """Extract JSON object from LLM response text."""
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        # Try to find JSON in the text
        m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def extract_bag_limits(airline_name: str, page_text: str) -> dict | None:
    """Try Gemini first, fall back to OpenAI."""
    if os.environ.get("GOOGLE_API_KEY"):
        print("    → Using Gemini")
        return extract_with_gemini(airline_name, page_text)
    elif os.environ.get("OPENAI_API_KEY"):
        print("    → Using OpenAI")
        return extract_with_openai(airline_name, page_text)
    else:
        print("    ✗ No LLM API key set (GOOGLE_API_KEY or OPENAI_API_KEY)")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# 3. Compare & update
# ═══════════════════════════════════════════════════════════════════════

def dims_differ(old: dict, new: dict) -> bool:
    """Check if extracted dimensions differ from stored ones."""
    for key in ("length_cm", "width_cm", "height_cm", "weight_kg"):
        if key in new and old.get(key) != new[key]:
            return True
    return False


def format_dims(d: dict) -> str:
    return f"{d.get('length_cm','?')}×{d.get('width_cm','?')}×{d.get('height_cm','?')} cm, {d.get('weight_kg','?')} kg"


# ═══════════════════════════════════════════════════════════════════════
# 4. Rebuild index.html from data
# ═══════════════════════════════════════════════════════════════════════

def rebuild_html(data: dict):
    """Regenerate the AIRLINES array inside index.html from airlines.json."""
    html_path = ROOT / "index.html"
    if not html_path.exists():
        print("  ⚠ index.html not found, skipping rebuild")
        return

    html = html_path.read_text(encoding="utf-8")

    # Build the JS array
    lines = []
    for a in data["airlines"]:
        b = a["cabin_bag"]
        aliases_js = json.dumps(a["aliases"])
        note_escaped = a.get("note", "").replace('"', '\\"')
        lines.append(
            f'  ["{a["name"]}", {b["length_cm"]}, {b["width_cm"]}, {b["height_cm"]}, '
            f'{b["weight_kg"]}, "{note_escaped}", {aliases_js}],'
        )
    js_array = "const AIRLINES = [\n" + "\n".join(lines) + "\n];"

    # Replace the existing AIRLINES array in the HTML
    pattern = r"const AIRLINES\s*=\s*\[[\s\S]*?\];"
    if not re.search(pattern, html):
        print("  ⚠ Could not find AIRLINES array in index.html")
        return

    html_new = re.sub(pattern, js_array, html, count=1)
    html_path.write_text(html_new, encoding="utf-8")
    print(f"  ✓ index.html rebuilt with {len(data['airlines'])} airlines")


# ═══════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Update airline cabin bag data")
    parser.add_argument("--apply", action="store_true", help="Write changes to airlines.json")
    parser.add_argument("--rebuild", action="store_true", help="Also regenerate index.html")
    parser.add_argument("--airline", type=str, help="Update only this airline (by name)")
    args = parser.parse_args()

    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    airlines = data["airlines"]
    changes = []
    errors = []

    print(f"{'='*60}")
    print(f"  Cabin Bag Checker — Data Updater")
    print(f"  {len(airlines)} airlines to check")
    print(f"  Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"{'='*60}\n")

    for i, airline in enumerate(airlines):
        name = airline["name"]

        # Skip if --airline flag is set and doesn't match
        if args.airline and args.airline.lower() not in [name.lower()] + airline["aliases"]:
            continue

        url = airline.get("source_url", "")
        print(f"[{i+1}/{len(airlines)}] {name}")
        if not url:
            print("    ⚠ No source URL, skipping")
            errors.append((name, "no source URL"))
            continue

        # Fetch page
        page_text = fetch_page_text(url)
        if not page_text:
            errors.append((name, "fetch failed"))
            continue

        # Extract with LLM
        extracted = extract_bag_limits(name, page_text)
        if not extracted or "error" in extracted:
            print(f"    ⚠ LLM could not extract data: {extracted}")
            errors.append((name, "extraction failed"))
            time.sleep(DELAY_BETWEEN)
            continue

        # Compare
        old = airline["cabin_bag"]
        if dims_differ(old, extracted):
            print(f"    🔄 CHANGED:  {format_dims(old)}  →  {format_dims(extracted)}")
            changes.append({
                "airline": name,
                "old": dict(old),
                "new": {k: extracted[k] for k in ("length_cm","width_cm","height_cm","weight_kg") if k in extracted},
            })
            if args.apply:
                for key in ("length_cm", "width_cm", "height_cm", "weight_kg"):
                    if key in extracted:
                        airline["cabin_bag"][key] = extracted[key]
                # Update note if LLM provided one and it's non-empty
                if extracted.get("note"):
                    airline["note"] = extracted["note"]
        else:
            print(f"    ✓ No change  ({format_dims(old)})")

        time.sleep(DELAY_BETWEEN)

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Airlines checked:   {len(airlines) - len(errors)}")
    print(f"  Changes found:      {len(changes)}")
    print(f"  Errors/skipped:     {len(errors)}")

    if changes:
        print(f"\n  Changes:")
        for c in changes:
            print(f"    • {c['airline']}:  {format_dims(c['old'])}  →  {format_dims(c['new'])}")

    if errors:
        print(f"\n  Errors:")
        for name, reason in errors:
            print(f"    • {name}: {reason}")

    # ── Write ──────────────────────────────────────────────────────────
    if args.apply and changes:
        data["meta"]["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        data["meta"]["updated_by"] = "update_airlines.py"
        DATA_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  ✓ airlines.json updated")

        # Append to changelog
        with open(CHANGELOG, "a", encoding="utf-8") as f:
            f.write(f"\n## {data['meta']['last_updated']}\n\n")
            for c in changes:
                f.write(f"- **{c['airline']}**: {format_dims(c['old'])} → {format_dims(c['new'])}\n")
            f.write("\n")
        print(f"  ✓ CHANGELOG.md updated")

        if args.rebuild:
            rebuild_html(data)

    elif not changes:
        print(f"\n  ✓ Everything is up to date — no changes needed.")

    else:
        print(f"\n  ℹ Dry run complete. Run with --apply to write changes.")


if __name__ == "__main__":
    main()
