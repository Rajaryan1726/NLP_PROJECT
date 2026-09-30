"""End-to-end test of the running summarizer (the backend must be running on port 8000).

Sends the demo texts in demo_texts/ to POST /api/summarize and checks each result:
  - the summary is in Roman script (no Devanagari)
  - the important numbers / names from the source are still there
  - negations were not dropped
  - the factuality score reaches the threshold (7/10)
  - claims were verified and the text was compressed
It also checks the input validation (empty text and >5,000 words must give HTTP 400).

Usage (from backend/):
  .venv\\Scripts\\python scripts\\test_summarizer.py            # everything
  .venv\\Scripts\\python scripts\\test_summarizer.py --only hindi code_mixed
  npm run test:summarizer
Exit code 0 = all checks passed.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8")
DEMO = Path(__file__).resolve().parents[2] / "demo_texts"
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# name -> (file, words that must appear in the summary, check negations?)
CASES = {
    "hindi": ("1_hindi_news.txt", ["Vidya Setu", "4,500", "2 crore"], False),
    "code_mixed": ("2_code_mixed.txt", ["Vidya Setu", "4,500", "2 crore"], False),
    "negation": ("3_numbers_and_negation.txt", ["212", "14"], True),
    "noisy": ("4_noisy_code_switched.txt", ["India", "Australia", "6"], False),
    "english": ("5_english.txt", ["6.5"], False),
}
NEGATION_WORDS = ("nahi", "nahin", "not", "no ", "chalu", "khule")

# Negated facts in 3_numbers_and_negation.txt: (topic, words that would mean the negation was flipped).
# A summary may leave a fact out, but if it mentions the topic it must not say the opposite.
NEGATED_FACTS = [
    ("western railway", ("band", "bandh", "ruk")),   # source: services band NAHI hui
    ("offices", ("band", "bandh")),                  # source: offices khule rahenge
    ("red alert", ("jaari", "diya", "issue")),       # source: orange alert, red alert NAHI
]


def flipped_negations(summary: str) -> tuple[list[str], list[str]]:
    """Return (flipped facts, omitted facts) by looking at the sentence that mentions each topic."""
    sentences = re.split(r"(?<=[.!?])\s+|,\s*|\s+(?:aur|lekin|par)\s+", summary.lower())
    flipped, omitted = [], []
    for topic, opposite in NEGATED_FACTS:
        parts = [s for s in sentences if topic in s]
        if not parts:
            omitted.append(topic)
        elif any(any(w in s for w in opposite) and not any(n in s for n in NEGATION_WORDS) for s in parts):
            flipped.append(topic)
    return flipped, omitted


def summarize(base: str, text: str, length: str = "medium") -> tuple[int, dict | None, str]:
    """Call the API and return (http status, result event or None, error message)."""
    body = {"text": text, "length": length, "style": "paragraph", "compare": False, "session_id": "test-script"}
    with httpx.stream("POST", f"{base}/api/summarize", json=body, timeout=300) as r:
        if r.status_code != 200:
            return r.status_code, None, json.loads(r.read() or b"{}").get("detail", "")
        event, result, error = None, None, ""
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "result":
                    result = data
                elif event == "error":
                    error = data["message"]
        return 200, result, error


class Report:
    def __init__(self):
        self.passed, self.failed = 0, 0

    def check(self, ok: bool, label: str, detail: str = ""):
        if ok:
            self.passed += 1
            print(f"   PASS  {label}")
        else:
            self.failed += 1
            print(f"   FAIL  {label}  {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--only", nargs="+", choices=list(CASES), help="run only these cases")
    args = ap.parse_args()
    base = args.base_url

    try:
        health = httpx.get(f"{base}/api/health", timeout=5).json()
    except httpx.HTTPError:
        print(f"Backend is not running at {base}. Start it first:  cd backend; npm run dev")
        return 2
    threshold = health.get("factuality_threshold", 7)
    print(f"Backend: {health['status']}  |  model: {health['llm']['model']}  |  threshold: {threshold}\n")
    rep = Report()

    print("[validation]")
    status, _, msg = summarize(base, "   ")
    rep.check(status == 400, "empty text is rejected with 400", f"(got {status})")
    status, _, msg = summarize(base, "shabd " * 5200)
    rep.check(status == 400 and "5,000" in msg, "text over 5,000 words is rejected with 400", f"(got {status}: {msg})")

    for name in args.only or CASES:
        file, must_have, negation = CASES[name]
        text = (DEMO / file).read_text(encoding="utf-8")
        print(f"\n[{name}]  {file}")
        t = time.perf_counter()
        status, r, error = summarize(base, text)
        if error or not r:
            rep.check(False, "summary generated", f"({error or status})")
            if "authentication" in error.lower() or "api key" in error.lower():
                print("\nThe OpenAI key in backend/.env is rejected (expired or wrong). Put a valid key there,"
                      " restart the backend and run this script again.")
                break
            continue
        summary = r["summary"]
        print(f"   summary ({time.perf_counter() - t:.1f}s): {summary}")
        rep.check(not DEVANAGARI.search(summary), "summary is Roman-script Hinglish (no Devanagari)")
        for word in must_have:
            rep.check(word.lower() in summary.lower(), f"keeps '{word}'")
        if negation:
            flipped, omitted = flipped_negations(summary)
            rep.check(not flipped, "no negation flipped", f"(flipped: {', '.join(flipped)})")
            if omitted:
                print(f"   note: left out (allowed in a short summary): {', '.join(omitted)}")
        rep.check(r["factuality_score"] >= threshold, f"factuality score {r['factuality_score']} >= {threshold}")
        rep.check(len(r["claims"]) > 0, f"claims verified ({len(r['claims'])} claims, "
                  f"{sum(c['verdict'] == 'entailed' for c in r['claims'])} supported)")
        rep.check(r["stats"]["compression_ratio"] > 0, f"text compressed ({r['stats']['compression_ratio']:.0%})")
        for m in r["entity_mismatches"]:
            print(f"   note: {m['reason']}")

    print(f"\nResult: {rep.passed} passed, {rep.failed} failed")
    return 0 if rep.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
