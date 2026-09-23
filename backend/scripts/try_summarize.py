"""Call the running API from the terminal and print the streamed result.

Usage:  python scripts/try_summarize.py path/to/text.txt [--compare] [--length short] [--bullets]
"""
import argparse
import json
import sys

import httpx

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--compare", action="store_true")
ap.add_argument("--length", default="medium")
ap.add_argument("--bullets", action="store_true")
ap.add_argument("--session", default="cli-test")
args = ap.parse_args()

body = {"text": open(args.file, encoding="utf-8").read(), "length": args.length,
        "style": "bullets" if args.bullets else "paragraph", "compare": args.compare, "session_id": args.session}

with httpx.stream("POST", "http://localhost:8000/api/summarize", json=body, timeout=600) as r:
    if r.status_code != 200:
        print(r.status_code, r.read().decode())
        sys.exit(1)
    event = None
    for line in r.iter_lines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            d = json.loads(line[6:])
            if event in ("status", "attempt", "error"):
                print(f"[{event}]", d)
            elif event == "result":
                print("\nSUMMARY:", d["summary"])
                print(f"score {d['factuality_score']} (claims {d['claim_score']}, entities {d['entity_score']}), "
                      f"attempts {d['attempts']}, low_confidence {d['low_confidence']}, stats {d['stats']}")
                for c in d["claims"]:
                    print(f"  {c['verdict']:<12} nli={c['nli_confidence']:.2f} rr={c['rerank_score']:.2f} | {c['claim_text']}")
                for m in d["entity_mismatches"]:
                    print("  MISMATCH:", m["reason"])
            elif event == "compare_result":
                s = d["standard"]
                print(f"\nSTANDARD ({s['factuality_score']}):", s["summary"])
                print("factuality-aware first attempt score:", d["factuality_aware"]["first_attempt_score"])
