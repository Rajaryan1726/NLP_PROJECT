"""Create the hand-labelling sheet for claim-level Precision / Recall / F1.

Picks ~30 summaries (mixed over experiments and both datasets) and writes one row per claim to
eval/labeling/claims_to_label.csv. The source articles go to eval/labeling/sources/<key>.txt.

How to label (open the CSV in Excel / Google Sheets):
  read the source file named in `source_file`, then fill `human_supported`:
    1 = the claim is fully supported by the source
    0 = the claim is wrong, partly wrong, or not in the source
  Leave `system_verdict` alone. Optional: write a reason in `notes`.
Then run:  python eval/score_labels.py

Usage:  python eval/make_label_template.py --n 30
"""
import argparse
import csv
import random

from common import DATASETS, EXPERIMENTS, LABEL_DIR, raw_path, read_jsonl


def main(n: int, seed: int):
    out = LABEL_DIR / "claims_to_label.csv"
    if out.exists():
        print(f"{out} already exists; delete it first if you really want a new sample (your labels would be lost)")
        return
    sources = {(ds, it["id"]): it["text"] for ds, path in DATASETS.items() for it in read_jsonl(path)}
    pool = [r for exp in EXPERIMENTS for ds in DATASETS for r in read_jsonl(raw_path(exp, ds)) if r["claims"]]
    if not pool:
        print("no results yet; run eval/run_experiments.py first")
        return
    random.Random(seed).shuffle(pool)
    # balance: take summaries round-robin over (experiment, dataset)
    groups = {}
    for r in pool:
        groups.setdefault((r["experiment"], r["dataset"]), []).append(r)
    picked = []
    while len(picked) < n and any(groups.values()):
        for key in list(groups):
            if groups[key] and len(picked) < n:
                picked.append(groups[key].pop())

    (LABEL_DIR / "sources").mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig so Excel shows Hindi correctly
        w = csv.writer(f)
        w.writerow(["summary_key", "experiment", "dataset", "claim_no", "claim_text", "claim_en",
                    "evidence_text", "system_verdict", "human_supported", "notes", "source_file", "summary"])
        for r in picked:
            key = f"{r['experiment']}__{r['dataset']}__{r['id']}"
            src_file = f"sources/{r['dataset']}__{r['id']}.txt"
            (LABEL_DIR / src_file).write_text(sources.get((r["dataset"], r["id"]), ""), encoding="utf-8")
            for i, c in enumerate(r["claims"], 1):
                w.writerow([key, r["experiment"], r["dataset"], i, c["claim_text"], c["claim_en"],
                            c["evidence_text"], c["verdict"], "", "", src_file, r["summary"] if i == 1 else ""])
    n_claims = sum(len(r["claims"]) for r in picked)
    print(f"wrote {out} with {len(picked)} summaries / {n_claims} claims; sources in {LABEL_DIR / 'sources'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    main(**vars(ap.parse_args()))
