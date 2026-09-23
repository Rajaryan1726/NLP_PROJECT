"""Claim-level Precision / Recall / F1 of the verifier against your hand labels.

Reads eval/labeling/claims_to_label.csv (after you filled `human_supported` with 1/0).

Two views are reported:
1. Verifier quality (the fact-checker as a classifier). The positive class is an ERROR,
   i.e. a claim that is NOT supported. The system flags a claim when its verdict is anything
   other than "entailed".
     precision = flagged claims that really are errors / all flagged claims
     recall    = real errors that were flagged / all real errors
   The same numbers for the "supported" class and the macro average are also reported.
2. Summary factual precision per experiment/dataset: the share of claims humans accepted.

Usage:  python eval/score_labels.py
"""
import json

import pandas as pd

from common import LABEL_DIR, RESULTS_DIR


def prf(y_true, y_pred, positive):
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    fp = sum(t != positive and p == positive for t, p in zip(y_true, y_pred))
    fn = sum(t == positive and p != positive for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": tp + fn}


def main():
    path = LABEL_DIR / "claims_to_label.csv"
    if not path.exists():
        print("run eval/make_label_template.py first, then fill in human_supported")
        return
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["human_supported"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])].copy()
    if df.empty:
        print("no labels yet: fill the human_supported column with 1 (supported) or 0 (not supported)")
        return
    df["human"] = df["human_supported"].astype(float).astype(int).map({1: "supported", 0: "error"})
    df["system"] = df["system_verdict"].map(lambda v: "supported" if v == "entailed" else "error")

    error = prf(df.human, df.system, "error")
    supported = prf(df.human, df.system, "supported")
    macro = {k: round((error[k] + supported[k]) / 2, 4) for k in ("precision", "recall", "f1")}
    accuracy = round((df.human == df.system).mean(), 4)
    per_group = (df.assign(ok=df.human == "supported")
                   .groupby(["experiment", "dataset"])["ok"].agg(["mean", "count"])
                   .rename(columns={"mean": "human_factual_precision", "count": "claims"}).reset_index())

    report = {"labelled_claims": len(df), "summaries": df.summary_key.nunique(), "accuracy": accuracy,
              "error_class": error, "supported_class": supported, "macro": macro,
              "per_verdict": df.groupby("system_verdict")["human"].value_counts().unstack(fill_value=0).to_dict("index"),
              "human_factual_precision": per_group.to_dict("records")}
    (RESULTS_DIR / "label_scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    per_group.to_csv(RESULTS_DIR / "human_factual_precision.csv", index=False)

    print(f"{len(df)} labelled claims from {report['summaries']} summaries, accuracy {accuracy}")
    print(f"error class      P {error['precision']}  R {error['recall']}  F1 {error['f1']}  (n={error['support']})")
    print(f"supported class  P {supported['precision']}  R {supported['recall']}  F1 {supported['f1']}  (n={supported['support']})")
    print(f"macro            P {macro['precision']}  R {macro['recall']}  F1 {macro['f1']}")
    print(per_group.to_string(index=False))
    print(f"saved {RESULTS_DIR / 'label_scores.json'}")


if __name__ == "__main__":
    main()
