"""Turn the raw experiment results into paper-ready tables and charts.

Outputs (eval/results/):
  summary_table.csv / .json / .tex   one row per (experiment, dataset)
  error_categories.csv               LLM-classified failed claims (cached)
  charts/*.png (300 dpi) + *.pdf     prompt comparison, Hindi vs code-mixed,
                                     score distribution, error categories, retry effect

Usage:  python eval/analyze.py            (add --no-llm to skip the error classification)

ROUGE caveat: XL-Sum references are Devanagari Hindi, our summaries are romanized Hinglish.
References are transliterated (ITRANS + simple Hindi schwa deletion) and both sides are
spelling-normalized before ROUGE, but spelling variation (e.g. "sarkaar" vs "sarkar") still
lowers the scores. Use ROUGE only to compare our systems with each other, not with
published XL-Sum numbers.
"""
import argparse
import asyncio
import json
import re
import statistics as st
from datetime import datetime, timezone

import common
from common import CHARTS_DIR, EXPERIMENTS, RESULTS_DIR, raw_path, read_jsonl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from indic_transliteration import sanscript  # noqa: E402
from rouge_score import rouge_scorer  # noqa: E402

import config  # noqa: E402
import db  # noqa: E402
import llm_service  # noqa: E402

DATASET_LABEL = {"hindi": "Hindi (Devanagari)", "code_mixed": "Code-mixed (synthetic)"}
EXP_LABEL = {"standard": "Standard", "factuality_aware": "Factuality-aware",
             "factuality_aware_retry": "Fact.-aware + retry"}
# validated categorical palette (blue, orange, aqua): fixed order, one colour per experiment
EXP_COLOR = {"standard": "#2a78d6", "factuality_aware": "#eb6834", "factuality_aware_retry": "#1baf7a"}
DATASET_COLOR = {"hindi": "#2a78d6", "code_mixed": "#eb6834"}
ERROR_CATEGORIES = ["number_mismatch", "entity_swap", "negation_drop", "added_info", "code_mix_misread", "no_error"]
ERROR_LABEL = {"number_mismatch": "Number mismatch", "entity_swap": "Entity swap", "negation_drop": "Negation drop",
               "added_info": "Added info", "code_mix_misread": "Code-mix misread", "no_error": "Verifier false alarm"}
SHORT_EXP_LABEL = {"standard": "Standard", "factuality_aware": "Fact.-aware", "factuality_aware_retry": "Fact.-aware\n+ retry"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 300, "savefig.bbox": "tight",
})


# ---------------------------------------------------------------- ROUGE with transliteration

def _normalize_roman(text: str) -> str:
    t = text.lower()
    for a, b in (("aa", "a"), ("ii", "i"), ("ee", "i"), ("oo", "u"), ("uu", "u"), ("w", "v"), ("z", "j")):
        t = t.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", t)  # collapse doubled letters


def romanize_devanagari(text: str) -> str:
    t = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
    words = re.findall(r"[a-z0-9]+", t)
    words = [re.sub(r"(?<=[bcdfghjklmnpqrstvxyz])a$", "", w) for w in words]  # schwa deletion at word end
    return " ".join(words)


_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def rouge(summary: str, reference: str) -> dict:
    ref = _normalize_roman(romanize_devanagari(reference))
    hyp = _normalize_roman(summary)
    s = _scorer.score(ref, hyp)
    return {k: v.fmeasure for k, v in s.items()}


# ---------------------------------------------------------------- metrics

def mean(xs):
    xs = list(xs)
    return round(st.mean(xs), 4) if xs else None


def summarize_rows(rows: list[dict]) -> dict:
    claims = [c for r in rows for c in r["claims"]]
    verdicts = [c["verdict"] for c in claims]
    facts = [f for r in rows for f in r["facts"] if f["type"] in ("number", "date", "entity")]
    num_facts = [f for f in facts if f["type"] in ("number", "date")]
    mism = [m for r in rows for m in r["entity_mismatches"] if m["type"] in ("number", "date", "entity")]
    timed = [r for r in rows if r["llm_cache_hits"] == 0]
    rouges = [rouge(r["summary"], r["reference"]) for r in rows if r.get("reference")]
    return {
        "n": len(rows),
        "factuality_score": mean(r["factuality_score"] for r in rows),
        "factuality_std": round(st.pstdev([r["factuality_score"] for r in rows]), 3) if rows else None,
        "claim_score": mean(r["claim_score"] for r in rows),
        "entity_score": mean(r["entity_score"] for r in rows),
        "claims_total": len(claims),
        "pct_supported": mean(v == "entailed" for v in verdicts),
        "pct_not_sure": mean(v == "neutral" for v in verdicts),
        "pct_contradicted": mean(v == "contradicted" for v in verdicts),
        "pct_unsupported": mean(v == "unsupported" for v in verdicts),
        "fact_mismatch_rate": round(len(mism) / len(facts), 4) if facts else None,
        "number_mismatch_rate": round(sum(1 for f in num_facts if not f["matched"]) / len(num_facts), 4) if num_facts else None,
        "pct_summaries_with_mismatch": mean(any(m["type"] in ("number", "date", "entity") for m in r["entity_mismatches"]) for r in rows),
        "avg_summary_words": mean(r["stats"]["words_out"] for r in rows),
        "avg_source_words": mean(r["stats"]["words_in"] for r in rows),
        "compression_ratio": mean(r["stats"]["compression_ratio"] for r in rows),
        "avg_inference_s": mean(r["stats"]["inference_time_ms"] / 1000 for r in timed),
        "n_timed": len(timed),
        "avg_attempts": mean(r["attempts"] for r in rows),
        "low_confidence_rate": mean(r["low_confidence"] for r in rows),
        "rouge1": mean(x["rouge1"] for x in rouges),
        "rouge2": mean(x["rouge2"] for x in rouges),
        "rougeL": mean(x["rougeL"] for x in rouges),
    }


def load_all() -> dict[tuple[str, str], list[dict]]:
    data = {}
    for exp in EXPERIMENTS:
        for ds in ("hindi", "code_mixed"):
            rows = read_jsonl(raw_path(exp, ds))
            if rows:
                data[(exp, ds)] = rows
    return data


def paired(data, a, b, ds):
    """Keep only items present in both runs so the comparison is on the same articles."""
    ids = {r["id"] for r in data.get((a, ds), [])} & {r["id"] for r in data.get((b, ds), [])}
    return ids


# ---------------------------------------------------------------- error classification

ERROR_PROMPT = """You analyse errors of a Hinglish summarizer. A fact-checker flagged a claim from the summary as
not supported by the source. Decide the error category:
- number_mismatch: a number, amount, date or quantity differs from the source
- entity_swap: a person, place or organisation is wrong or swapped
- negation_drop: a negation or polarity was lost or inverted
- added_info: the claim adds information that is not in the source
- code_mix_misread: a Hindi/English/code-mixed phrase in the source was misunderstood or mistranslated
- no_error: the claim is actually supported by the source (the fact-checker was too strict)
Return JSON: {"category": "<one of the above>", "explanation": "<one short sentence>"}"""


async def classify_errors(data) -> pd.DataFrame:
    sources = {}
    for ds in ("hindi", "code_mixed"):
        for item in read_jsonl(common.DATASETS[ds]):
            sources[(ds, item["id"])] = item["text"]
    llm_service.cache_namespace = "error_analysis"
    sem = asyncio.Semaphore(6)
    rows = []

    async def one(exp, ds, r, c):
        async with sem:
            msg = [{"role": "system", "content": ERROR_PROMPT},
                   {"role": "user", "content": f"SOURCE:\n{sources.get((ds, r['id']), '')[:6000]}\n\n"
                                               f"CLAIM: {c['claim_text']}\nCLAIM (English): {c['claim_en']}\n"
                                               f"FACT-CHECKER VERDICT: {c['verdict']}"}]
            try:
                out = await llm_service.complete_json(msg)
            except llm_service.LLMError:
                return
            cat = out.get("category", "no_error")
            rows.append({"experiment": exp, "dataset": ds, "id": r["id"], "claim": c["claim_text"],
                         "verdict": c["verdict"], "category": cat if cat in ERROR_CATEGORIES else "no_error",
                         "explanation": out.get("explanation", "")})

    tasks = [one(exp, ds, r, c) for (exp, ds), rs in data.items() for r in rs
             for c in r["claims"] if c["verdict"] != "entailed"]
    print(f"classifying {len(tasks)} failed claims…")
    await asyncio.gather(*tasks)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- charts

def _bar_labels(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color="#52514e")


def _save(fig, name):
    fig.savefig(CHARTS_DIR / f"{name}.png")
    fig.savefig(CHARTS_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  chart: results/charts/{name}.png")


def chart_prompt_comparison(table: pd.DataFrame):
    """Mean factuality score per experiment, grouped by dataset."""
    exps = [e for e in EXPERIMENTS if e in set(table.experiment)]
    dss = [d for d in ("hindi", "code_mixed") if d in set(table.dataset)]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    width = 0.8 / len(exps)
    for i, exp in enumerate(exps):
        vals = [table[(table.experiment == exp) & (table.dataset == d)].factuality_score.squeeze() for d in dss]
        xs = [j + (i - (len(exps) - 1) / 2) * width for j in range(len(dss))]
        bars = ax.bar(xs, vals, width * 0.92, color=EXP_COLOR[exp], label=EXP_LABEL[exp])
        _bar_labels(ax, bars)
    ax.set_xticks(range(len(dss)), [DATASET_LABEL[d] for d in dss])
    ax.set_ylabel("Mean factuality score (0-10)")
    ax.set_ylim(0, 11)
    ax.set_title("Standard vs factuality-aware prompting")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3)
    _save(fig, "prompt_comparison")


def chart_hindi_vs_codemixed(table: pd.DataFrame, exp: str):
    """Component scores for the Hindi vs the code-mixed set, one experiment."""
    metrics = [("claim_score", "Claim\nscore"), ("entity_score", "Entity\nscore"),
               ("pct_supported", "Claims\nsupported"), ("rougeL", "ROUGE-L*")]
    dss = [d for d in ("hindi", "code_mixed") if ((table.experiment == exp) & (table.dataset == d)).any()]
    if len(dss) < 2:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    width = 0.38
    for i, ds in enumerate(dss):
        row = table[(table.experiment == exp) & (table.dataset == ds)].iloc[0]
        xs = [j + (i - 0.5) * width for j in range(len(metrics))]
        bars = ax.bar(xs, [row[m] or 0 for m, _ in metrics], width * 0.92, color=DATASET_COLOR[ds], label=DATASET_LABEL[ds])
        _bar_labels(ax, bars)
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (0-1)")
    ax.set_title(f"Hindi vs code-mixed input ({EXP_LABEL[exp]})")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=2)
    _save(fig, "hindi_vs_code_mixed")


def chart_score_distribution(data):
    """Distribution of factuality scores per experiment (both datasets pooled)."""
    exps = [e for e in EXPERIMENTS if any(k[0] == e for k in data)]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    series = [[r["factuality_score"] for (e, _), rs in data.items() if e == exp for r in rs] for exp in exps]
    bp = ax.boxplot(series, widths=0.5, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#0b0b0b", "markersize": 4},
                    medianprops={"color": "#0b0b0b", "linewidth": 1})
    for patch, exp in zip(bp["boxes"], exps):
        patch.set_facecolor(EXP_COLOR[exp])
        patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(exps) + 1), [SHORT_EXP_LABEL[e] for e in exps])
    ax.set_ylabel("Factuality score (0-10)")
    lowest = min(min(x) for x in series if x)
    ax.set_ylim(max(0, int(lowest) - 1), 10.3)  # zoomed to the data: the axis does not start at 0
    ax.set_title("Distribution of factuality scores (◇ = mean)")
    _save(fig, "score_distribution")


def chart_error_categories(errors: pd.DataFrame):
    """Horizontal grouped bars: one panel per prompt, categories on the y axis."""
    if errors.empty:
        return
    exps = [e for e in ("standard", "factuality_aware", "factuality_aware_retry") if e in set(errors.experiment)]
    fig, axes = plt.subplots(1, len(exps), figsize=(7.16, 2.3), sharey=True, sharex=True, squeeze=False)
    cats = ERROR_CATEGORIES[::-1]  # first category at the top
    height = 0.38
    for ax, exp in zip(axes[0], exps):
        sub = errors[errors.experiment == exp]
        for i, ds in enumerate(("hindi", "code_mixed")):
            counts = sub[sub.dataset == ds].category.value_counts()
            ys = [j - (i - 0.5) * height for j in range(len(cats))]
            vals = [int(counts.get(c, 0)) for c in cats]
            bars = ax.barh(ys, vals, height * 0.92, color=DATASET_COLOR[ds], label=DATASET_LABEL[ds])
            for b, v in zip(bars, vals):
                if v:
                    ax.annotate(str(v), (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(2, 0),
                                textcoords="offset points", va="center", fontsize=6.5, color="#52514e")
        ax.set_yticks(range(len(cats)), [ERROR_LABEL[c] for c in cats])
        ax.set_title(EXP_LABEL[exp])
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_xlabel("Flagged claims")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.suptitle("Error categories of claims the verifier did not accept", fontsize=9, y=1.04)
    _save(fig, "error_categories")


def chart_retry_effect(data):
    """Score before (first attempt) and after the corrective loop, for items that needed a retry."""
    rows = [r for (e, _), rs in data.items() if e == "factuality_aware_retry" for r in rs if r["attempts"] > 1]
    if not rows:
        return
    before = mean(r["attempt_scores"][0] for r in rows)
    after = mean(r["factuality_score"] for r in rows)
    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    bars = ax.bar([0, 1], [before, after], 0.55, color=[EXP_COLOR["factuality_aware"], EXP_COLOR["factuality_aware_retry"]])
    _bar_labels(ax, bars)
    ax.set_xticks([0, 1], ["First attempt", "After retry loop"])
    ax.set_ylim(0, 11)
    ax.set_ylabel("Mean factuality score")
    ax.set_title(f"Corrective retry (n={len(rows)} summaries)")
    _save(fig, "retry_effect")


# ---------------------------------------------------------------- main

def to_latex(table: pd.DataFrame) -> str:
    cols = [("experiment", "System"), ("dataset", "Data"), ("n", "N"), ("factuality_score", "Fact."),
            ("claim_score", "Claim"), ("entity_score", "Entity"), ("number_mismatch_rate", "Num. err."),
            ("avg_summary_words", "Len."), ("compression_ratio", "Compr."), ("avg_inference_s", "Time (s)"),
            ("rougeL", "R-L*")]
    t = table[[c for c, _ in cols]].copy()
    t["experiment"] = t["experiment"].map(EXP_LABEL)
    t["dataset"] = t["dataset"].map({"hindi": "Hindi", "code_mixed": "Code-mixed"})
    t.columns = [label for _, label in cols]
    return t.to_latex(index=False, float_format="%.2f", na_rep="--")


async def main(args):
    data = load_all()
    if not data:
        print("no results yet; run eval/run_experiments.py first")
        return
    rows = [{"experiment": e, "dataset": d, **summarize_rows(rs)} for (e, d), rs in data.items()]
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    (RESULTS_DIR / "summary_table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (RESULTS_DIR / "summary_table.tex").write_text(to_latex(table), encoding="utf-8")
    with pd.option_context("display.max_columns", 30, "display.width", 250):
        print(table[["experiment", "dataset", "n", "factuality_score", "claim_score", "entity_score",
                     "number_mismatch_rate", "avg_summary_words", "compression_ratio", "avg_inference_s",
                     "avg_attempts", "rougeL"]].to_string(index=False))

    # paired check: are standard and factuality-aware compared on the same articles?
    for ds in ("hindi", "code_mixed"):
        ids = paired(data, "standard", "factuality_aware", ds)
        if ids:
            s = {r["id"]: r["factuality_score"] for r in data[("standard", ds)] if r["id"] in ids}
            f = {r["id"]: r["factuality_score"] for r in data[("factuality_aware", ds)] if r["id"] in ids}
            better = sum(f[i] > s[i] for i in ids)
            worse = sum(f[i] < s[i] for i in ids)
            print(f"{ds}: paired n={len(ids)}  standard {mean(s.values())}  factuality-aware {mean(f.values())}  "
                  f"(factuality-aware better on {better}, worse on {worse}, tied on {len(ids) - better - worse})")

    chart_prompt_comparison(table)
    chart_hindi_vs_codemixed(table, "factuality_aware" if ("factuality_aware", "code_mixed") in data else "standard")
    chart_score_distribution(data)
    chart_retry_effect(data)
    if not args.no_llm:
        errors = await classify_errors(data)
        errors.to_csv(RESULTS_DIR / "error_categories.csv", index=False)
        if not errors.empty:
            print(errors.groupby(["experiment", "dataset", "category"]).size().to_string())
        chart_error_categories(errors)
    await db.evaluation_runs.insert_one({"created_at": datetime.now(timezone.utc), "model": config.MODEL_NAME,
                                         "rerank_model": config.RERANK_MODEL, "summary": rows})
    print(f"\ntables: {RESULTS_DIR / 'summary_table.csv'} (+ .json, .tex); also saved to MongoDB evaluation_runs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM error classification")
    asyncio.run(main(ap.parse_args()))
