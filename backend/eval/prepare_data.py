"""Build the evaluation datasets.

(a) hindi.jsonl: a random sample of the XL-Sum Hindi test split (real BBC Hindi articles
    with human-written reference summaries in Devanagari).
(b) code_mixed.jsonl: SYNTHETIC Hindi-English code-mixed rewrites of the first N of those
    articles, made by the LLM with a strict fact-preserving prompt. Each rewrite is checked:
    every number in the original must still be in the rewrite, otherwise it is regenerated
    once and then dropped. This synthetic set is a stated limitation in the paper.

Usage:  python eval/prepare_data.py --n-hindi 100 --n-code-mixed 50
"""
import argparse
import asyncio
import random

import common  # noqa: F401  (sets up paths + LLM cache)
from common import DATASETS, read_jsonl, write_jsonl

import pandas as pd
from huggingface_hub import hf_hub_download

import llm_service
import preprocessing as pp

CODE_MIX_PROMPT = """Rewrite the Hindi news article below as natural Hindi-English code-mixed text, the way
people in India write on WhatsApp or social media: romanized Hindi (Latin script) mixed with English
words and phrases, e.g. "Sarkar ne aaj nayi scheme launch ki, jiska budget 4,500 crore hai".

STRICT RULES:
- Keep EVERY fact: every name, place, organisation, number, amount, date and event.
- Copy every number exactly (same digits; keep units like crore/lakh/percent).
- Do not add, remove or change any information. Do not summarize. Keep the same order.
- Output only the rewritten article, in Roman script (no Devanagari)."""


def load_xlsum_hindi() -> pd.DataFrame:
    path = hf_hub_download("csebuetnlp/xlsum", "hindi/test/0000.parquet", repo_type="dataset",
                           revision="refs/convert/parquet")
    return pd.read_parquet(path)


def numbers_kept(original: str, rewrite: str) -> bool:
    _, orig_rest = pp.parse_dates(original)
    wanted = {v for _, v in pp.parse_numbers(orig_rest)}
    have = set(pp.bare_numbers(rewrite)) | {v for _, v in pp.parse_numbers(rewrite)}
    return all(any(pp.numbers_match(w, h) for h in have) for w in wanted)


async def make_code_mixed(article: str) -> str | None:
    for temperature in (0.3, 0.5):  # second try with a different sample
        messages = [{"role": "system", "content": CODE_MIX_PROMPT}, {"role": "user", "content": article}]
        text = (await llm_service.complete(messages, temperature=temperature)).strip()
        if text and not pp.DEVANAGARI_RE.search(text) and numbers_kept(article, text):
            return text
    return None


async def main(n_hindi: int, n_code_mixed: int, seed: int):
    df = load_xlsum_hindi()
    df["text"] = df["text"].map(pp.clean_text)
    df["words"] = df["text"].map(pp.word_count)
    # keep normal-length articles (the app's limit is 5,000 words; very short ones are not worth summarizing)
    df = df[(df["words"] >= 150) & (df["words"] <= 1200)]
    sample = df.sample(n=min(n_hindi, len(df)), random_state=seed)
    hindi = [{"id": str(r.id), "title": r.title, "text": r.text, "reference": pp.clean_text(r.summary),
              "url": r.url, "words": int(r.words)} for r in sample.itertuples()]
    write_jsonl(DATASETS["hindi"], hindi)
    print(f"hindi: {len(hindi)} articles (from {len(df)} eligible XL-Sum test articles, seed {seed})")

    existing = {r["id"]: r for r in read_jsonl(DATASETS["code_mixed"])}
    rows, dropped = [], 0
    sem = asyncio.Semaphore(4)

    async def one(item):
        nonlocal dropped
        if item["id"] in existing:
            return existing[item["id"]]
        async with sem:
            text = await make_code_mixed(item["text"])
        if text is None:
            dropped += 1
            print(f"  dropped {item['id']}: rewrite lost a number or used Devanagari")
            return None
        return {"id": item["id"], "title": item["title"], "text": text, "reference": item["reference"],
                "source_id": item["id"], "synthetic": True, "words": pp.word_count(text)}

    # take extra candidates so that dropped rewrites can be replaced
    results = await asyncio.gather(*(one(it) for it in hindi[: n_code_mixed + 10]))
    rows = [r for r in results if r][:n_code_mixed]
    write_jsonl(DATASETS["code_mixed"], rows)
    langs = {pp.detect_language(r["text"]) for r in rows}
    print(f"code_mixed: {len(rows)} synthetic articles ({dropped} rewrites rejected); detected as {langs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hindi", type=int, default=100)
    ap.add_argument("--n-code-mixed", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    asyncio.run(main(args.n_hindi, args.n_code_mixed, args.seed))
