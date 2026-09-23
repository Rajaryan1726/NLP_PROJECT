"""Run the summarization experiments on the evaluation datasets.

Experiments (see common.EXPERIMENTS):
  standard                 plain prompt, no retry loop
  factuality_aware         factuality-aware prompt, no retry loop
  factuality_aware_retry   factuality-aware prompt + corrective retry loop (max 2 retries)

Resumable: every finished item is appended to results/raw/<experiment>__<dataset>.jsonl and
skipped on the next run. LLM outputs are also cached on disk (eval/cache/llm).
Mem0 is off during evaluation so every run is independent.

Usage:
  python eval/run_experiments.py --limit 5                 # quick smoke test
  python eval/run_experiments.py                           # everything
  python eval/run_experiments.py --experiments standard --datasets hindi --concurrency 3

Timing note: --concurrency > 1 is faster but inflates the per-summary inference time
(requests share the CPU models and the API). Use 1 for the timing numbers in the paper.
"""
import argparse
import asyncio
import logging
import time
import warnings

import common
from common import DATASETS, EXPERIMENTS, append_jsonl, raw_path, read_jsonl

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

import db  # noqa: E402
import llm_service  # noqa: E402
import pipeline  # noqa: E402
import reranker  # noqa: E402
import vector_store  # noqa: E402
import verifier  # noqa: E402


async def run_one(item: dict, experiment: str, dataset: str) -> dict:
    prompt_type, max_retries = EXPERIMENTS[experiment]
    req = pipeline.Request(text=item["text"], length="medium", style="paragraph", compare=False,
                           session_id="eval", prompt_type=prompt_type, max_retries=max_retries,
                           is_eval=True, use_memory=False)
    hits = [0]
    llm_service.cache_hits.set(hits)  # context-local, so concurrent runs count separately
    t = time.perf_counter()
    r = await pipeline.summarize(req)
    wall_ms = round((time.perf_counter() - t) * 1000)
    attempts = await db.summary_attempts.find({"doc_id": r["doc_id"], "attempt_number": {"$gte": 1}}) \
        .sort("attempt_number", 1).to_list(10)
    return {
        "id": item["id"], "experiment": experiment, "dataset": dataset,
        "language_type": r["language_type"], "doc_id": r["doc_id"],
        "summary": r["summary"], "reference": item.get("reference", ""),
        "factuality_score": r["factuality_score"], "claim_score": r["claim_score"],
        "entity_score": r["entity_score"], "low_confidence": r["low_confidence"],
        "attempts": r["attempts"], "attempt_scores": [a["factuality_score"] for a in attempts],
        "stats": r["stats"], "wall_ms": wall_ms, "llm_cache_hits": hits[0],
        "timings": [a.get("timings", {}) for a in attempts],
        "claims": [{k: c[k] for k in ("claim_text", "claim_en", "verdict", "nli_confidence",
                                      "rerank_score", "evidence_text", "evidence_chunk_id")}
                   for c in r["claims"]],
        "facts": r["facts"], "entity_mismatches": r["entity_mismatches"],
    }


async def main(args):
    print("loading models…")
    vector_store.load_embedder()
    reranker.load()
    verifier.load_nli()
    await db.ping()
    await vector_store.ensure_collection()

    sem = asyncio.Semaphore(args.concurrency)
    for experiment in args.experiments:
        llm_service.cache_namespace = experiment
        for dataset in args.datasets:
            items = read_jsonl(DATASETS[dataset])
            if not items:
                print(f"no data for {dataset}; run eval/prepare_data.py first")
                continue
            if args.limit:
                items = items[: args.limit]
            out = raw_path(experiment, dataset)
            done = {r["id"] for r in read_jsonl(out)}
            todo = [it for it in items if it["id"] not in done]
            print(f"\n== {experiment} / {dataset}: {len(items) - len(todo)} done, {len(todo)} to run")

            async def worker(item, i):
                async with sem:
                    try:
                        row = await run_one(item, experiment, dataset)
                    except Exception as e:  # keep going; the item is retried on the next run
                        print(f"  [{i}] {item['id']} FAILED: {e}")
                        return
                    append_jsonl(out, row)
                    print(f"  [{i}/{len(todo)}] {item['id']} score {row['factuality_score']:.2f} "
                          f"attempts {row['attempts']} {row['wall_ms'] / 1000:.1f}s")

            await asyncio.gather(*(worker(it, i + 1) for i, it in enumerate(todo)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--limit", type=int, default=0, help="only the first N items of each dataset")
    ap.add_argument("--concurrency", type=int, default=1)
    asyncio.run(main(ap.parse_args()))
