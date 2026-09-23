"""Multilingual cross-encoder reranker (scores a claim against each candidate chunk)."""
import asyncio

import config

_model = None


def load():
    global _model
    import torch
    from sentence_transformers import CrossEncoder
    # Sigmoid so every reranker gives scores in [0, 1] (mMiniLM returns raw logits otherwise)
    _model = CrossEncoder(config.RERANK_MODEL, device="cpu", max_length=512, activation_fn=torch.nn.Sigmoid())


def is_loaded() -> bool:
    return _model is not None


async def rerank_many(queries: list[str], candidate_lists: list[list[dict]],
                      top_k: int = config.RERANK_TOP_K) -> list[list[dict]]:
    """Rerank candidates for several queries in ONE model call (much faster on CPU than
    one call per claim). Returns the top_k per query with a `rerank_score` in [0, 1]."""
    pairs = [(q, c["text_en"]) for q, cands in zip(queries, candidate_lists) for c in cands]
    if not pairs:
        return [[] for _ in queries]
    scores = iter(await asyncio.to_thread(_model.predict, pairs, batch_size=16))
    results = []
    for cands in candidate_lists:
        scored = [{**c, "rerank_score": round(float(next(scores)), 4)} for c in cands]
        scored.sort(key=lambda c: c["rerank_score"], reverse=True)
        results.append(scored[:top_k])
    return results
