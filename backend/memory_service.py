"""Mem0 (hosted) memory of user corrections, keyed by session_id.

mem0ai 2.x API: add(messages, user_id=...) but search(query, filters={"user_id": ...}).
Passing user_id at the top level of search() raises ValueError in this version.
"""
import logging

import config
import prompts

log = logging.getLogger("memory")
_client = None


def client():
    global _client
    if _client is None:
        from mem0 import AsyncMemoryClient
        _client = AsyncMemoryClient(api_key=config.MEM0_API_KEY)
    return _client


async def add_correction(session_id: str, claim_text: str, note: str, doc_id: str = "") -> dict:
    rule = prompts.memory_rule(claim_text, note)
    # infer=False stores our sentence as-is instead of letting Mem0's LLM rewrite it
    return await client().add(rule, user_id=session_id, infer=False,
                              metadata={"type": "correction", "doc_id": doc_id})


async def search(session_id: str, query: str, top_k: int = 5) -> list[str]:
    res = await client().search(query[:1000], filters={"user_id": session_id}, top_k=top_k)
    items = res.get("results", []) if isinstance(res, dict) else res
    return [m["memory"] for m in items if m.get("memory")]


async def recall(session_id: str, text: str) -> list[str]:
    """Past corrections relevant to this text, as soft rules for the prompt."""
    if not config.MEM0_API_KEY:
        log.warning("MEM0_API_KEY not set; skipping memory recall")
        return []
    return await search(session_id, text)
