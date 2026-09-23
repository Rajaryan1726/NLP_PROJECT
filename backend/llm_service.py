"""Every LLM call goes through this module (LiteLLM, so MODEL_NAME can be any provider)."""
import asyncio
import hashlib
import json
import logging
import os
import re
from contextvars import ContextVar
from pathlib import Path

import litellm

import config
import db
import prompts

log = logging.getLogger("llm")
litellm.drop_params = True  # e.g. Ollama ignores response_format instead of erroring
litellm.suppress_debug_info = True
# httpx transport: idle connections expire after 5s. The default aiohttp pool keeps them for
# 120s, and a connection silently dropped by the router meanwhile makes the next call hang.
litellm.disable_aiohttp_transport = True

# Optional disk cache (used by the evaluation scripts to make reruns free).
CACHE_DIR = os.getenv("LLM_CACHE_DIR")
cache_namespace = ""  # the evaluation sets one per experiment, so experiments never share samples
# Per-run counter of cache hits (the evaluation ignores timings of runs that hit the cache).
cache_hits: ContextVar[list | None] = ContextVar("cache_hits", default=None)


def _count_hit():
    counter = cache_hits.get()
    if counter is not None:
        counter[0] += 1


class LLMError(Exception):
    """A clean, user-facing LLM failure message."""


def _cache_path(messages, temperature, json_mode) -> Path | None:
    if not CACHE_DIR:
        return None
    key = json.dumps([cache_namespace, config.MODEL_NAME, messages, temperature, json_mode], ensure_ascii=False)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    return Path(CACHE_DIR) / (hashlib.sha256(key.encode()).hexdigest() + ".txt")


def _friendly(e: Exception) -> LLMError:
    name = type(e).__name__
    if "Authentication" in name:
        return LLMError("LLM authentication failed. Check OPENAI_API_KEY in backend/.env.")
    if "RateLimit" in name:
        return LLMError("LLM rate limit or quota exceeded. Try again in a moment.")
    if "Connection" in name or "Timeout" in name or "ServiceUnavailable" in name:
        return LLMError(f"Could not reach the LLM ({config.MODEL_NAME}). Is it running / online?")
    return LLMError(f"LLM call failed ({name}).")


async def complete(messages: list[dict], temperature: float = 0.0, json_mode: bool = False) -> str:
    path = _cache_path(messages, temperature, json_mode)
    if path and path.exists():
        _count_hit()
        return path.read_text(encoding="utf-8")
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    try:
        resp = await litellm.acompletion(model=config.MODEL_NAME, messages=messages,
                                         temperature=temperature, timeout=config.LLM_TIMEOUT,
                                         num_retries=2, **kwargs)
    except Exception as e:
        log.error("LLM call failed: %s", e)
        raise _friendly(e) from e
    text = resp.choices[0].message.content or ""
    if path:
        path.write_text(text, encoding="utf-8")
    return text


async def stream(messages: list[dict], temperature: float = 0.3):
    """Yield text pieces as they arrive."""
    path = _cache_path(messages, temperature, False)
    if path and path.exists():
        _count_hit()
        yield path.read_text(encoding="utf-8")
        return
    parts = []
    try:
        resp = await litellm.acompletion(model=config.MODEL_NAME, messages=messages,
                                         temperature=temperature, stream=True, timeout=config.LLM_TIMEOUT,
                                         num_retries=2)
        async for chunk in resp:
            piece = chunk.choices[0].delta.content or ""
            if piece:
                parts.append(piece)
                yield piece
    except Exception as e:
        log.error("LLM stream failed: %s", e)
        raise _friendly(e) from e
    if path:
        path.write_text("".join(parts), encoding="utf-8")


def parse_json(text: str) -> dict:
    """Parse JSON even if the model wrapped it in ``` fences (common with local models)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


async def complete_json(messages: list[dict]) -> dict:
    text = await complete(messages, temperature=0.0, json_mode=True)
    try:
        return parse_json(text)
    except (json.JSONDecodeError, ValueError) as e:
        log.error("LLM returned invalid JSON: %.200s", text)
        raise LLMError("The LLM returned malformed JSON.") from e


# ---------------------------------------------------------------- task helpers

async def decompose_claims(summary: str) -> list[dict]:
    data = await complete_json(prompts.claims_prompt(summary))
    claims = []
    for c in data.get("claims", []):
        if isinstance(c, dict) and c.get("claim"):
            claims.append({"claim": c["claim"].strip(), "claim_en": (c.get("claim_en") or c["claim"]).strip()})
        elif isinstance(c, str) and c.strip():
            claims.append({"claim": c.strip(), "claim_en": c.strip()})
    return claims


async def translate(texts: list[str]) -> list[str]:
    """Translate a batch of texts to English in one call; falls back to one call per text."""
    if not texts:
        return []
    data = await complete_json(prompts.translate_prompt(texts))
    out = data.get("translations", [])
    if len(out) == len(texts) and all(isinstance(t, str) for t in out):
        return out
    log.warning("translation batch returned %d items for %d inputs, retrying one by one", len(out), len(texts))
    results = []
    for t in texts:
        single = (await complete_json(prompts.translate_prompt([t]))).get("translations", [t])
        results.append(single[0] if single and isinstance(single[0], str) else t)
    return results


async def translate_cached(texts: list[str], batch_size: int = 15) -> list[str]:
    """translate() with a MongoDB cache, in parallel batches (long documents have many chunks)."""
    keys = [hashlib.sha1(f"{config.MODEL_NAME}::{t}".encode()).hexdigest() for t in texts]
    cached = {d["_id"]: d["en"] async for d in db.translation_cache.find({"_id": {"$in": keys}})}
    todo = sorted({i for i, k in enumerate(keys) if k not in cached})
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    results = await asyncio.gather(*(translate([texts[i] for i in b]) for b in batches))
    for batch, translated in zip(batches, results):
        for i, en in zip(batch, translated):
            cached[keys[i]] = en
            await db.translation_cache.update_one({"_id": keys[i]}, {"$set": {"en": en}}, upsert=True)
    return [cached[k] for k in keys]


async def extract_entities(summary: str) -> list[dict]:
    data = await complete_json(prompts.entities_prompt(summary))
    return [e for e in data.get("entities", []) if isinstance(e, dict) and e.get("text")]
