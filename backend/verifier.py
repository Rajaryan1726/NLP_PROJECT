"""Claim-level factuality verification: retrieval -> rerank -> NLI, plus entity/number/date matching."""
import asyncio
import logging
import re
import time
from difflib import SequenceMatcher

import config
import llm_service
import preprocessing as pp
import reranker
import vector_store

log = logging.getLogger("verifier")
_nli_model = None
_nli_tokenizer = None
_nli_labels = None  # index -> "entailment" | "neutral" | "contradiction"


def load_nli():
    global _nli_model, _nli_tokenizer, _nli_labels
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _nli_tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL)
    # float32: the checkpoint is stored in float16, which is ~10x slower on CPU
    _nli_model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL, dtype=torch.float32).eval()
    _nli_labels = {i: label.lower() for i, label in _nli_model.config.id2label.items()}


def is_loaded() -> bool:
    return _nli_model is not None


def _nli_batch(pairs: list[tuple[str, str]], batch_size: int = 16) -> list[dict]:
    import torch
    # Batch pairs of similar length together so short pairs are not padded to the longest one.
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    results = [None] * len(pairs)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = _nli_tokenizer([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation=True,
                             max_length=512, padding=True, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(_nli_model(**enc).logits, dim=-1).tolist()
        for i, row in zip(idx, probs):
            results[i] = {_nli_labels[j]: p for j, p in enumerate(row)}
    return results


async def nli(pairs: list[tuple[str, str]]) -> list[dict]:
    """(premise, hypothesis) pairs -> [{entailment, neutral, contradiction}] probabilities."""
    if not pairs:
        return []
    return await asyncio.to_thread(_nli_batch, pairs)


# ---------------------------------------------------------------- claims

def _premises(evidence: list[dict]) -> list[tuple[str, dict]]:
    """NLI premises for one claim, each with the chunk it came from.

    NLI models are unreliable on multi-sentence premises (a true claim about sentence 1
    can be called a contradiction because of sentence 3), so every single sentence of the
    top chunks is a premise too, as well as each whole chunk and the top two joined
    (for claims that span two chunks).
    """
    out = []
    for chunk in evidence:
        sentences = [s for s, _, _ in pp.split_sentences(chunk["text_en"])]
        if len(sentences) > 1:
            out += [(s, chunk) for s in sentences]
        out.append((chunk["text_en"], chunk))
    if len(evidence) > 1:
        out.append((evidence[0]["text_en"] + " " + evidence[1]["text_en"], evidence[0]))
    return out


def _decide(premises: list[tuple[str, dict]], probs: list[dict]) -> tuple[str, float, dict]:
    """Turn NLI probabilities over several premises into one verdict.

    Any premise that entails the claim wins; otherwise any contradiction; otherwise neutral.
    Returns (verdict, confidence, chunk that decided it).
    """
    for label, verdict in (("entailment", "entailed"), ("contradiction", "contradicted")):
        scored = [(p[label], i) for i, p in enumerate(probs) if max(p, key=p.get) == label]
        if scored:
            conf, i = max(scored)
            return verdict, conf, premises[i][1]
    return "neutral", sum(p["neutral"] for p in probs) / len(probs), premises[0][1]


def _with_evidence(result: dict, chunk: dict) -> dict:
    result.update(evidence_chunk_id=chunk["chunk_id"], evidence_text=chunk["text"],
                  evidence_start=chunk["start"], evidence_end=chunk["end"], rerank_score=chunk["rerank_score"])
    return result


async def verify_claims(doc_id: str, claims: list[dict]) -> list[dict]:
    """Retrieve (Qdrant top-10) -> rerank (top-3) -> NLI for every claim.

    All claims go through the reranker and the NLI model in one batch each,
    which is several times faster on CPU than one model call per claim.
    """
    queries = [c["claim_en"] for c in claims]
    candidates = await vector_store.search_many(doc_id, queries, config.RETRIEVE_TOP_K)
    tops = await reranker.rerank_many(queries, candidates, config.RERANK_TOP_K)

    results, pairs, spans = [], [], []
    for claim, top in zip(claims, tops):
        result = {"claim_text": claim["claim"], "claim_en": claim["claim_en"], "verdict": "unsupported",
                  "nli_confidence": 0.0, "evidence_chunk_id": None, "evidence_text": "",
                  "evidence_start": None, "evidence_end": None, "rerank_score": 0.0}
        if top:
            _with_evidence(result, top[0])
        # Abstain: nothing in the source is even about this claim, so NLI would only guess.
        evidence = [c for c in top if c["rerank_score"] >= config.MIN_RERANK_SCORE]
        premises = _premises(evidence)
        spans.append((premises, len(pairs), len(pairs) + len(premises)))
        pairs += [(p, claim["claim_en"]) for p, _ in premises]
        results.append(result)

    probs = await nli(pairs)
    for result, (premises, start, end) in zip(results, spans):
        if not premises:
            continue
        verdict, conf, chunk = _decide(premises, probs[start:end])
        _with_evidence(result, chunk)
        result.update(verdict=verdict, nli_confidence=round(conf, 4))
    return results


# ---------------------------------------------------------------- entities / numbers / dates

def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def entity_in_source(name: str, source_texts: list[str]) -> bool:
    """Is this (Roman-script) name in the source or in its English translation?

    Exact match after normalization, else every word of the name must have a close
    spelling in the source (difflib ratio >= 0.8), which allows "Pradhaan" vs "Pradhan".
    """
    name_n = _norm(name)
    if not name_n:
        return True
    haystack = " ".join(_norm(t) for t in source_texts)
    if f" {name_n} " in f" {haystack} ":
        return True
    words = set(haystack.split())
    return all(any(SequenceMatcher(None, tok, w).ratio() >= 0.8 for w in words) for tok in name_n.split())


async def match_facts(source: str, summary: str, source_en: str = "") -> tuple[list[dict], list[dict]]:
    """Check every number, date and named entity in the summary against the source.

    Numbers and dates: regex + normalization (2 crore == 20,000,000, अगस्त == August).
    Named entities: the LLM only extracts the proper names from the summary; whether each one
    is in the source is decided here by string matching against the source and its English
    translation (the source may be Devanagari, the summary is always Roman script).
    Returns (facts, mismatches).
    """
    src_dates, src_rest = pp.parse_dates(source)
    sum_dates, sum_rest = pp.parse_dates(summary)
    src_values = [v for _, v in pp.parse_numbers(src_rest)]
    src_values += pp.bare_numbers(src_rest)  # "4,500 crore" also allows a summary's plain "4,500"
    src_values += list(pp.number_words(source))
    src_values += [float(part) for _, d in src_dates for part in d if part is not None]

    facts = []
    for raw, value in pp.parse_numbers(sum_rest):
        ok = any(pp.numbers_match(value, s) for s in src_values)
        facts.append({"text": raw, "type": "number", "matched": ok,
                      "reason": "" if ok else f"number {raw} (= {value:,.0f}) not found in source"})
    for raw, d in sum_dates:
        ok = any(pp.dates_match(d, s) for _, s in src_dates) or (d[0] is not None and d[1] is None and float(d[0]) in src_values)
        facts.append({"text": raw, "type": "date", "matched": ok,
                      "reason": "" if ok else f"date {raw} not found in source"})

    for e in await llm_service.extract_entities(summary):
        if any(ch.isdigit() for ch in e["text"]):
            continue  # numbers and dates were checked above
        ok = entity_in_source(e["text"], [source, source_en])
        facts.append({"text": e["text"], "type": "entity", "entity_type": e.get("type", "OTHER"), "matched": ok,
                      "reason": "" if ok else f"entity '{e['text']}' not found in source"})
    mismatches = [f for f in facts if not f["matched"]]
    return facts, mismatches


# ---------------------------------------------------------------- scoring

def compute_scores(verdicts: list[str], n_facts: int, n_matched: int) -> dict:
    """factuality = 10 * (0.6 * claim_score + 0.4 * entity_score)."""
    claim_score = verdicts.count("entailed") / len(verdicts) if verdicts else 0.0
    entity_score = n_matched / n_facts if n_facts else 1.0  # nothing to get wrong
    score = 10 * (0.6 * claim_score + 0.4 * entity_score)
    return {"claim_score": round(claim_score, 4), "entity_score": round(entity_score, 4),
            "factuality_score": round(score, 2)}


async def verify_summary(doc_id: str, source: str, summary: str, source_en: str = "") -> dict:
    """Full verification of one summary against its (already indexed) source document."""
    t0 = time.perf_counter()
    claims, (facts, mismatches) = await asyncio.gather(
        llm_service.decompose_claims(summary), match_facts(source, summary, source_en))
    t1 = time.perf_counter()
    verified = await verify_claims(doc_id, claims)
    t2 = time.perf_counter()
    scores = compute_scores([c["verdict"] for c in verified], len(facts), len(facts) - len(mismatches))
    log.info("verify: %d claims, %d facts (%d mismatches) | decompose+facts %.2fs, retrieve+rerank+nli %.2fs",
             len(verified), len(facts), len(mismatches), t1 - t0, t2 - t1)
    return {**scores, "claims": verified, "facts": facts, "entity_mismatches": mismatches,
            "timings": {"decompose_ms": round((t1 - t0) * 1000), "verify_ms": round((t2 - t1) * 1000)}}


def problems_for_retry(result: dict) -> list[str]:
    """Human-readable problems that go into the corrective prompt."""
    problems = []
    for c in result["claims"]:
        if c["verdict"] == "entailed":
            continue
        label = {"contradicted": "CONTRADICTED by the source", "neutral": "NOT SUPPORTED by the source",
                 "unsupported": "NOT FOUND in the source"}[c["verdict"]]
        line = f'Claim "{c["claim_text"]}" is {label}.'
        if c["evidence_text"] and c["verdict"] != "unsupported":
            line += f' Closest source text: "{c["evidence_text"]}"'
        problems.append(line)
    problems += [m["reason"] for m in result["entity_mismatches"]]
    return problems
