"""The full summarize -> verify -> retry flow. Yields (event, data) pairs that main.py sends as SSE."""
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import config
import db
import llm_service
import memory_service
import preprocessing as pp
import prompts
import vector_store
import verifier

log = logging.getLogger("pipeline")


@dataclass
class Request:
    text: str
    length: str = "medium"       # short | medium | long
    style: str = "paragraph"     # paragraph | bullets
    compare: bool = False
    session_id: str = "anonymous"
    # used by the evaluation scripts
    prompt_type: str = "factuality_aware"  # or "standard"
    max_retries: int = config.MAX_RETRIES
    is_eval: bool = False
    use_memory: bool = True  # evaluation turns Mem0 off so runs are independent


def now():
    return datetime.now(timezone.utc)


def ms(seconds: float) -> int:
    return round(seconds * 1000)


async def index_document(doc_id: str, text: str, language: str) -> list[dict]:
    """Chunk the source, normalize each chunk to English, embed and upsert into Qdrant.

    Claims are romanized Hinglish and the source may be Devanagari; multilingual models
    compare these badly, so retrieval, reranking and NLI all run on English versions.
    """
    t0 = time.perf_counter()
    chunks = pp.chunk_text(text)
    texts = [c["text"] for c in chunks]
    english = texts if language == "english" else await llm_service.translate_cached(texts)
    for c, en in zip(chunks, english):
        c["text_en"] = en
    t1 = time.perf_counter()
    await vector_store.index_chunks(doc_id, chunks)
    log.info("index: %d chunks | translate %.2fs, embed+upsert %.2fs", len(chunks), t1 - t0, time.perf_counter() - t1)
    return chunks


class SummaryStream:
    """Splits the factuality-aware output 'FACTS: ... SUMMARY: ...' and only forwards the summary."""
    MARKER = "SUMMARY:"

    STRIP = " \n\t*#:"  # models sometimes write "**SUMMARY:**"

    def __init__(self, has_facts_section: bool):
        self.buffer = ""
        self.in_summary = not has_facts_section
        self.sent = 0          # how much of the buffer was already forwarded
        self.started = False   # forwarded any non-whitespace yet?

    def feed(self, piece: str) -> str:
        self.buffer += piece
        if not self.in_summary:
            idx = self.buffer.upper().find(self.MARKER)
            if idx == -1:
                return ""
            self.in_summary = True
            self.sent = idx + len(self.MARKER)
        out = self.buffer[self.sent:]
        self.sent = len(self.buffer)
        if not self.started:
            out = out.lstrip(self.STRIP)
            self.started = bool(out)
        return out

    def result(self) -> tuple[str, str]:
        """(key_facts, summary). If the marker never appeared, the whole output is the summary."""
        idx = self.buffer.upper().find(self.MARKER)
        if idx == -1:
            return "", self.buffer.strip()
        facts = re.sub(r"^[\s*#]*facts:[\s*]*", "", self.buffer[:idx], flags=re.IGNORECASE).strip(self.STRIP)
        return facts, self.buffer[idx + len(self.MARKER):].strip(self.STRIP)


async def generate(messages: list[dict], has_facts_section: bool):
    """Stream the LLM output. Yields ('token', text) and finally ('done', (facts, summary))."""
    parser = SummaryStream(has_facts_section)
    async for piece in llm_service.stream(messages):
        out = parser.feed(piece)
        if out:
            yield "token", out
    yield "done", parser.result()


async def save_attempt(doc_id, number, prompt_type, summary, key_facts, verification, elapsed_ms, timings):
    attempt = {
        "doc_id": doc_id, "attempt_number": number, "prompt_type": prompt_type, "summary": summary,
        "key_facts": key_facts,
        "factuality_score": verification["factuality_score"], "claim_score": verification["claim_score"],
        "entity_score": verification["entity_score"],
        "low_confidence": verification["factuality_score"] < config.FACTUALITY_THRESHOLD,
        "inference_time_ms": elapsed_ms, "timings": timings, "created_at": now(),
        "claims": verification["claims"], "facts": verification["facts"],
        "entity_mismatches": verification["entity_mismatches"],
    }
    res = await db.summary_attempts.insert_one(attempt)
    attempt["_id"] = res.inserted_id
    return attempt


def build_result(doc, attempt, source_text, attempts_count, total_ms) -> dict:
    words_in = pp.word_count(source_text)
    words_out = pp.word_count(attempt["summary"])
    return {
        "doc_id": str(doc["_id"]), "attempt_id": str(attempt["_id"]),
        "summary": attempt["summary"], "prompt_type": attempt["prompt_type"],
        "factuality_score": attempt["factuality_score"], "claim_score": attempt["claim_score"],
        "entity_score": attempt["entity_score"], "low_confidence": attempt["low_confidence"],
        "claims": attempt["claims"], "entity_mismatches": attempt["entity_mismatches"],
        "facts": attempt["facts"], "attempts": attempts_count, "language_type": doc["language_type"],
        "source_text": source_text,
        "stats": {"words_in": words_in, "words_out": words_out,
                  "compression_ratio": round(1 - words_out / words_in, 4) if words_in else 0,
                  "inference_time_ms": total_ms},
    }


async def run(req: Request):
    start = time.perf_counter()
    text = pp.clean_text(req.text)
    language = pp.detect_language(text)
    doc = {"session_id": req.session_id, "source_text": text, "language_type": language,
           "word_count": pp.word_count(text), "created_at": now(), "is_eval": req.is_eval,
           "options": {"length": req.length, "style": req.style, "compare": req.compare}}
    doc["_id"] = (await db.documents.insert_one(doc)).inserted_id
    doc_id = str(doc["_id"])
    yield "status", {"phase": "Preparing", "language_type": language, "doc_id": doc_id}

    # Indexing and the standard baseline run in the background while we generate.
    index_task = asyncio.create_task(index_document(doc_id, text, language))
    standard_task = None
    if req.compare:
        standard_task = asyncio.create_task(
            llm_service.complete(prompts.standard_prompt(text, req.length, req.style), temperature=0.3))

    try:
        memory_rules = []
        if req.use_memory:
            yield "status", {"phase": "Recalling past corrections"}
            t = time.perf_counter()
            try:
                memory_rules = await memory_service.recall(req.session_id, text)
                log.info("memory: %d rules in %.2fs", len(memory_rules), time.perf_counter() - t)
            except Exception as e:
                # Memory is optional for a summary, but the failure must be visible.
                log.error("Mem0 recall failed: %s: %s", type(e).__name__, e)
                yield "status", {"phase": "Memory unavailable, continuing without past corrections", "warning": True}

        best, previous = None, None
        max_attempts = 1 + max(0, req.max_retries)
        for number in range(1, max_attempts + 1):
            attempt_start = time.perf_counter()
            if number == 1:
                prompt_type = req.prompt_type
                if prompt_type == "standard":
                    messages = prompts.standard_prompt(text, req.length, req.style)
                else:
                    messages = prompts.factuality_prompt(text, req.length, req.style, memory_rules)
                yield "status", {"phase": "Generating"}
            else:
                prompt_type = "corrective"
                messages = prompts.corrective_prompt(text, req.length, req.style, previous["summary"],
                                                     verifier.problems_for_retry(previous["verification"]),
                                                     memory_rules)
                yield "attempt", {"attempt": number, "previous_score": previous["verification"]["factuality_score"]}
                yield "status", {"phase": "Improving summary"}

            async for kind, value in generate(messages, has_facts_section=prompt_type != "standard"):
                if kind == "token":
                    yield "token", {"text": value}
                else:
                    key_facts, summary = value
            gen_ms = ms(time.perf_counter() - attempt_start)
            if not summary:
                raise llm_service.LLMError("The LLM returned an empty summary.")

            yield "status", {"phase": "Verifying claims"}
            t = time.perf_counter()
            chunks = await index_task  # usually finished long before generation ends
            source_en = " ".join(c["text_en"] for c in chunks)
            index_ms = ms(time.perf_counter() - t)
            verification = await verifier.verify_summary(doc_id, text, summary, source_en)
            timings = {"generate_ms": gen_ms, "wait_index_ms": index_ms, **verification["timings"]}
            elapsed = ms(time.perf_counter() - attempt_start)
            attempt = await save_attempt(doc_id, number, prompt_type, summary, key_facts, verification, elapsed, timings)
            log.info("attempt %d (%s): score %.2f | %s", number, prompt_type, verification["factuality_score"], timings)

            previous = {"summary": summary, "verification": verification}
            if best is None or attempt["factuality_score"] > best["factuality_score"]:
                best = attempt
            # Output must be Hinglish in Roman script: leftover Devanagari also triggers a retry.
            if pp.DEVANAGARI_RE.search(summary):
                verification["entity_mismatches"] = verification["entity_mismatches"] + [{
                    "text": "Devanagari", "type": "script", "matched": False,
                    "reason": "the summary contains Devanagari script; write every word in Roman script"}]
            elif attempt["factuality_score"] >= config.FACTUALITY_THRESHOLD:
                break

        total_ms = ms(time.perf_counter() - start)
        result = build_result(doc, best, text, number, total_ms)
        await db.documents.update_one({"_id": doc["_id"]}, {"$set": {
            "final_attempt_id": best["_id"], "final_score": best["factuality_score"],
            "low_confidence": best["low_confidence"], "attempts": number,
            "inference_time_ms": total_ms, "title": title_from(text)}})
        yield "result", result

        if standard_task:
            yield "status", {"phase": "Verifying standard summary"}
            t = time.perf_counter()
            std_summary = (await standard_task).strip()
            std_ver = await verifier.verify_summary(doc_id, text, std_summary, source_en)
            std_attempt = await save_attempt(doc_id, 0, "standard", std_summary, "", std_ver,
                                             ms(time.perf_counter() - t), std_ver["timings"])
            first = await db.summary_attempts.find_one({"doc_id": doc_id, "attempt_number": 1})
            compare = {"standard": {"summary": std_summary, "factuality_score": std_ver["factuality_score"],
                                    "claim_score": std_ver["claim_score"], "entity_score": std_ver["entity_score"],
                                    "claims": std_ver["claims"], "entity_mismatches": std_ver["entity_mismatches"],
                                    "attempt_id": str(std_attempt["_id"])},
                       "factuality_aware": {"summary": result["summary"], "factuality_score": result["factuality_score"],
                                            "first_attempt_score": first["factuality_score"]}}
            await db.documents.update_one({"_id": doc["_id"]}, {"$set": {"compare_attempt_id": std_attempt["_id"]}})
            yield "compare_result", compare
    except llm_service.LLMError as e:
        yield "error", {"message": str(e)}
    except Exception as e:
        log.exception("pipeline failed")
        yield "error", {"message": f"Summarization failed: {type(e).__name__}. See the server log."}
    finally:
        for task in (index_task, standard_task):
            if task and not task.done():
                task.cancel()


def title_from(source: str) -> str:
    words = source.split("\n")[0].split()
    title = " ".join(words[:6])
    return title + ("…" if len(words) > 6 else "")


async def summarize(req: Request) -> dict:
    """Non-streaming helper (evaluation): run the pipeline and return the final result."""
    result, compare = None, None
    async for event, data in run(req):
        if event == "result":
            result = data
        elif event == "compare_result":
            compare = data
        elif event == "error":
            raise RuntimeError(data["message"])
    if compare:
        result["compare"] = compare
    return result
