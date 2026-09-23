"""FastAPI app: routes only. The real work is in pipeline.py."""
import asyncio
import json
import logging
import time
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import config
import db
import memory_service
import pipeline
import preprocessing as pp
import reranker
import vector_store
import verifier

warnings.filterwarnings("ignore", category=DeprecationWarning)  # noisy pydantic warnings inside litellm
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
for noisy in ("httpx", "LiteLLM", "litellm", "sentence_transformers", "pymongo"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load every local model once, before the first request.
    for name, loader in [("embedder", vector_store.load_embedder), ("reranker", reranker.load),
                         ("nli", verifier.load_nli)]:
        t = time.perf_counter()
        await asyncio.to_thread(loader)
        log.info("loaded %s in %.1fs", name, time.perf_counter() - t)
    if config.MEM0_API_KEY:
        try:
            await asyncio.to_thread(memory_service.client)  # the constructor makes a slow blocking ping
        except Exception as e:
            log.error("Mem0 client init failed (will retry on first use): %s", e)
    try:
        await db.init_indexes()
        await vector_store.ensure_collection()
    except Exception as e:
        log.error("database setup failed (is docker compose up?): %s", e)
    yield


app = FastAPI(title="Saaraansh (FactIndic)", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Internal error ({type(exc).__name__})."})


def to_json(value):
    """Make MongoDB documents JSON-safe (ObjectId -> str, datetime -> ISO)."""
    if isinstance(value, dict):
        return {("id" if k == "_id" else k): to_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json(v) for v in value]
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def object_id(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        raise HTTPException(400, "Invalid document id.")


# ---------------------------------------------------------------- summarize

class SummarizeBody(BaseModel):
    text: str = ""
    length: Literal["short", "medium", "long"] = "medium"
    style: Literal["paragraph", "bullets"] = "paragraph"
    compare: bool = False
    session_id: str = "anonymous"


@app.post("/api/summarize")
async def summarize(body: SummarizeBody):
    words = pp.word_count(body.text)
    if words == 0:
        raise HTTPException(400, "Please enter some text to summarize.")
    if words > config.MAX_WORDS:
        raise HTTPException(400, f"Text is too long ({words:,} words). The limit is {config.MAX_WORDS:,} words.")
    try:
        await db.ping()
    except Exception:
        raise HTTPException(503, "MongoDB is not reachable. Start it with: docker compose up -d")

    req = pipeline.Request(text=body.text, length=body.length, style=body.style,
                           compare=body.compare, session_id=body.session_id)

    async def events():
        async for event, data in pipeline.run(req):
            yield f"event: {event}\ndata: {json.dumps(to_json(data), ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- corrections / feedback

class CorrectionBody(BaseModel):
    session_id: str
    doc_id: str
    claim_text: str
    note: str = ""


@app.post("/api/corrections")
async def add_correction(body: CorrectionBody):
    if not body.claim_text.strip():
        raise HTTPException(400, "claim_text is required.")
    doc = {"session_id": body.session_id, "doc_id": body.doc_id, "claim_text": body.claim_text,
           "user_note": body.note, "created_at": datetime.now(timezone.utc)}
    res = await db.corrections.insert_one(doc)
    try:
        await memory_service.add_correction(body.session_id, body.claim_text, body.note, body.doc_id)
        saved_to_memory = True
    except Exception as e:
        log.error("Mem0 add failed: %s", e)
        saved_to_memory = False
    await db.corrections.update_one({"_id": res.inserted_id}, {"$set": {"in_mem0": saved_to_memory}})
    return {"id": str(res.inserted_id), "saved_to_memory": saved_to_memory}


class FeedbackBody(BaseModel):
    doc_id: str
    rating: Literal["up", "down"]


@app.post("/api/feedback")
async def feedback(body: FeedbackBody):
    await db.documents.update_one({"_id": object_id(body.doc_id)}, {"$set": {"feedback": body.rating}})
    return {"ok": True}


# ---------------------------------------------------------------- history

@app.get("/api/history")
async def history(session_id: str = Query(...)):
    cursor = db.documents.find(
        {"session_id": session_id, "final_attempt_id": {"$exists": True}},
        {"title": 1, "created_at": 1, "language_type": 1, "final_score": 1, "low_confidence": 1},
    ).sort("created_at", -1).limit(20)
    return [to_json(d) async for d in cursor]


@app.get("/api/history/{doc_id}")
async def history_item(doc_id: str):
    doc = await db.documents.find_one({"_id": object_id(doc_id)})
    if not doc or "final_attempt_id" not in doc:
        raise HTTPException(404, "Summary not found.")
    final = await db.summary_attempts.find_one({"_id": doc["final_attempt_id"]})
    result = pipeline.build_result(doc, final, doc["source_text"], doc.get("attempts", 1),
                                   doc.get("inference_time_ms", 0))
    result["options"] = doc.get("options", {})
    result["feedback"] = doc.get("feedback")
    if "compare_attempt_id" in doc:
        std = await db.summary_attempts.find_one({"_id": doc["compare_attempt_id"]})
        first = await db.summary_attempts.find_one({"doc_id": doc_id, "attempt_number": 1})
        result["compare"] = {
            "standard": {k: std[k] for k in ("summary", "factuality_score", "claim_score", "entity_score",
                                              "claims", "entity_mismatches")},
            "factuality_aware": {"summary": final["summary"], "factuality_score": final["factuality_score"],
                                 "first_attempt_score": first["factuality_score"]},
        }
    return to_json(result)


@app.delete("/api/history")
async def clear_history(session_id: str = Query(...)):
    ids = [d["_id"] async for d in db.documents.find({"session_id": session_id}, {"_id": 1})]
    str_ids = [str(i) for i in ids]
    await db.summary_attempts.delete_many({"doc_id": {"$in": str_ids}})
    await db.documents.delete_many({"_id": {"$in": ids}})
    for doc_id in str_ids:
        await vector_store.delete_document(doc_id)
    return {"deleted": len(ids)}


# ---------------------------------------------------------------- stats / health

@app.get("/api/stats")
async def stats():
    match = {"final_attempt_id": {"$exists": True}, "is_eval": {"$ne": True}}
    rows = await db.documents.aggregate([
        {"$match": match},
        {"$group": {"_id": None, "total": {"$sum": 1}, "avg_score": {"$avg": "$final_score"},
                    "avg_time": {"$avg": "$inference_time_ms"},
                    "code_mixed": {"$sum": {"$cond": [{"$eq": ["$language_type", "code_mixed"]}, 1, 0]}}}},
    ]).to_list(1)
    if not rows:
        return {"total_documents": 0, "avg_factuality_score": None, "avg_inference_time_ms": None,
                "code_mixed_documents": 0}
    r = rows[0]
    return {"total_documents": r["total"], "avg_factuality_score": round(r["avg_score"], 2),
            "avg_inference_time_ms": round(r["avg_time"]), "code_mixed_documents": r["code_mixed"]}


@app.get("/api/health")
async def health():
    async def check(fn):
        try:
            await asyncio.wait_for(fn(), timeout=5)
            return True
        except Exception as e:
            log.warning("health check failed: %s", e)
            return False

    mongo_ok, qdrant_ok = await asyncio.gather(check(db.ping), check(vector_store.ping))
    llm_ok = config.MODEL_NAME.startswith("ollama") or bool(config.OPENAI_API_KEY)
    models = {"embedder": vector_store.is_loaded(), "reranker": reranker.is_loaded(), "nli": verifier.is_loaded()}
    ok = mongo_ok and qdrant_ok and llm_ok and all(models.values())
    return {"status": "ok" if ok else "degraded", "mongo": mongo_ok, "qdrant": qdrant_ok,
            "llm": {"model": config.MODEL_NAME, "configured": llm_ok},
            "mem0": {"configured": bool(config.MEM0_API_KEY)},
            "models": models, "model_names": {"embed": config.EMBED_MODEL, "rerank": config.RERANK_MODEL,
                                              "nli": config.NLI_MODEL},
            "factuality_threshold": config.FACTUALITY_THRESHOLD, "max_retries": config.MAX_RETRIES}
