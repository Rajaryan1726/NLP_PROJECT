"""Qdrant vector store + sentence-transformer embeddings with a MongoDB embedding cache."""
import asyncio
import hashlib
import logging
import uuid

from pymongo.errors import BulkWriteError
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct, VectorParams

import config
import db

log = logging.getLogger("vector_store")
qdrant = AsyncQdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=20)
_embedder = None


def load_embedder():
    global _embedder
    from sentence_transformers import SentenceTransformer
    _embedder = SentenceTransformer(config.EMBED_MODEL, device="cpu")


def is_loaded() -> bool:
    return _embedder is not None


def _hash(text: str) -> str:
    return hashlib.sha1(f"{config.EMBED_MODEL}::{text}".encode()).hexdigest()


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts, reusing cached vectors so identical chunks are never embedded twice."""
    hashes = [_hash(t) for t in texts]
    cached = {d["_id"]: d["vector"] async for d in db.embedding_cache.find({"_id": {"$in": hashes}})}
    missing = [i for i, h in enumerate(hashes) if h not in cached]
    if missing:
        vectors = await asyncio.to_thread(
            _embedder.encode, [texts[i] for i in missing], normalize_embeddings=True, batch_size=16)
        docs = []
        for i, vec in zip(missing, vectors):
            cached[hashes[i]] = vec.tolist()
            docs.append({"_id": hashes[i], "vector": vec.tolist(), "model_name": config.EMBED_MODEL})
        try:
            await db.embedding_cache.insert_many(docs, ordered=False)
        except BulkWriteError:
            pass  # another request cached the same text at the same time
    log.info("embed: %d texts, %d from cache", len(texts), len(texts) - len(missing))
    return [cached[h] for h in hashes]


async def ensure_collection():
    if not await qdrant.collection_exists(config.QDRANT_COLLECTION):
        await qdrant.create_collection(
            config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE))
        await qdrant.create_payload_index(config.QDRANT_COLLECTION, "doc_id", PayloadSchemaType.KEYWORD)


async def index_chunks(doc_id: str, chunks: list[dict]):
    """Upsert chunks. Each chunk has text (original) and text_en (English, used for the vector)."""
    vectors = await embed([c["text_en"] for c in chunks])
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}/{c['chunk_id']}")),
            vector=vec,
            payload={"doc_id": doc_id, "chunk_id": c["chunk_id"], "text": c["text"],
                     "text_en": c["text_en"], "start": c["start"], "end": c["end"]},
        )
        for c, vec in zip(chunks, vectors)
    ]
    await qdrant.upsert(config.QDRANT_COLLECTION, points=points, wait=True)


async def search_many(doc_id: str, queries: list[str], top_k: int = config.RETRIEVE_TOP_K) -> list[list[dict]]:
    """Top-k chunks of ONE document (always filtered by doc_id) for each query."""
    if not queries:
        return []
    vectors = await embed(queries)
    doc_filter = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    responses = await asyncio.gather(*(
        qdrant.query_points(config.QDRANT_COLLECTION, query=v, query_filter=doc_filter,
                            limit=top_k, with_payload=True)
        for v in vectors))
    return [[{**p.payload, "vector_score": p.score} for p in r.points] for r in responses]


async def delete_document(doc_id: str):
    await qdrant.delete(config.QDRANT_COLLECTION, points_selector=Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]))


async def ping() -> bool:
    await qdrant.get_collections()
    return True
