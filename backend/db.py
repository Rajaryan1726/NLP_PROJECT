"""MongoDB (motor async driver)."""
from motor.motor_asyncio import AsyncIOMotorClient

import config

client = AsyncIOMotorClient(config.MONGO_URI, serverSelectionTimeoutMS=4000, tz_aware=True)
db = client[config.MONGO_DB]

documents = db.documents
summary_attempts = db.summary_attempts
corrections = db.corrections
embedding_cache = db.embedding_cache
translation_cache = db.translation_cache
evaluation_runs = db.evaluation_runs


async def ping() -> bool:
    await client.admin.command("ping")
    return True


async def init_indexes():
    await documents.create_index([("session_id", 1), ("created_at", -1)])
    await summary_attempts.create_index([("doc_id", 1), ("attempt_number", 1)])
    await corrections.create_index("session_id")
