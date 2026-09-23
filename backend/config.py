"""All settings come from backend/.env."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")  # also puts OPENAI_API_KEY in os.environ for LiteLLM

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "factindic")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6335")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = "factindic_chunks"

MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
EMBED_DIM = 768

FACTUALITY_THRESHOLD = float(os.getenv("FACTUALITY_THRESHOLD", "7"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.05"))

MAX_WORDS = 5000
RETRIEVE_TOP_K = 10
RERANK_TOP_K = 3

# Seconds without a response before an LLM request is retried. Kept short because an idle
# keep-alive connection can be silently dropped and would otherwise hang for minutes.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))
