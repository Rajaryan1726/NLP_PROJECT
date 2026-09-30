"""Download the three local models once (needs internet). After this the backend can start offline.

Usage (from backend/):  .venv\\Scripts\\python scripts\\download_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reranker  # noqa: E402
import vector_store  # noqa: E402
import verifier  # noqa: E402

for name, load in [("embedder", vector_store.load_embedder), ("reranker", reranker.load), ("NLI", verifier.load_nli)]:
    print(f"downloading / loading {name}...", flush=True)
    load()
print("all models are downloaded")
