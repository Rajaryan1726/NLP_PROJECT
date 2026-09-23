"""Shared paths and helpers for the evaluation scripts.

Import this module BEFORE any backend module: it turns on the LLM disk cache
(eval/cache/llm) so that re-running an experiment costs nothing.
"""
import json
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = EVAL_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw"
CHARTS_DIR = RESULTS_DIR / "charts"
LABEL_DIR = EVAL_DIR / "labeling"
CACHE_DIR = EVAL_DIR / "cache"

os.environ.setdefault("LLM_CACHE_DIR", str(CACHE_DIR / "llm"))
sys.path.insert(0, str(BACKEND_DIR))
sys.stdout.reconfigure(encoding="utf-8")

for d in (DATA_DIR, RAW_DIR, CHARTS_DIR, LABEL_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "hindi": DATA_DIR / "hindi.jsonl",            # XL-Sum Hindi (Devanagari), real
    "code_mixed": DATA_DIR / "code_mixed.jsonl",  # synthetic code-mixed versions of the same articles
}

# name -> (prompt_type, max_retries)
EXPERIMENTS = {
    "standard": ("standard", 0),
    "factuality_aware": ("factuality_aware", 0),
    "factuality_aware_retry": ("factuality_aware", 2),
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def raw_path(experiment: str, dataset: str) -> Path:
    return RAW_DIR / f"{experiment}__{dataset}.jsonl"
