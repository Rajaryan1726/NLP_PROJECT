# Saaraansh (FactIndic)

**Factuality-aware summarization for Hindi and Hindi-English code-mixed text.**
Course project for CSET 346 (NLP), Project 6: *FactIndic: Factuality-Aware Summarization for Indic and Code-Mixed Languages*.

Saaraansh takes Hindi (Devanagari), English or code-mixed text, writes a **Hinglish** summary (Hindi in Roman
script) with an LLM, and then **checks every claim of the summary against the source**: retrieval, reranking,
NLI, and entity/number/date matching. If the factuality score is too low, it regenerates the summary with a
corrective prompt. Corrections the user makes are remembered (Mem0) and used in later prompts.

---

## 1. Quick start (Windows, PowerShell)

Requirements: Python 3.11+ (tested on 3.13), Node 20+, Docker Desktop.

```powershell
# 1. databases (MongoDB + Qdrant), from the project folder
cd "C:\NLP PROJECT"
docker compose up -d

# 2. backend
cd backend
python -m venv .venv
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env        # then put your OPENAI_API_KEY and MEM0_API_KEY in .env
.\.venv\Scripts\python -m uvicorn main:app --port 8000

# 3. frontend (second terminal)
cd "C:\NLP PROJECT\frontend"
npm install
npm run dev                   # http://localhost:5174
```

The first backend start downloads the three local models (~2 GB) from HuggingFace. After that you can set
`$env:HF_HUB_OFFLINE="1"` before starting uvicorn so startup doesn't make any network checks.

Check everything is up: <http://localhost:8000/api/health> should report `"status": "ok"`.

Tests: `cd backend; .\.venv\Scripts\python -m pytest tests -q` (the Mem0 test needs internet + `MEM0_API_KEY`).

> **Ports.** Qdrant is mapped to host port **6335** and the frontend runs on **5174**, because 6333 and 5173
> were already taken by other containers on the development machine. Change them in `docker-compose.yml`,
> `backend/.env` (`QDRANT_URL`) and `frontend/vite.config.ts` if you like.

### Switch to a local Llama (Ollama)

All LLM calls go through LiteLLM (`backend/llm_service.py`), so the only change is one line in `backend/.env`:

```
MODEL_NAME=ollama/llama3
```

(Install Ollama, run `ollama pull llama3`, keep it running on the default port 11434. No OpenAI key is needed then.)

---

## 2. How it works

```
text ─► preprocess ─► chunk + translate chunks to English ─► embed ─► Qdrant (filtered by doc_id)
          │                                                              ▲
          ├─► Mem0: past corrections for this session ─┐                 │ top-10
          ▼                                            ▼                 │
      factuality-aware prompt (list facts, then write) ──► Hinglish summary (streamed)
                                                             │
                     atomic claims (+ English) ◄─────────────┘
                              │
          retrieve top-10 ──► cross-encoder rerank top-3 ──► NLI (entail / neutral / contradict)
                              │           (abstain if best rerank score < MIN_RERANK_SCORE)
          numbers / dates / entities ──► normalize (2 crore = 20,000,000, अगस्त = August) + match
                              │
       score = 10 × (0.6 × claim_score + 0.4 × entity_score)
                              │
         score < 7 ? ──► corrective prompt with the failed claims + their evidence (max 2 retries)
                              │
                         MongoDB (documents, attempts) ──► UI
```

| Step | File | Details |
|---|---|---|
| Preprocess | `preprocessing.py` | NFC, Devanagari digits → ASCII, whitespace; language = `hindi` / `english` / `code_mixed` from script ratio + romanized-Hindi function words |
| Chunking | `preprocessing.py` | 2-3 sentences per chunk (split on । . ? ! and newlines, decimals kept), character offsets kept for highlighting |
| Language normalization | `pipeline.index_document` | Chunks are translated to English once (cached in MongoDB). Claims come with an English version from the decomposition call. Multilingual retrievers and NLI models work poorly across scripts and on romanized Hindi, so retrieval, reranking and NLI all compare **English with English**. |
| Retrieval | `vector_store.py` | `paraphrase-multilingual-mpnet-base-v2` (768-d), Qdrant collection `factindic_chunks`, every query filtered by `doc_id`, embedding cache (`text_hash → vector`) in MongoDB |
| Reranking | `reranker.py` | cross-encoder, top-10 → top-3, sigmoid scores in [0, 1], all claims in one batch |
| NLI | `verifier.py` | `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, premise = each top-3 chunk and all three joined; any entailment wins, else contradiction, else neutral |
| Entity matching | `verifier.match_facts` | numbers & dates by regex + normalization (crore/lakh/million/हजार, Devanagari digits, Hindi/English/romanized month names); the LLM only extracts the proper names from the summary; whether each is in the source is checked in code against the source and its English translation (normalized, with fuzzy spelling) |
| Score | `verifier.compute_scores` | `claim_score` = entailed / claims, `entity_score` = matched / facts, factuality = 10 × (0.6·claim + 0.4·entity) |
| Retry | `pipeline.run` | below `FACTUALITY_THRESHOLD` (7): corrective prompt, at most `MAX_RETRIES` (2); the best attempt is returned with `low_confidence=true` if it still fails. Leftover Devanagari in the summary also triggers a retry. |
| Memory | `memory_service.py` | "Mark wrong" in the UI → `POST /api/corrections` → MongoDB + Mem0 `add()`. Before generating, Mem0 `search()` (mem0ai 2.x needs `filters={"user_id": ...}`) and the hits go into the prompt as soft rules |

### Verdicts shown in the UI

| Backend | UI chip | Meaning |
|---|---|---|
| `entailed` | Supported | NLI says the evidence entails the claim |
| `neutral` | Not sure | evidence found, but it neither entails nor contradicts the claim |
| `contradicted` | Contradicted | the evidence contradicts the claim |
| `unsupported` | Unsupported | nothing in the source is relevant enough (abstained without running NLI) |

### Design decisions worth explaining

* **Reranker.** The spec's first choice, `BAAI/bge-reranker-v2-m3`, took 6.3 s per 14 pairs on this CPU
  (no GPU), about 36 s for a long document. `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` needs 0.4 s for the
  same pairs and ranked our English-normalized test pairs identically, so it is the default. Use bge by
  changing `RERANK_MODEL` in `.env` if you have a GPU.
* **NLI in float32.** The mDeBERTa checkpoint is stored in float16. On CPU, fp16 is ~10× slower
  (41 s → 4 s for 21 pairs), so it is loaded as float32.
* **Batching.** All claims of a summary go through the reranker and the NLI model in one batch each
  (34 s → 1.6 s compared to one call per claim).
* **LLM connections.** LiteLLM uses its httpx transport and a 30 s timeout. The default aiohttp pool keeps idle
  connections for 120 s; after the app sat idle, a request on a connection the router had silently dropped
  hung for minutes.
* **Windows Smart App Control** blocks the DLLs of very new releases of scipy, scikit-learn, pandas and
  jiter. `requirements.txt` pins versions that load.

---

## 3. API

| Method | Path | |
|---|---|---|
| POST | `/api/summarize` | `{text, length: short\|medium\|long, style: paragraph\|bullets, compare, session_id}` → Server-Sent Events: `status`, `token`, `attempt`, `result`, `compare_result`, `error` |
| POST | `/api/corrections` | `{session_id, doc_id, claim_text, note}` → MongoDB + Mem0 |
| POST | `/api/feedback` | `{doc_id, rating: up\|down}` (thumbs in the UI) |
| GET | `/api/history?session_id=` | last 20 documents |
| GET | `/api/history/{doc_id}` | full saved result |
| DELETE | `/api/history?session_id=` | clear history (MongoDB + Qdrant) |
| GET | `/api/stats` | documents summarized, average factuality, average inference time, code-mixed count (evaluation runs excluded) |
| GET | `/api/health` | MongoDB, Qdrant, LLM config, models loaded |

Empty text → 400; more than 5,000 words → 400; LLM or database errors → a clear message, never a stack trace.

There is no login: the frontend creates a random `session_id`, keeps it in localStorage, and sends it with
every request; it is also the Mem0 `user_id`.

MongoDB collections: `documents`, `summary_attempts`, `corrections`, `embedding_cache`, `translation_cache`,
`evaluation_runs`.

---

## 4. Evaluation (for the paper)

All scripts are in `backend/eval/` and are run from `backend/`:

```powershell
cd "C:\NLP PROJECT\backend"
.\.venv\Scripts\python eval\prepare_data.py --n-hindi 100 --n-code-mixed 50   # datasets
.\.venv\Scripts\python eval\run_experiments.py --limit 5                       # smoke test
.\.venv\Scripts\python eval\run_experiments.py                                 # full run (resumable)
.\.venv\Scripts\python eval\analyze.py                                         # tables + charts
.\.venv\Scripts\python eval\make_label_template.py --n 30                      # labelling sheet
#   ... fill eval\labeling\claims_to_label.csv (human_supported = 1 / 0) ...
.\.venv\Scripts\python eval\score_labels.py                                    # P / R / F1
```

**Datasets** (`eval/data/`)
* `hindi.jsonl`: 100 random articles (seed 42) from the XL-Sum **Hindi test split** (`csebuetnlp/xlsum`), 150-1,200 words, with the human reference summaries.
* `code_mixed.jsonl`: **SYNTHETIC.** The first 50 of those articles rewritten by the LLM into romanized
  Hindi-English code-mixed text, with a strict prompt that must keep every name, number and date. Rewrites
  that lose a number or contain Devanagari are rejected. *This is a limitation to state in the paper:* the
  code-mixed text is machine-generated, so it is cleaner and more regular than real social-media code-mixing,
  and it was produced by the same model family that summarizes it.

**Experiments**: `standard` (plain prompt), `factuality_aware` (no retry), `factuality_aware_retry`
(retry loop), each on both datasets. Mem0 is off during evaluation. LLM outputs are cached in
`eval/cache/llm` (one namespace per experiment, so experiments never share samples), and finished items are
skipped when the script is re-run. Only runs without cache hits count towards the inference time.
Use `--concurrency 1` (the default) for the timing numbers you report.

**Outputs** (`eval/results/`)
* `summary_table.csv/.json/.tex`: factuality score, claim score, entity score, verdict distribution, fact
  and number mismatch rate, summary length, compression ratio, inference time, attempts, low-confidence rate,
  ROUGE-1/2/L.
* `charts/*.png` (300 dpi, plus PDF for LaTeX): `prompt_comparison`, `hindi_vs_code_mixed`,
  `score_distribution`, `retry_effect`, `error_categories` (number mismatch / entity swap / negation drop /
  added info / code-mix misread / verifier false alarm, classified by the LLM and saved in `error_categories.csv`).
* `label_scores.json`: precision / recall / F1 of the verifier against your labels (positive class = factual
  error), plus the human-judged factual precision of each system.

**ROUGE caveat.** XL-Sum references are Devanagari Hindi and our summaries are romanized Hinglish.
References are transliterated (ITRANS + word-final schwa deletion) and both sides are spelling-normalized,
but spelling variation (e.g. *sarkaar* / *sarkar*) still lowers ROUGE. Only compare our systems with each
other; do not compare with published XL-Sum ROUGE numbers.

---

## 5. Project structure

```
docker-compose.yml     MongoDB + Qdrant (restart: unless-stopped)
backend/
  main.py              FastAPI routes
  config.py            settings from .env
  llm_service.py       every LiteLLM call (generate, stream, decompose, translate, extract)
  prompts.py           all prompt templates
  preprocessing.py     cleaning, language detection, chunking, number/date normalization
  vector_store.py      Qdrant + embeddings + embedding cache
  reranker.py          cross-encoder
  verifier.py          claims, NLI, entity matching, scoring
  pipeline.py          the full flow + corrective retry loop
  memory_service.py    Mem0
  db.py                MongoDB
  scripts/try_summarize.py   call the API from the terminal
  eval/                evaluation scripts, data, results
  tests/               pytest: preprocessing, numbers/dates, scoring, Mem0 round trip
frontend/              Vite + React + TypeScript + Tailwind v4 + shadcn/ui + AI Elements
  src/App.tsx          state + wiring
  src/lib/api.ts       REST + SSE client
  src/components/      TopBar, StatCards, HistoryPanel, InputPanel, OutputPanel, ClaimList, ScoreBadge
```

## 6. Demo texts

`demo_texts/` has one input per feature: `1_hindi_news.txt` (Devanagari, numbers + dates),
`2_code_mixed.txt` (the design's Vidya Setu example), `3_numbers_and_negation.txt` (negations that must
survive), `4_noisy_code_switched.txt` (chatty, noisy code-switching), `5_english.txt`.
From the terminal: `.\.venv\Scripts\python scripts\try_summarize.py ..\demo_texts\2_code_mixed.txt --compare`.

## 7. Limitations

* The code-mixed evaluation set is synthetic (see above).
* Translation to English is part of the verifier, so a translation error can cause a wrong verdict.
* Entity presence is string matching against the source and its English translation, so a correct name
  that the translation spelled very differently can be flagged. Numbers and dates are checked deterministically.
* NLI sometimes calls a claim "contradicted" when it only adds an implicit agent (source: "messages were
  posted from the account", claim: "hackers posted messages"). The hand-labelled precision/recall and the
  "verifier false alarm" error category measure how often this happens.
* NLI on long, multi-sentence premises is less reliable; the verifier uses 2-3 sentence chunks for that reason.
* Scores are relative to the source only; the system never checks world knowledge.
